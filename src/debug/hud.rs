use std::collections::VecDeque;

use bevy::ecs::hierarchy::ChildSpawnerCommands;
use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::ui::widget::{Text, TextUiWriter};
use bevy::ui::{
    AlignItems, BackgroundColor, Display, FlexDirection, JustifyContent, Node, PositionType,
    UiRect, Val,
};

use crate::agent::observation::SensorReadings;
use crate::brain::a2c::A2cTrainingStats;
use crate::debug::overlays::DebugOverlayState;
use crate::game::car::Car;
use crate::game::collision::CollisionEvent;
use crate::game::episode::{EpisodeConfig, EpisodeEndReason, EpisodeMovingAverages, EpisodeState};
use crate::game::progress::TrackProgress;

const HUD_QUARTER_COUNT: usize = 4;
const FIXED_TICK_SECONDS: f32 = 1.0 / 60.0;

/// Runtime HUD state that tracks deaths and the best observed progress.
#[derive(Resource, Debug)]
pub struct DrivingHudStats {
    pub deaths: u32,
    pub best_progress_fraction: f32,
    pub best_progress_episode: u32,
}

impl Default for DrivingHudStats {
    fn default() -> Self {
        Self {
            deaths: 0,
            best_progress_fraction: 0.0,
            best_progress_episode: 1,
        }
    }
}

/// Accumulates one episode's centreline-following metrics before snapshotting them into history.
#[derive(Resource, Debug)]
pub struct DrivingHudEpisodeAccumulator {
    episode_id: u32,
    tick_count: u32,
    centreline_distance_sum: f32,
    abs_heading_error_sum_deg: f32,
}

impl Default for DrivingHudEpisodeAccumulator {
    fn default() -> Self {
        Self {
            episode_id: 1,
            tick_count: 0,
            centreline_distance_sum: 0.0,
            abs_heading_error_sum_deg: 0.0,
        }
    }
}

impl DrivingHudEpisodeAccumulator {
    fn reset_for_episode(&mut self, episode_id: u32) {
        self.episode_id = episode_id;
        self.tick_count = 0;
        self.centreline_distance_sum = 0.0;
        self.abs_heading_error_sum_deg = 0.0;
    }

    fn record_tick(&mut self, episode_state: &EpisodeState) {
        self.tick_count = self.tick_count.saturating_add(1);
        self.centreline_distance_sum += episode_state.current_tick_centerline_distance;
        self.abs_heading_error_sum_deg +=
            episode_state.current_tick_heading_error.abs().to_degrees();
    }
}

/// Rolling debug-only history used to split recent episodes into four real-time quarters.
#[derive(Resource, Debug, Default)]
pub struct DrivingHudHistory {
    episodes: VecDeque<CompletedHudEpisode>,
}

#[derive(Clone, Copy, Debug)]
struct CompletedHudEpisode {
    end_reason: EpisodeEndReason,
    best_progress_fraction: f32,
    total_return: f32,
    life_seconds: f32,
    mean_centreline_distance: f32,
    mean_abs_heading_error_deg: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct QuarterSummary {
    count: usize,
    crash_count: usize,
    lap_count: usize,
    timeout_count: usize,
    mean_progress_pct: f32,
    mean_return: f32,
    mean_life_seconds: f32,
    mean_centreline_distance: f32,
    mean_abs_heading_error_deg: f32,
}

#[derive(Component)]
pub(crate) struct DrivingHudRoot;

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum HudTextRole {
    Assessment,
    Current,
    Run,
    Learning,
    Legend,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct QuarterCell {
    row: usize,
    column: QuarterColumn,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QuarterColumn {
    Quarter,
    Count,
    Gap,
    Heading,
    Progress,
    Life,
    Return,
    Ends,
}

fn quarter_column_width(column: QuarterColumn) -> f32 {
    match column {
        QuarterColumn::Quarter => 30.0,
        QuarterColumn::Count => 28.0,
        QuarterColumn::Gap => 52.0,
        QuarterColumn::Heading => 58.0,
        QuarterColumn::Progress => 54.0,
        QuarterColumn::Life => 48.0,
        QuarterColumn::Return => 58.0,
        QuarterColumn::Ends => 88.0,
    }
}

/// Spawns the runtime diagnostics HUD used by `F3`.
pub(crate) fn spawn_driving_hud_system(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(12.0),
                left: Val::Px(12.0),
                width: Val::Px(620.0),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(12.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(6.0),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.09, 0.11, 0.91)),
            DrivingHudRoot,
        ))
        .with_children(|parent| {
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(4.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.19, 0.69, 0.61)),
            ));

            parent.spawn((
                Text::new("Run Diagnostics  |  F1 geometry  |  F2 sensors  |  F3 panel"),
                TextFont::from_font_size(16.0),
                TextColor(Color::srgb(0.95, 0.98, 0.97)),
            ));

            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(13.0),
                TextColor(Color::srgb(0.61, 0.87, 0.80)),
                HudTextRole::Assessment,
            ));
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.90, 0.94, 0.93)),
                HudTextRole::Current,
            ));
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.90, 0.94, 0.93)),
                HudTextRole::Run,
            ));
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.80, 0.88, 0.87)),
                HudTextRole::Learning,
            ));

            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    ..default()
                },
                BackgroundColor(Color::srgba(0.72, 0.83, 0.82, 0.16)),
            ));

            parent.spawn((
                Text::new("Recent quarters (oldest -> newest)"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.93, 0.96, 0.95)),
            ));

            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(3.0),
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.03, 0.05, 0.06, 0.34)),
                ))
                .with_children(|table| {
                    let spawn_cell =
                        |row: &mut ChildSpawnerCommands<'_>, label: &str, width: f32, bg: Color| {
                            row.spawn((
                                Node {
                                    width: Val::Px(width),
                                    padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                                    justify_content: JustifyContent::FlexStart,
                                    align_items: AlignItems::Center,
                                    ..default()
                                },
                                BackgroundColor(bg),
                            ))
                            .with_children(
                                |cell: &mut ChildSpawnerCommands<'_>| {
                                    cell.spawn((
                                        Text::new(label),
                                        TextFont::from_font_size(10.5),
                                        TextColor(Color::srgb(0.95, 0.98, 0.97)),
                                    ));
                                },
                            );
                        };

                    table
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_direction: FlexDirection::Row,
                                column_gap: Val::Px(4.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgba(0.08, 0.13, 0.15, 0.82)),
                        ))
                        .with_children(|row| {
                            spawn_cell(
                                row,
                                "Q",
                                quarter_column_width(QuarterColumn::Quarter),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "N",
                                quarter_column_width(QuarterColumn::Count),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "Gap",
                                quarter_column_width(QuarterColumn::Gap),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "Head",
                                quarter_column_width(QuarterColumn::Heading),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "Prog",
                                quarter_column_width(QuarterColumn::Progress),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "Life",
                                quarter_column_width(QuarterColumn::Life),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "Return",
                                quarter_column_width(QuarterColumn::Return),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                            spawn_cell(
                                row,
                                "C/L/T",
                                quarter_column_width(QuarterColumn::Ends),
                                Color::srgba(0.10, 0.18, 0.20, 0.92),
                            );
                        });

                    for row_index in 0..HUD_QUARTER_COUNT {
                        table
                            .spawn((
                                Node {
                                    width: Val::Percent(100.0),
                                    flex_direction: FlexDirection::Row,
                                    column_gap: Val::Px(4.0),
                                    ..default()
                                },
                                BackgroundColor(Color::srgba(0.03, 0.08, 0.10, 0.55)),
                            ))
                            .with_children(|row| {
                                for column in [
                                    QuarterColumn::Quarter,
                                    QuarterColumn::Count,
                                    QuarterColumn::Gap,
                                    QuarterColumn::Heading,
                                    QuarterColumn::Progress,
                                    QuarterColumn::Life,
                                    QuarterColumn::Return,
                                    QuarterColumn::Ends,
                                ] {
                                    row.spawn((
                                        Node {
                                            width: Val::Px(quarter_column_width(column)),
                                            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                                            justify_content: JustifyContent::FlexStart,
                                            align_items: AlignItems::Center,
                                            ..default()
                                        },
                                        BackgroundColor(Color::srgba(0.05, 0.11, 0.13, 0.82)),
                                    ))
                                    .with_children(|cell| {
                                        cell.spawn((
                                            Text::new(""),
                                            TextFont::from_font_size(10.5),
                                            TextColor(Color::srgb(0.90, 0.94, 0.93)),
                                            QuarterCell {
                                                row: row_index,
                                                column,
                                            },
                                        ));
                                    });
                                }
                            });
                    }
                });

            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(10.5),
                TextColor(Color::srgb(0.61, 0.78, 0.76)),
                HudTextRole::Legend,
            ));
        });
}

/// Tracks live death count and the best progress reached in any episode so far.
pub(crate) fn update_driving_hud_stats_system(
    mut hud_stats: ResMut<DrivingHudStats>,
    mut collision_events: MessageReader<CollisionEvent>,
    episode_state: Res<EpisodeState>,
    progress_query: Query<&TrackProgress, With<Car>>,
) {
    for _ in collision_events.read() {
        hud_stats.deaths = hud_stats.deaths.saturating_add(1);
    }

    let Ok(progress) = progress_query.single() else {
        return;
    };

    if progress.fraction > hud_stats.best_progress_fraction {
        hud_stats.best_progress_fraction = progress.fraction;
        hud_stats.best_progress_episode = episode_state.current_episode;
    }
}

/// Captures per-tick centreline-following metrics and snapshots one summary per completed episode.
pub(crate) fn capture_driving_hud_episode_metrics_system(
    config: Res<EpisodeConfig>,
    episode_state: Res<EpisodeState>,
    mut accumulator: ResMut<DrivingHudEpisodeAccumulator>,
    mut history: ResMut<DrivingHudHistory>,
) {
    let finished_episode_id = episode_state.current_episode.saturating_sub(1);
    let target_episode_id = if episode_state.current_tick_end_reason.is_some() {
        finished_episode_id
    } else {
        episode_state.current_episode
    };

    if accumulator.episode_id != target_episode_id {
        accumulator.reset_for_episode(target_episode_id);
    }

    accumulator.record_tick(&episode_state);

    let Some(end_reason) = episode_state.current_tick_end_reason else {
        return;
    };

    let tick_count = accumulator.tick_count.max(1) as f32;
    history.episodes.push_back(CompletedHudEpisode {
        end_reason,
        best_progress_fraction: episode_state.last_episode_best_progress_fraction,
        total_return: episode_state.last_episode_return,
        life_seconds: episode_state.last_episode_ticks as f32 * FIXED_TICK_SECONDS,
        mean_centreline_distance: accumulator.centreline_distance_sum / tick_count,
        mean_abs_heading_error_deg: accumulator.abs_heading_error_sum_deg / tick_count,
    });
    while history.episodes.len() > config.moving_average_window.max(1) {
        let _ = history.episodes.pop_front();
    }

    accumulator.reset_for_episode(episode_state.current_episode);
}

/// Shows or hides the diagnostics panel according to the `F3` toggle.
pub(crate) fn update_driving_hud_visibility_system(
    overlay: Res<DebugOverlayState>,
    mut root_query: Query<&mut Node, With<DrivingHudRoot>>,
) {
    let Ok(mut node) = root_query.single_mut() else {
        return;
    };

    node.display = if overlay.telemetry {
        Display::Flex
    } else {
        Display::None
    };
}

/// Rebuilds the diagnostics text and quarter grid shown in the HUD.
pub(crate) fn update_driving_hud_text_system(
    overlay: Res<DebugOverlayState>,
    hud_stats: Res<DrivingHudStats>,
    history: Res<DrivingHudHistory>,
    episode_state: Res<EpisodeState>,
    moving_avg: Res<EpisodeMovingAverages>,
    a2c_stats: Option<Res<A2cTrainingStats>>,
    car_query: Query<(&TrackProgress, &SensorReadings), With<Car>>,
    summary_query: Query<(Entity, &HudTextRole)>,
    quarter_query: Query<(Entity, &QuarterCell)>,
    mut text_writer: TextUiWriter,
) {
    if !overlay.telemetry {
        return;
    }

    let Ok((progress, sensors)) = car_query.single() else {
        return;
    };

    let progress_pct = (progress.fraction * 100.0).clamp(0.0, 100.0);
    let best_progress_pct = (hud_stats.best_progress_fraction * 100.0).clamp(0.0, 100.0);
    let life_best_progress_pct =
        (episode_state.current_best_progress_fraction * 100.0).clamp(0.0, 100.0);
    let current_life_seconds = episode_state.ticks_in_episode as f32 * FIXED_TICK_SECONDS;
    let heading_error_deg = sensors.heading_error.to_degrees();
    let avg_progress_pct = (moving_avg.best_progress_mean * 100.0).clamp(0.0, 100.0);
    let last_reason = match episode_state.last_end_reason {
        Some(EpisodeEndReason::Crash) => "Crash",
        Some(EpisodeEndReason::Timeout) => "Timeout",
        Some(EpisodeEndReason::LapComplete) => "Lap",
        None => "N/A",
    };
    let recent_quarters = summarise_recent_history(&history);
    let (assessment, guidance) = assess_recent_run(&recent_quarters);

    let current_line = format!(
        "Now  progress {progress_pct:5.2}%  life-best {life_best_progress_pct:5.2}%  offset {offset:+6.2}  line-gap {line_gap:5.2}  heading {heading_error_deg:5.2} deg",
        offset = sensors.signed_lateral_offset,
        line_gap = progress.distance,
    );
    let run_line = format!(
        "Run  ep {}  deaths {}  life {:5.2}s  reward {:+7.2}  last {}  best {:5.2}% @ ep {}  recent avg {:5.2}% / {:+6.2}",
        episode_state.current_episode,
        hud_stats.deaths,
        current_life_seconds,
        episode_state.current_return,
        last_reason,
        best_progress_pct,
        hud_stats.best_progress_episode,
        avg_progress_pct,
        moving_avg.return_mean,
    );
    let learning_line = match a2c_stats {
        Some(stats) if stats.last_completed_update > 0 => format!(
            "A2C  upd {}  EV {:5.3}  Vloss {:5.3}  Ent {:5.3}  steer std {:5.3}  throttle std {:5.3}",
            stats.last_completed_update,
            stats.explained_variance,
            stats.value_loss,
            stats.policy_entropy,
            stats.steering_std,
            stats.throttle_std,
        ),
        _ => "A2C  no completed updates yet".to_string(),
    };
    let legend_line =
        "Lower Gap/Head is better. Higher Prog/Life/Return is better. C/L/T = crashes/laps/timeouts."
            .to_string();

    for (entity, role) in &summary_query {
        let text = match role {
            HudTextRole::Assessment => format!("Status  {assessment}  |  {guidance}"),
            HudTextRole::Current => current_line.clone(),
            HudTextRole::Run => run_line.clone(),
            HudTextRole::Learning => learning_line.clone(),
            HudTextRole::Legend => legend_line.clone(),
        };
        *text_writer.text(entity, 0) = text;
    }

    for (entity, cell) in &quarter_query {
        let text = render_quarter_cell(recent_quarters[cell.row], cell.column, cell.row);
        *text_writer.text(entity, 0) = text;
    }
}

fn summarise_recent_history(history: &DrivingHudHistory) -> [QuarterSummary; HUD_QUARTER_COUNT] {
    let recent: Vec<_> = history.episodes.iter().copied().collect();
    let total = recent.len();
    let mut quarters = [QuarterSummary::default(); HUD_QUARTER_COUNT];

    for (quarter_index, quarter) in quarters.iter_mut().enumerate() {
        let start = total * quarter_index / HUD_QUARTER_COUNT;
        let end = total * (quarter_index + 1) / HUD_QUARTER_COUNT;
        if start >= end {
            continue;
        }

        let slice = &recent[start..end];
        quarter.count = slice.len();
        for episode in slice {
            match episode.end_reason {
                EpisodeEndReason::Crash => quarter.crash_count += 1,
                EpisodeEndReason::LapComplete => quarter.lap_count += 1,
                EpisodeEndReason::Timeout => quarter.timeout_count += 1,
            }
            quarter.mean_progress_pct += episode.best_progress_fraction * 100.0;
            quarter.mean_return += episode.total_return;
            quarter.mean_life_seconds += episode.life_seconds;
            quarter.mean_centreline_distance += episode.mean_centreline_distance;
            quarter.mean_abs_heading_error_deg += episode.mean_abs_heading_error_deg;
        }

        let count = quarter.count as f32;
        quarter.mean_progress_pct /= count;
        quarter.mean_return /= count;
        quarter.mean_life_seconds /= count;
        quarter.mean_centreline_distance /= count;
        quarter.mean_abs_heading_error_deg /= count;
    }

    quarters
}

fn render_quarter_cell(
    quarter: QuarterSummary,
    column: QuarterColumn,
    quarter_index: usize,
) -> String {
    if quarter.count == 0 {
        return match column {
            QuarterColumn::Quarter => format!("Q{}", quarter_index + 1),
            QuarterColumn::Count => "-".to_string(),
            QuarterColumn::Gap => "-".to_string(),
            QuarterColumn::Heading => "-".to_string(),
            QuarterColumn::Progress => "-".to_string(),
            QuarterColumn::Life => "-".to_string(),
            QuarterColumn::Return => "-".to_string(),
            QuarterColumn::Ends => "collect".to_string(),
        };
    }

    match column {
        QuarterColumn::Quarter => format!("Q{}", quarter_index + 1),
        QuarterColumn::Count => format!("{}", quarter.count),
        QuarterColumn::Gap => format!("{:.2}", quarter.mean_centreline_distance),
        QuarterColumn::Heading => format!("{:.1}", quarter.mean_abs_heading_error_deg),
        QuarterColumn::Progress => format!("{:.1}%", quarter.mean_progress_pct),
        QuarterColumn::Life => format!("{:.2}s", quarter.mean_life_seconds),
        QuarterColumn::Return => format!("{:+.2}", quarter.mean_return),
        QuarterColumn::Ends => format!(
            "{}/{}/{}",
            quarter.crash_count, quarter.lap_count, quarter.timeout_count
        ),
    }
}

fn assess_recent_run(
    quarters: &[QuarterSummary; HUD_QUARTER_COUNT],
) -> (&'static str, &'static str) {
    let populated: Vec<_> = quarters
        .iter()
        .copied()
        .filter(|quarter| quarter.count > 0)
        .collect();
    if populated.len() < 2 {
        return ("Warm-up", "too little data to judge yet");
    }

    let first = populated.first().copied().unwrap_or_default();
    let last = populated.last().copied().unwrap_or_default();
    let mut score = 0i32;

    if first.mean_centreline_distance - last.mean_centreline_distance >= 2.0 {
        score += 1;
    } else if last.mean_centreline_distance - first.mean_centreline_distance >= 2.0 {
        score -= 1;
    }

    if first.mean_abs_heading_error_deg - last.mean_abs_heading_error_deg >= 4.0 {
        score += 1;
    } else if last.mean_abs_heading_error_deg - first.mean_abs_heading_error_deg >= 4.0 {
        score -= 1;
    }

    if last.mean_progress_pct - first.mean_progress_pct >= 3.0 {
        score += 1;
    } else if first.mean_progress_pct - last.mean_progress_pct >= 3.0 {
        score -= 1;
    }

    if last.mean_life_seconds - first.mean_life_seconds >= 0.75 {
        score += 1;
    } else if first.mean_life_seconds - last.mean_life_seconds >= 0.75 {
        score -= 1;
    }

    if last.mean_return - first.mean_return >= 0.75 {
        score += 1;
    } else if first.mean_return - last.mean_return >= 0.75 {
        score -= 1;
    }

    if score >= 3 {
        ("Improving", "recent quarter is cleaner, worth continuing")
    } else if score <= -2 {
        (
            "Regressing",
            "latest quarter looks worse, ending the run is reasonable",
        )
    } else {
        ("Mixed", "watch a few more deaths before deciding")
    }
}

#[cfg(test)]
mod tests {
    use super::{HUD_QUARTER_COUNT, QuarterSummary, assess_recent_run};

    #[test]
    fn assess_recent_run_reports_improvement_when_latest_quarter_is_cleaner() {
        let mut quarters = [QuarterSummary::default(); HUD_QUARTER_COUNT];
        quarters[0] = QuarterSummary {
            count: 5,
            mean_progress_pct: 18.0,
            mean_return: -2.0,
            mean_life_seconds: 3.0,
            mean_centreline_distance: 28.0,
            mean_abs_heading_error_deg: 34.0,
            ..QuarterSummary::default()
        };
        quarters[3] = QuarterSummary {
            count: 5,
            mean_progress_pct: 31.0,
            mean_return: 1.5,
            mean_life_seconds: 5.0,
            mean_centreline_distance: 18.0,
            mean_abs_heading_error_deg: 22.0,
            ..QuarterSummary::default()
        };

        let (assessment, guidance) = assess_recent_run(&quarters);
        assert_eq!(assessment, "Improving");
        assert!(guidance.contains("worth continuing"));
    }
}
