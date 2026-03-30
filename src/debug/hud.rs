use std::collections::VecDeque;

use bevy::prelude::*;
use bevy::ui::widget::{Text, TextUiWriter};
use bevy::ui::{
    AlignItems, BackgroundColor, Display, FlexDirection, JustifyContent, Node, PositionType,
    UiRect, Val,
};

use crate::agent::observation::SensorReadings;
use crate::brain::a2c::A2cTrainingStats;
use crate::brain::ranking::TrainerLiveRanking;
use crate::debug::overlays::DebugOverlayState;
use crate::game::car::{Car, EnvInstanceId};
use crate::game::collision::Collided;
use crate::game::episode::{EpisodeConfig, EpisodeEndReason, EpisodeMovingAverages, EpisodeState};

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

/// Placeholder resource retained for compatibility with DebugPlugin resource init.
/// Episode-level accumulation is no longer needed since we fold completions directly
/// from EpisodeState per car.
#[derive(Resource, Debug, Default)]
pub struct DrivingHudEpisodeAccumulator;

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
    RunDetail,
    Learning,
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
    Progress,
    Life,
    Return,
    Ends,
}

const QUARTER_COLUMNS: [QuarterColumn; 6] = [
    QuarterColumn::Quarter,
    QuarterColumn::Count,
    QuarterColumn::Progress,
    QuarterColumn::Life,
    QuarterColumn::Return,
    QuarterColumn::Ends,
];

fn quarter_column_width(column: QuarterColumn) -> f32 {
    match column {
        QuarterColumn::Quarter => 24.0,
        QuarterColumn::Count => 28.0,
        QuarterColumn::Progress => 56.0,
        QuarterColumn::Life => 50.0,
        QuarterColumn::Return => 58.0,
        QuarterColumn::Ends => 72.0,
    }
}

/// Spawns the runtime diagnostics HUD used by `F3`.
pub(crate) fn spawn_driving_hud_system(mut commands: Commands) {
    let bg = Color::srgba(0.04, 0.06, 0.09, 0.72);
    let accent = Color::srgb(0.35, 0.58, 0.93);
    let text_bright = Color::srgb(0.90, 0.93, 0.96);
    let text_primary = Color::srgb(0.78, 0.82, 0.86);
    let text_dim = Color::srgb(0.55, 0.62, 0.70);
    let divider = Color::srgba(0.40, 0.50, 0.60, 0.18);
    let header_bg = Color::srgba(0.08, 0.12, 0.18, 0.80);
    let cell_bg = Color::srgba(0.06, 0.09, 0.14, 0.70);
    let table_bg = Color::srgba(0.03, 0.05, 0.08, 0.30);

    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                width: Val::Px(440.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                display: Display::None,
                ..default()
            },
            BackgroundColor(bg),
            DrivingHudRoot,
        ))
        .with_children(|parent| {
            // Accent bar
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(2.0),
                    ..default()
                },
                BackgroundColor(accent),
            ));

            // Title
            parent.spawn((
                Text::new("NeuroDrive"),
                TextFont::from_font_size(13.0),
                TextColor(text_bright),
            ));

            // Assessment
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(10.0),
                TextColor(accent),
                HudTextRole::Assessment,
            ));

            // Divider
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    ..default()
                },
                BackgroundColor(divider),
            ));

            // Current (live metrics)
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(11.0),
                TextColor(text_primary),
                HudTextRole::Current,
            ));

            // Run (episode stats)
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(11.0),
                TextColor(text_primary),
                HudTextRole::Run,
            ));

            // Run detail (best/avg)
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(11.0),
                TextColor(text_dim),
                HudTextRole::RunDetail,
            ));

            // Learning (PPO)
            parent.spawn((
                Text::new(""),
                TextFont::from_font_size(11.0),
                TextColor(text_dim),
                HudTextRole::Learning,
            ));

            // Divider
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(1.0),
                    ..default()
                },
                BackgroundColor(divider),
            ));

            // Quarters subtitle
            parent.spawn((
                Text::new("Quarters"),
                TextFont::from_font_size(10.0),
                TextColor(text_dim),
            ));

            // Quarter table
            parent
                .spawn((
                    Node {
                        width: Val::Percent(100.0),
                        flex_direction: FlexDirection::Column,
                        row_gap: Val::Px(1.0),
                        ..default()
                    },
                    BackgroundColor(table_bg),
                ))
                .with_children(|table| {
                    // Header row
                    let headers: [(&str, QuarterColumn); 6] = [
                        ("Q", QuarterColumn::Quarter),
                        ("N", QuarterColumn::Count),
                        ("Prog", QuarterColumn::Progress),
                        ("Life", QuarterColumn::Life),
                        ("Rwd", QuarterColumn::Return),
                        ("C/T", QuarterColumn::Ends),
                    ];

                    table
                        .spawn((
                            Node {
                                width: Val::Percent(100.0),
                                flex_direction: FlexDirection::Row,
                                column_gap: Val::Px(2.0),
                                ..default()
                            },
                            BackgroundColor(header_bg),
                        ))
                        .with_children(|row| {
                            for (label, col) in &headers {
                                row.spawn((
                                    Node {
                                        width: Val::Px(quarter_column_width(*col)),
                                        padding: UiRect::axes(Val::Px(3.0), Val::Px(2.0)),
                                        justify_content: JustifyContent::FlexStart,
                                        align_items: AlignItems::Center,
                                        ..default()
                                    },
                                    BackgroundColor(header_bg),
                                ))
                                .with_children(|cell| {
                                    cell.spawn((
                                        Text::new(*label),
                                        TextFont::from_font_size(9.5),
                                        TextColor(text_dim),
                                    ));
                                });
                            }
                        });

                    // Data rows
                    for row_index in 0..HUD_QUARTER_COUNT {
                        table
                            .spawn((
                                Node {
                                    width: Val::Percent(100.0),
                                    flex_direction: FlexDirection::Row,
                                    column_gap: Val::Px(2.0),
                                    ..default()
                                },
                                BackgroundColor(cell_bg),
                            ))
                            .with_children(|row| {
                                for column in QUARTER_COLUMNS {
                                    row.spawn((
                                        Node {
                                            width: Val::Px(quarter_column_width(column)),
                                            padding: UiRect::axes(Val::Px(3.0), Val::Px(2.0)),
                                            justify_content: JustifyContent::FlexStart,
                                            align_items: AlignItems::Center,
                                            ..default()
                                        },
                                        BackgroundColor(cell_bg),
                                    ))
                                    .with_children(|cell| {
                                        cell.spawn((
                                            Text::new(""),
                                            TextFont::from_font_size(9.5),
                                            TextColor(text_primary),
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
        });
}

/// Tracks best distance-driven progress and death count across ALL cars.
pub(crate) fn update_driving_hud_stats_system(
    mut hud_stats: ResMut<DrivingHudStats>,
    car_query: Query<(&EnvInstanceId, &EpisodeState, Has<Collided>), With<Car>>,
) {
    for (_env_id, episode_state, collided) in car_query.iter() {
        if collided {
            hud_stats.deaths = hud_stats.deaths.saturating_add(1);
        }

        if episode_state.current_best_progress_fraction > hud_stats.best_progress_fraction {
            hud_stats.best_progress_fraction = episode_state.current_best_progress_fraction;
            hud_stats.best_progress_episode = episode_state.current_episode;
        }
    }
}

/// Folds ALL cars' completed episodes into HUD history.
pub(crate) fn capture_driving_hud_episode_metrics_system(
    config: Res<EpisodeConfig>,
    car_query: Query<(&EnvInstanceId, &EpisodeState), With<Car>>,
    mut history: ResMut<DrivingHudHistory>,
) {
    for (_env_id, episode_state) in car_query.iter() {
        let Some(end_reason) = episode_state.current_tick_end_reason else {
            continue;
        };

        // This car finished an episode this tick.
        history.episodes.push_back(CompletedHudEpisode {
            end_reason,
            best_progress_fraction: episode_state.last_episode_best_progress_fraction,
            total_return: episode_state.last_episode_return,
            life_seconds: episode_state.last_episode_ticks as f32 * FIXED_TICK_SECONDS,
            mean_centreline_distance: episode_state.current_tick_centerline_distance,
            mean_abs_heading_error_deg: episode_state
                .current_tick_heading_error
                .abs()
                .to_degrees(),
        });
        while history.episodes.len() > config.moving_average_window.max(1) {
            let _ = history.episodes.pop_front();
        }
    }
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

/// Displays the best car's live stats. Uses TrainerLiveRanking to select
/// which car to show; falls back to the first car if no ranking exists yet.
pub(crate) fn update_driving_hud_text_system(
    overlay: Res<DebugOverlayState>,
    hud_stats: Res<DrivingHudStats>,
    history: Res<DrivingHudHistory>,
    a2c_stats: Option<Res<A2cTrainingStats>>,
    ranking: Option<Res<TrainerLiveRanking>>,
    car_query: Query<
        (
            &EnvInstanceId,
            &SensorReadings,
            &EpisodeState,
            &EpisodeMovingAverages,
        ),
        With<Car>,
    >,
    summary_query: Query<(Entity, &HudTextRole)>,
    quarter_query: Query<(Entity, &QuarterCell)>,
    mut text_writer: TextUiWriter,
) {
    if !overlay.telemetry {
        return;
    }

    // Select the best car from the ranking, falling back to any car.
    let best_env_id = ranking.as_ref().and_then(|r| r.best_env_id);
    let Some((_, sensors, episode_state, moving_avg)) = (match best_env_id {
        Some(target_id) => car_query
            .iter()
            .find(|(env_id, _, _, _)| env_id.0 == target_id)
            .or_else(|| car_query.iter().next()),
        None => car_query.iter().next(),
    }) else {
        return;
    };

    let progress_pct = (episode_state.current_tick_progress_fraction * 100.0).clamp(0.0, 100.0);
    let best_progress_pct = (hud_stats.best_progress_fraction * 100.0).clamp(0.0, 100.0);
    let life_best_progress_pct =
        (episode_state.current_best_progress_fraction * 100.0).clamp(0.0, 100.0);
    let current_life_seconds = episode_state.ticks_in_episode as f32 * FIXED_TICK_SECONDS;
    let heading_error_deg = sensors.heading_error.to_degrees();
    let avg_progress_pct = (moving_avg.best_progress_mean * 100.0).clamp(0.0, 100.0);
    let last_reason = match episode_state.last_end_reason {
        Some(EpisodeEndReason::Crash) => "Crash",
        Some(EpisodeEndReason::Timeout) => "Timeout",
        None => "N/A",
    };
    let recent_quarters = summarise_recent_history(&history);
    let (assessment, guidance) = assess_recent_run(&recent_quarters);

    let current_line = format!(
        "Prog {progress_pct:.1}%  Best {life_best_progress_pct:.1}%  Gap {gap:.1}  Head {heading_error_deg:.1}\u{00b0}  Off {offset:+.1}",
        gap = episode_state.current_tick_centerline_distance,
        offset = sensors.signed_lateral_offset,
    );
    let run_line = format!(
        "Ep {}  Life {:.1}s  Rwd {:+.1}  Deaths {}  Last {}",
        episode_state.current_episode,
        current_life_seconds,
        episode_state.current_return,
        hud_stats.deaths,
        last_reason,
    );
    let run_detail_line = format!(
        "Best {:.1}% (ep{})  Avg {:.1}% / {:+.1}",
        best_progress_pct,
        hud_stats.best_progress_episode,
        avg_progress_pct,
        moving_avg.return_mean,
    );
    let learning_line = match a2c_stats {
        Some(stats) if stats.last_completed_update > 0 => format!(
            "PPO #{}  EV {:.3}  VL {:.3}  Ent {:.3}  Clip {:.0}%  KL {:.4}",
            stats.last_completed_update,
            stats.explained_variance,
            stats.value_loss,
            stats.policy_entropy,
            stats.clip_fraction * 100.0,
            stats.approx_kl,
        ),
        _ => "PPO  waiting for first update".to_string(),
    };

    for (entity, role) in &summary_query {
        let text = match role {
            HudTextRole::Assessment => format!("{assessment}  \u{2014}  {guidance}"),
            HudTextRole::Current => current_line.clone(),
            HudTextRole::Run => run_line.clone(),
            HudTextRole::RunDetail => run_detail_line.clone(),
            HudTextRole::Learning => learning_line.clone(),
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
            QuarterColumn::Progress => "-".to_string(),
            QuarterColumn::Life => "-".to_string(),
            QuarterColumn::Return => "-".to_string(),
            QuarterColumn::Ends => "--".to_string(),
        };
    }

    match column {
        QuarterColumn::Quarter => format!("Q{}", quarter_index + 1),
        QuarterColumn::Count => format!("{}", quarter.count),
        QuarterColumn::Progress => format!("{:.1}%", quarter.mean_progress_pct),
        QuarterColumn::Life => format!("{:.1}s", quarter.mean_life_seconds),
        QuarterColumn::Return => format!("{:+.1}", quarter.mean_return),
        QuarterColumn::Ends => format!(
            "{}/{}",
            quarter.crash_count, quarter.timeout_count
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
