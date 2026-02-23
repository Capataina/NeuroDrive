use bevy::ecs::message::MessageReader;
use bevy::prelude::*;
use bevy::ui::widget::{Text, TextUiWriter};
use bevy::ui::{BackgroundColor, Display, Node, PositionType, Val};

use crate::agent::observation::SensorReadings;
use crate::debug::overlays::DebugOverlayState;
use crate::game::car::Car;
use crate::game::collision::CollisionEvent;
use crate::game::episode::{EpisodeEndReason, EpisodeMovingAverages, EpisodeState};
use crate::game::progress::TrackProgress;

#[derive(Resource, Debug)]
pub struct DrivingHudStats {
    pub deaths: u32,
    pub current_life: u32,
    pub best_progress_fraction: f32,
    pub best_progress_life: u32,
}

impl Default for DrivingHudStats {
    fn default() -> Self {
        Self {
            deaths: 0,
            current_life: 1,
            best_progress_fraction: 0.0,
            best_progress_life: 1,
        }
    }
}

#[derive(Component)]
pub struct DrivingHudRoot;

#[derive(Component)]
pub struct DrivingHudText;

pub fn spawn_driving_hud_system(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                left: Val::Px(10.0),
                padding: bevy::ui::UiRect::all(Val::Px(8.0)),
                display: Display::None,
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.05, 0.05, 0.78)),
            DrivingHudRoot,
        ))
        .with_children(|parent| {
            parent
                .spawn((
                    Text::new("Driving State\n"),
                    TextFont::from_font_size(18.0),
                    TextColor(Color::srgb(0.95, 0.95, 0.95)),
                    DrivingHudText,
                ))
                .with_child((TextSpan::default(), TextFont::from_font_size(16.0)));
        });
}

pub fn update_driving_hud_stats_system(
    mut hud_stats: ResMut<DrivingHudStats>,
    mut collision_events: MessageReader<CollisionEvent>,
    progress_query: Query<&TrackProgress, With<Car>>,
) {
    for _ in collision_events.read() {
        hud_stats.deaths = hud_stats.deaths.saturating_add(1);
        hud_stats.current_life = hud_stats.current_life.saturating_add(1);
    }

    let Ok(progress) = progress_query.single() else {
        return;
    };

    if progress.fraction > hud_stats.best_progress_fraction {
        hud_stats.best_progress_fraction = progress.fraction;
        hud_stats.best_progress_life = hud_stats.current_life;
    }
}

pub fn update_driving_hud_visibility_system(
    overlay: Res<DebugOverlayState>,
    mut root_query: Query<&mut Node, With<DrivingHudRoot>>,
) {
    let Ok(mut node) = root_query.single_mut() else {
        return;
    };

    node.display = if overlay.telemetry {
        Display::DEFAULT
    } else {
        Display::None
    };
}

pub fn update_driving_hud_text_system(
    overlay: Res<DebugOverlayState>,
    hud_stats: Res<DrivingHudStats>,
    episode_state: Res<EpisodeState>,
    moving_avg: Res<EpisodeMovingAverages>,
    car_query: Query<(&TrackProgress, &SensorReadings), With<Car>>,
    text_query: Query<Entity, With<DrivingHudText>>,
    mut text_writer: TextUiWriter,
) {
    if !overlay.telemetry {
        return;
    }

    let Ok((progress, sensors)) = car_query.single() else {
        return;
    };
    let Ok(text_entity) = text_query.single() else {
        return;
    };

    let progress_pct = (progress.fraction * 100.0).clamp(0.0, 100.0);
    let best_progress_pct = (hud_stats.best_progress_fraction * 100.0).clamp(0.0, 100.0);
    let heading_error_deg = sensors.heading_error.to_degrees();
    let avg_progress_pct = (moving_avg.best_progress_mean * 100.0).clamp(0.0, 100.0);
    let last_reason = match episode_state.last_end_reason {
        Some(EpisodeEndReason::Crash) => "Crash",
        Some(EpisodeEndReason::Timeout) => "Timeout",
        Some(EpisodeEndReason::LapComplete) => "Lap",
        None => "N/A",
    };

    *text_writer.text(text_entity, 1) = format!(
        "Progress: {progress_pct:6.2}%\nHeading error: {heading_error_deg:7.2} deg\nDeaths: {}\nBest progress: {best_progress_pct:6.2}% (Life {})\n\nEpisode: {}\nCurrent reward: {:+8.2}\nEpisode crashes: {}\nLast end: {}\nAvg reward ({}): {:+8.2}\nAvg progress ({}): {avg_progress_pct:6.2}%\nAvg crashes ({}): {:6.2}",
        hud_stats.deaths,
        hud_stats.best_progress_life,
        episode_state.current_episode,
        episode_state.current_return,
        episode_state.current_crashes,
        last_reason,
        moving_avg.returns.len(),
        moving_avg.return_mean,
        moving_avg.best_progress_fractions.len(),
        moving_avg.crash_counts.len(),
        moving_avg.crash_mean,
    );
}
