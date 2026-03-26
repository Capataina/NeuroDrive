use bevy::prelude::*;
use bevy::ui::{
    AlignItems, BackgroundColor, Display, FlexDirection, Node, PositionType, UiRect, Val,
};

use crate::brain::ranking::TrainerLiveRanking;
use crate::debug::overlays::DebugOverlayState;
use crate::game::car::{Car, CarColour, EnvInstanceId, TrainerConfig};
use crate::game::episode::EpisodeMovingAverages;
use crate::game::progress::TrackProgress;

/// Marker for the leaderboard root UI node.
#[derive(Component)]
pub struct LeaderboardRoot;

/// Marker for a leaderboard row, keyed by position index.
#[derive(Component)]
pub struct LeaderboardRow(#[allow(dead_code)] pub u32);

/// Marker for the text element in a leaderboard row.
#[derive(Component)]
pub struct LeaderboardRowText(pub u32);

/// Marker for the colour swatch in a leaderboard row.
#[derive(Component)]
pub struct LeaderboardRowSwatch(pub u32);

/// Spawns the leaderboard panel (top-right).
pub fn spawn_leaderboard_system(mut commands: Commands, trainer_config: Res<TrainerConfig>) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(10.0),
                right: Val::Px(10.0),
                width: Val::Px(210.0),
                padding: UiRect::axes(Val::Px(10.0), Val::Px(8.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(3.0),
                display: Display::Flex,
                ..default()
            },
            BackgroundColor(Color::srgba(0.04, 0.06, 0.09, 0.72)),
            LeaderboardRoot,
        ))
        .with_children(|parent| {
            // Accent bar
            parent.spawn((
                Node {
                    width: Val::Percent(100.0),
                    height: Val::Px(2.0),
                    ..default()
                },
                BackgroundColor(Color::srgb(0.35, 0.58, 0.93)),
            ));

            parent.spawn((
                Text::new("Leaderboard"),
                TextFont::from_font_size(12.0),
                TextColor(Color::srgb(0.90, 0.93, 0.96)),
            ));

            // Pre-spawn rows for each env
            for i in 0..trainer_config.num_envs {
                let env_id = i as u32;
                parent
                    .spawn((
                        Node {
                            width: Val::Percent(100.0),
                            flex_direction: FlexDirection::Row,
                            align_items: AlignItems::Center,
                            column_gap: Val::Px(6.0),
                            padding: UiRect::axes(Val::Px(4.0), Val::Px(2.0)),
                            ..default()
                        },
                        BackgroundColor(Color::srgba(0.06, 0.09, 0.14, 0.50)),
                        LeaderboardRow(env_id),
                    ))
                    .with_children(|row| {
                        // Rank number
                        row.spawn((
                            Text::new(format!("{}.", i + 1)),
                            TextFont::from_font_size(10.0),
                            TextColor(Color::srgb(0.55, 0.62, 0.70)),
                        ));

                        // Colour swatch
                        row.spawn((
                            Node {
                                width: Val::Px(10.0),
                                height: Val::Px(10.0),
                                ..default()
                            },
                            BackgroundColor(Color::srgb(0.5, 0.5, 0.5)),
                            LeaderboardRowSwatch(env_id),
                        ));

                        // Car info text
                        row.spawn((
                            Text::new("---"),
                            TextFont::from_font_size(10.0),
                            TextColor(Color::srgb(0.78, 0.82, 0.86)),
                            LeaderboardRowText(env_id),
                        ));
                    });
            }
        });
}

/// Updates the leaderboard display from the trainer ranking.
pub fn update_leaderboard_system(
    ranking: Res<TrainerLiveRanking>,
    overlay: Res<DebugOverlayState>,
    car_query: Query<
        (&EnvInstanceId, &CarColour, &TrackProgress, &EpisodeMovingAverages),
        With<Car>,
    >,
    mut root_query: Query<&mut Node, With<LeaderboardRoot>>,
    mut text_query: Query<(Entity, &LeaderboardRowText)>,
    mut swatch_query: Query<(Entity, &LeaderboardRowSwatch, &mut BackgroundColor)>,
    mut text_writer: bevy::ui::widget::TextUiWriter,
) {
    // Show/hide based on F3 toggle
    if let Ok(mut node) = root_query.single_mut() {
        node.display = if overlay.telemetry {
            Display::Flex
        } else {
            Display::None
        };
    }

    if !overlay.telemetry {
        return;
    }

    // Build a lookup of car data by env_id
    let mut car_data: Vec<(u32, CarColour, f32, f32, f32)> = Vec::new();
    for (env_id, colour, progress, avg) in car_query.iter() {
        let score = if avg.best_progress_fractions.is_empty() {
            -1.0
        } else {
            0.7 * avg.best_progress_mean + 0.3 * normalise_return(avg.return_mean)
        };
        car_data.push((env_id.0, *colour, progress.fraction, avg.best_progress_mean, score));
    }

    // Sort by ranking if available, otherwise by env_id
    if !ranking.rankings.is_empty() {
        // Order car_data to match ranking order
        let rank_order: Vec<u32> = ranking.rankings.iter().map(|(id, _)| *id).collect();
        car_data.sort_by(|a, b| {
            let a_pos = rank_order.iter().position(|id| *id == a.0).unwrap_or(usize::MAX);
            let b_pos = rank_order.iter().position(|id| *id == b.0).unwrap_or(usize::MAX);
            a_pos.cmp(&b_pos)
        });
    }

    // Update each row
    for (rank, data) in car_data.iter().enumerate() {
        let (env_id, colour, live_progress, avg_progress, _score) = *data;
        let is_best = ranking.best_env_id == Some(env_id);

        // Update swatch colour
        for (_, swatch, mut bg) in swatch_query.iter_mut() {
            if swatch.0 == rank as u32 {
                *bg = BackgroundColor(Color::srgb(colour.r, colour.g, colour.b));
            }
        }

        // Update text
        let live_pct = (live_progress * 100.0).clamp(0.0, 100.0);
        let avg_pct = (avg_progress * 100.0).clamp(0.0, 100.0);
        let best_marker = if is_best { " *" } else { "" };
        let label = format!("Car {env_id}  {live_pct:5.1}%  avg {avg_pct:4.1}%{best_marker}");

        for (entity, row_text) in text_query.iter_mut() {
            if row_text.0 == rank as u32 {
                *text_writer.text(entity, 0) = label.clone();
            }
        }
    }
}

fn normalise_return(ret: f32) -> f32 {
    ((ret + 5.0) / 105.0).clamp(0.0, 1.0)
}
