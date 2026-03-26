use bevy::prelude::*;

use crate::game::car::{Car, CarColour, EnvInstanceId, TrainerConfig};
use crate::game::episode::EpisodeMovingAverages;

/// Trainer-level live ranking of car performance, used to determine which
/// car gets the visual highlight and for debug/HUD display.
#[derive(Resource, Debug)]
pub struct TrainerLiveRanking {
    /// Current best-performing env instance, if any car has completed episodes.
    pub best_env_id: Option<u32>,
    /// Current worst-performing env instance.
    pub worst_env_id: Option<u32>,
    /// All cars ranked by score, descending. (env_id, score).
    pub rankings: Vec<(u32, f32)>,
    /// Tick counter for bounded update cadence.
    ticks_since_update: u32,
    /// Minimum ticks between ranking recomputations to prevent flicker.
    update_cadence_ticks: u32,
    /// A new car must exceed the current best by this relative margin to take over.
    hysteresis_margin: f32,
}

impl Default for TrainerLiveRanking {
    fn default() -> Self {
        Self {
            best_env_id: None,
            worst_env_id: None,
            rankings: Vec::new(),
            ticks_since_update: 0,
            update_cadence_ticks: 60, // recompute at most once per second at 60Hz
            hysteresis_margin: 0.05,
        }
    }
}

/// Recomputes the trainer ranking from per-car EpisodeMovingAverages.
/// Runs in Update schedule at a bounded cadence to prevent flicker.
pub fn update_trainer_ranking_system(
    mut ranking: ResMut<TrainerLiveRanking>,
    car_query: Query<(&EnvInstanceId, &EpisodeMovingAverages), With<Car>>,
) {
    ranking.ticks_since_update += 1;
    if ranking.ticks_since_update < ranking.update_cadence_ticks {
        return;
    }
    ranking.ticks_since_update = 0;

    // Compute score for each car that has data
    let mut scores: Vec<(u32, f32)> = Vec::new();
    for (env_id, avg) in car_query.iter() {
        // Cars with no completed episodes get no score
        if avg.best_progress_fractions.is_empty() {
            continue;
        }

        // Score: weighted combination of progress and return
        let score = 0.7 * avg.best_progress_mean + 0.3 * normalise_return(avg.return_mean);
        scores.push((env_id.0, score));
    }

    if scores.is_empty() {
        return;
    }

    // Sort descending by score
    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Apply hysteresis: new best must exceed current best by margin
    let new_best_id = scores[0].0;
    let new_best_score = scores[0].1;

    let best = if let Some(current_best_id) = ranking.best_env_id {
        if new_best_id == current_best_id {
            // Same car is still best
            Some(current_best_id)
        } else if let Some(current_score) = scores.iter().find(|(id, _)| *id == current_best_id) {
            // Current best still exists — new car must exceed by margin
            if new_best_score > current_score.1 * (1.0 + ranking.hysteresis_margin) {
                Some(new_best_id)
            } else {
                Some(current_best_id)
            }
        } else {
            // Current best no longer in scores (shouldn't happen, but handle it)
            Some(new_best_id)
        }
    } else {
        Some(new_best_id)
    };

    let worst = scores.last().map(|(id, _)| *id);

    ranking.best_env_id = best;
    ranking.worst_env_id = worst;
    ranking.rankings = scores;
}

/// Adjusts sprite alpha and z-ordering based on the current trainer ranking.
/// Best car gets full opacity and renders on top; all others are dimmed.
/// Each car retains its unique assigned colour.
pub fn update_car_visual_roles_system(
    ranking: Res<TrainerLiveRanking>,
    trainer_config: Res<TrainerConfig>,
    mut car_query: Query<(&EnvInstanceId, &CarColour, &mut Sprite, &mut Transform), With<Car>>,
) {
    let Some(best_id) = ranking.best_env_id else {
        return;
    };

    for (env_id, colour, mut sprite, mut transform) in car_query.iter_mut() {
        let alpha = if env_id.0 == best_id {
            trainer_config.best_car_alpha
        } else {
            trainer_config.default_car_alpha
        };
        sprite.color = Color::srgba(colour.r, colour.g, colour.b, alpha);
        transform.translation.z = if env_id.0 == best_id { 11.0 } else { 10.0 };
    }
}

/// Normalises return to roughly [0, 1] range for score combination.
/// This is a simple sigmoid-like scaling — exact range doesn't matter,
/// just needs to be in a similar scale as progress fraction.
fn normalise_return(ret: f32) -> f32 {
    // Typical returns range from ~-5 (instant crash) to ~+100 (lap complete).
    // Map to [0, 1] using a simple linear scale clamped.
    ((ret + 5.0) / 105.0).clamp(0.0, 1.0)
}
