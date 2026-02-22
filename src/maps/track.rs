use bevy::prelude::*;

use crate::maps::grid::TrackGrid;

/// Component attached to the single track entity.
///
/// Carries the tile grid used for collision detection and the world-space
/// spawn position and heading that the car uses on reset.
///
/// All callers that previously used `outer_boundary` / `inner_boundary` for
/// collision now query `grid.is_road_at(world_pos)` instead.
#[derive(Component)]
pub struct Track {
    /// Grid of tile parts that define the driveable surface and walls.
    pub grid: TrackGrid,

    /// World-space position at which the car spawns or resets.
    /// Derived from the `SpawnPoint` tile centre.
    pub spawn_position: Vec2,

    /// Heading angle in radians at which the car spawns.
    /// 0.0 means facing east (+X). Derived from `SpawnPoint` connectivity.
    pub spawn_rotation: f32,
}
