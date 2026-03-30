use bevy::prelude::*;

use crate::maps::centerline::TrackCenterline;
use crate::maps::grid::TrackGrid;

/// Component attached to the single track entity.
///
/// Carries the tile grid used for collision detection and the closed
/// centreline polyline used for progress measurement.
///
/// All callers that previously used `outer_boundary` / `inner_boundary` for
/// collision now query `grid.is_road_at(world_pos)` instead.
#[derive(Component)]
pub struct Track {
    /// Grid of tile parts that define the driveable surface and walls.
    pub grid: TrackGrid,

    /// Closed centreline polyline used for progress measurement.
    pub centerline: TrackCenterline,
}
