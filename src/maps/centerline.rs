use bevy::prelude::*;

use crate::maps::grid::TrackGrid;
use crate::maps::parts::TilePart;

/// Number of samples used to approximate each quarter-circle centreline arc.
///
/// Higher values produce a smoother tangent estimate at the cost of more
/// projection work.
const CENTERLINE_ARC_SAMPLES: usize = 8;

/// Cardinal directions in grid space.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GridDir {
    North,
    South,
    East,
    West,
}

impl GridDir {
    /// Returns the opposite direction.
    pub fn opposite(self) -> Self {
        match self {
            GridDir::North => GridDir::South,
            GridDir::South => GridDir::North,
            GridDir::East => GridDir::West,
            GridDir::West => GridDir::East,
        }
    }

    /// Returns the `(d_row, d_col)` offset for this direction.
    pub fn delta(self) -> (isize, isize) {
        match self {
            GridDir::North => (-1, 0),
            GridDir::South => (1, 0),
            GridDir::East => (0, 1),
            GridDir::West => (0, -1),
        }
    }
}

/// Errors that can occur while constructing a centreline from a tile grid.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub enum CenterlineBuildError {
    /// The start cell was out of bounds or not a road tile.
    InvalidStartCell { row: usize, col: usize },
    /// A road tile did not provide a valid next step.
    DeadEnd { row: usize, col: usize },
    /// A road tile provided multiple next-step options (branching track).
    AmbiguousBranch { row: usize, col: usize, options: Vec<GridDir> },
    /// The traversal failed to close back to the start.
    NotClosedLoop,
    /// The computed loop was too short to be meaningful.
    TooShort,
}

/// A closed polyline centreline for progress measurement.
///
/// The centreline is represented as an ordered set of points in world space.
/// Segments connect point `i` to point `(i+1) % N`.
#[derive(Clone, Debug)]
pub struct TrackCenterline {
    /// World-space polyline points, in traversal order.
    pub points: Vec<Vec2>,
    cumulative_lengths: Vec<f32>,
    total_length: f32,
}

impl TrackCenterline {
    /// Total arc length of the closed polyline.
    pub fn total_length(&self) -> f32 {
        self.total_length
    }

    /// Builds a closed centreline by traversing grid connectivity.
    ///
    /// This assumes a single closed loop (degree-2 track). If junction tiles
    /// exist (T-junctions or crossroads), the builder will reject ambiguous
    /// branches rather than guessing.
    ///
    /// The resulting centreline follows the midline of each tile:
    /// - Straights are line segments between open-edge midpoints.
    /// - Corners are quarter-circle arcs between open-edge midpoints.
    pub fn build_closed_loop(
        grid: &TrackGrid,
        start_cell: (usize, usize),
        start_dir: GridDir,
    ) -> Result<Self, CenterlineBuildError> {
        let (cells, dirs) = traverse_cells(grid, start_cell, start_dir)?;
        let points = build_polyline_points(grid, &cells, &dirs);
        if points.len() < 3 {
            return Err(CenterlineBuildError::TooShort);
        }

        let (cumulative_lengths, total_length) = compute_lengths(&points);
        Ok(Self {
            points,
            cumulative_lengths,
            total_length,
        })
    }

    /// Projects a world position onto the centreline polyline.
    ///
    /// Returns the closest point on the polyline, its segment tangent, and the
    /// arc-length progress `s` along the track in `[0, total_length)`.
    pub fn project(&self, world: Vec2) -> CenterlineProjection {
        let n = self.points.len();
        debug_assert!(n >= 2, "centreline must have at least two points");

        let mut best = CenterlineProjection {
            closest_point: self.points[0],
            tangent: Vec2::X,
            s: 0.0,
            fraction: 0.0,
            distance: f32::INFINITY,
        };

        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            let d = b - a;
            let len2 = d.length_squared();
            if len2 <= 1e-8 {
                continue;
            }

            let t = ((world - a).dot(d) / len2).clamp(0.0, 1.0);
            let p = a + d * t;
            let dist = world.distance(p);

            if dist < best.distance {
                let seg_len = len2.sqrt();
                let s = self.cumulative_lengths[i] + seg_len * t;
                best = CenterlineProjection {
                    closest_point: p,
                    tangent: d / seg_len,
                    s,
                    fraction: (s / self.total_length).clamp(0.0, 1.0),
                    distance: dist,
                };
            }
        }

        best
    }
}

/// Result of projecting a point onto a centreline.
#[derive(Clone, Copy, Debug)]
pub struct CenterlineProjection {
    /// Closest point on the centreline polyline.
    pub closest_point: Vec2,
    /// Unit-length tangent vector of the closest segment.
    pub tangent: Vec2,
    /// Arc-length distance along the track from the start point.
    pub s: f32,
    /// `s / total_length` in `[0, 1]`.
    pub fraction: f32,
    /// Euclidean distance from the query point to `closest_point`.
    pub distance: f32,
}

fn traverse_cells(
    grid: &TrackGrid,
    start_cell: (usize, usize),
    start_dir: GridDir,
) -> Result<(Vec<(usize, usize)>, Vec<GridDir>), CenterlineBuildError> {
    let (start_row, start_col) = start_cell;
    if grid.tile_at(start_row, start_col) == TilePart::Empty {
        return Err(CenterlineBuildError::InvalidStartCell {
            row: start_row,
            col: start_col,
        });
    }

    let mut cells: Vec<(usize, usize)> = vec![start_cell];
    let mut dirs: Vec<GridDir> = Vec::new();

    let mut visited = std::collections::HashSet::<(usize, usize)>::new();
    visited.insert(start_cell);

    let mut current = start_cell;
    let mut incoming = start_dir.opposite();

    loop {
        let next_dir = if current == start_cell && dirs.is_empty() {
            start_dir
        } else {
            choose_next_dir(grid, current, incoming)?
        };

        let next = step_cell(current, next_dir).ok_or(CenterlineBuildError::DeadEnd {
            row: current.0,
            col: current.1,
        })?;

        dirs.push(next_dir);

        if next == start_cell {
            break;
        }

        if !visited.insert(next) {
            return Err(CenterlineBuildError::NotClosedLoop);
        }

        if grid.tile_at(next.0, next.1) == TilePart::Empty {
            return Err(CenterlineBuildError::DeadEnd {
                row: current.0,
                col: current.1,
            });
        }

        cells.push(next);
        incoming = next_dir.opposite();
        current = next;
    }

    Ok((cells, dirs))
}

fn build_polyline_points(grid: &TrackGrid, cells: &[(usize, usize)], dirs: &[GridDir]) -> Vec<Vec2> {
    let half = grid.tile_size * 0.5;
    let mut points: Vec<Vec2> = Vec::new();

    let n = cells.len();
    if n == 0 {
        return points;
    }

    for i in 0..n {
        let cell = cells[i];
        let tile = grid.tile_at(cell.0, cell.1);
        let center = grid.cell_center(cell.0, cell.1);

        let prev_dir = if i == 0 { dirs[n - 1] } else { dirs[i - 1] };
        let entry_dir = prev_dir.opposite();
        let exit_dir = dirs[i];

        let entry = center + dir_unit(entry_dir) * half;
        let exit = center + dir_unit(exit_dir) * half;

        push_unique(&mut points, entry);

        if tile.is_corner() {
            let arc_center = corner_arc_center(tile, center, half);
            push_corner_arc_samples(&mut points, arc_center, half, entry, exit);
        } else {
            push_unique(&mut points, exit);
        }
    }

    // Close the loop if needed (avoid duplicating the first point).
    if let (Some(first), Some(last)) = (points.first().copied(), points.last().copied()) {
        if last.distance(first) < 1e-3 {
            points.pop();
        }
    }

    points
}

fn push_unique(points: &mut Vec<Vec2>, p: Vec2) {
    if let Some(last) = points.last().copied() {
        if last.distance(p) < 1e-3 {
            return;
        }
    }
    points.push(p);
}

fn dir_unit(dir: GridDir) -> Vec2 {
    match dir {
        GridDir::North => Vec2::Y,
        GridDir::South => -Vec2::Y,
        GridDir::East => Vec2::X,
        GridDir::West => -Vec2::X,
    }
}

fn push_corner_arc_samples(points: &mut Vec<Vec2>, center: Vec2, radius: f32, entry: Vec2, exit: Vec2) {
    let a0 = (entry - center).y.atan2((entry - center).x);
    let a1 = (exit - center).y.atan2((exit - center).x);
    let delta = wrap_to_pi(a1 - a0);

    // Exclude the first point (already pushed) and include the final exit point.
    for i in 1..=CENTERLINE_ARC_SAMPLES {
        let t = i as f32 / CENTERLINE_ARC_SAMPLES as f32;
        let a = a0 + delta * t;
        let p = center + Vec2::new(a.cos(), a.sin()) * radius;
        push_unique(points, p);
    }
}

fn wrap_to_pi(mut a: f32) -> f32 {
    use std::f32::consts::PI;
    while a <= -PI {
        a += 2.0 * PI;
    }
    while a > PI {
        a -= 2.0 * PI;
    }
    a
}

fn corner_arc_center(tile: TilePart, cell_center: Vec2, half: f32) -> Vec2 {
    let cx = cell_center.x;
    let cy = cell_center.y;

    match tile {
        TilePart::CornerNW => Vec2::new(cx + half, cy - half),
        TilePart::CornerNE => Vec2::new(cx - half, cy - half),
        TilePart::CornerSW => Vec2::new(cx + half, cy + half),
        TilePart::CornerSE => Vec2::new(cx - half, cy + half),
        _ => cell_center,
    }
}

fn choose_next_dir(
    grid: &TrackGrid,
    cell: (usize, usize),
    incoming: GridDir,
) -> Result<GridDir, CenterlineBuildError> {
    let tile = grid.tile_at(cell.0, cell.1);
    let (open_n, open_s, open_e, open_w) = tile.open_edges();

    let mut options: Vec<GridDir> = Vec::new();
    if open_n { options.push(GridDir::North); }
    if open_s { options.push(GridDir::South); }
    if open_e { options.push(GridDir::East); }
    if open_w { options.push(GridDir::West); }

    // Avoid immediately returning to the cell we just came from.
    options.retain(|d| *d != incoming);

    match options.as_slice() {
        [only] => Ok(*only),
        [] => Err(CenterlineBuildError::DeadEnd { row: cell.0, col: cell.1 }),
        many => Err(CenterlineBuildError::AmbiguousBranch {
            row: cell.0,
            col: cell.1,
            options: many.to_vec(),
        }),
    }
}

fn step_cell((row, col): (usize, usize), dir: GridDir) -> Option<(usize, usize)> {
    let (dr, dc) = dir.delta();
    let next_row = row.checked_add_signed(dr)?;
    let next_col = col.checked_add_signed(dc)?;
    Some((next_row, next_col))
}

fn compute_lengths(points: &[Vec2]) -> (Vec<f32>, f32) {
    let n = points.len();
    let mut cumulative: Vec<f32> = vec![0.0; n];
    let mut total = 0.0;

    for i in 0..n {
        cumulative[i] = total;
        let a = points[i];
        let b = points[(i + 1) % n];
        total += a.distance(b);
    }

    (cumulative, total)
}
