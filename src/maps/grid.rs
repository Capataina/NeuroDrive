use bevy::asset::RenderAssetUsages;
use bevy::prelude::*;
use bevy::render::render_resource::PrimitiveTopology;

use crate::maps::parts::TilePart;

/// Number of line segments used to approximate each quarter-circle corner arc.
/// Higher values produce smoother curves at the cost of more sprite entities.
const ARC_SEGMENTS: usize = 12;

/// Thickness in pixels of rendered wall sprites.
/// Also used by collision detection: the driveable area of each tile is
/// inset by half this value on every closed edge, so the car collides at
/// the inner face of the visual wall.
const WALL_THICKNESS: f32 = 5.0;

// ─────────────────────────────────────────────────────────────────────────────
// TrackGrid
// ─────────────────────────────────────────────────────────────────────────────

/// A grid-based track definition.
///
/// Tiles are stored row-major as `tiles[row][col]`.
/// Row 0 is the topmost row (highest world-Y); row index increases downward.
///
/// World-space coordinates of the top-left corner of cell `[row][col]`:
/// ```text
/// x = origin.x + col * tile_size
/// y = origin.y − row * tile_size
/// ```
///
/// The grid owns no Bevy entities. Rendering is performed by
/// [`render_tile_grid`]; spawning the track entity is the plugin's job.
pub struct TrackGrid {
    /// Row-major tile array. Index as `tiles[row][col]`.
    pub tiles: Vec<Vec<TilePart>>,

    /// World-space side length of each square cell in pixels.
    pub tile_size: f32,

    /// World-space position of the top-left corner of cell `[0][0]`.
    pub origin: Vec2,
}

impl TrackGrid {
    /// Constructs a new grid.
    ///
    /// All rows must have the same length; behaviour is undefined otherwise.
    /// `origin` is the world-space top-left corner of cell `[0][0]`.
    pub fn new(tiles: Vec<Vec<TilePart>>, tile_size: f32, origin: Vec2) -> Self {
        Self { tiles, tile_size, origin }
    }

    /// Number of rows in the grid.
    pub fn rows(&self) -> usize {
        self.tiles.len()
    }

    /// Number of columns (derived from the first row; 0 if empty).
    pub fn cols(&self) -> usize {
        self.tiles.first().map(|r| r.len()).unwrap_or(0)
    }

    /// Returns the `TilePart` at `(row, col)`, or `Empty` when out of bounds.
    pub fn tile_at(&self, row: usize, col: usize) -> TilePart {
        self.tiles
            .get(row)
            .and_then(|r| r.get(col))
            .copied()
            .unwrap_or(TilePart::Empty)
    }

    /// Returns the world-space centre of tile `(row, col)`.
    pub fn cell_center(&self, row: usize, col: usize) -> Vec2 {
        Vec2::new(
            self.origin.x + col as f32 * self.tile_size + self.tile_size * 0.5,
            self.origin.y - row as f32 * self.tile_size - self.tile_size * 0.5,
        )
    }

    /// Converts a world-space position to `(row, col)`.
    ///
    /// Returns `None` if the position lies outside the grid.
    pub fn world_to_cell(&self, world: Vec2) -> Option<(usize, usize)> {
        let rel_x = world.x - self.origin.x;
        let rel_y = self.origin.y - world.y; // Y increases downward in row space

        if rel_x < 0.0 || rel_y < 0.0 {
            return None;
        }

        let col = (rel_x / self.tile_size) as usize;
        let row = (rel_y / self.tile_size) as usize;

        if row >= self.rows() || col >= self.cols() {
            return None;
        }

        Some((row, col))
    }

    /// Returns `true` if `world` lies within the driveable area of a road tile.
    ///
    /// The driveable area is the tile cell minus a half-wall-thickness inset on
    /// every closed edge. This means:
    /// - Two adjacent road tiles with closed facing edges produce a full
    ///   wall-thickness dead zone at the boundary, preventing pass-through.
    /// - The car collides at the inner face of the visual wall, not at the
    ///   tile boundary.
    ///
    /// Corner tiles use an arc-distance check with the same inset applied to
    /// the arc radius.
    ///
    /// Positions outside the grid bounds always return `false`.
    pub fn is_road_at(&self, world: Vec2) -> bool {
        let Some((row, col)) = self.world_to_cell(world) else {
            return false;
        };

        let tile = self.tile_at(row, col);
        if !tile.is_road() {
            return false;
        }

        let center = self.cell_center(row, col);
        let half = self.tile_size * 0.5;
        let margin = WALL_THICKNESS * 0.5;

        if tile.is_corner() {
            let (arc_center, _, _) = corner_arc_params(tile, center, half);
            return world.distance(arc_center) <= self.tile_size - margin;
        }

        let (open_n, open_s, open_e, open_w) = tile.open_edges();

        if !open_n && world.y > center.y + half - margin { return false; }
        if !open_s && world.y < center.y - half + margin { return false; }
        if !open_e && world.x > center.x + half - margin { return false; }
        if !open_w && world.x < center.x - half + margin { return false; }

        true
    }

    /// Locates the `SpawnPoint` tile and returns `(world_centre, heading_radians)`.
    ///
    /// `SpawnPoint` shares `StraightH` connectivity so the car faces east
    /// (0.0 radians; +X direction in Bevy world space).
    ///
    /// Returns `None` if no `SpawnPoint` tile exists.
    pub fn find_spawn(&self) -> Option<(Vec2, f32)> {
        let (row, col) = self.find_spawn_cell()?;
        Some((self.cell_center(row, col), 0.0))
    }

    /// Locates the `SpawnPoint` tile and returns its `(row, col)` coordinates.
    ///
    /// Returns `None` if no `SpawnPoint` tile exists.
    pub fn find_spawn_cell(&self) -> Option<(usize, usize)> {
        for (row, row_tiles) in self.tiles.iter().enumerate() {
            for (col, &tile) in row_tiles.iter().enumerate() {
                if tile == TilePart::SpawnPoint {
                    return Some((row, col));
                }
            }
        }
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rendering
// ─────────────────────────────────────────────────────────────────────────────

/// Spawns visual sprites for every road tile in the grid.
///
/// Each road tile receives:
/// - A filled dark-grey road surface at z = 0.
/// - Wall sprites at z = 1:
///   - **Corner tiles** (`CornerNW/NE/SW/SE`): a smooth quarter-circle arc
///     approximated with [`ARC_SEGMENTS`] line-segment sprites.
///   - **All other road tiles**: straight white bar sprites on every closed
///     edge that borders a non-road cell.
///
/// Corner road surfaces are rendered as quarter-circle meshes that match the
/// outer wall arc, preventing the road from leaking outside the curved boundary.
///
/// ## Arc geometry (corners)
///
/// For each corner tile the arc is a quarter-circle with:
/// - **Radius = `tile_size`** (equals the full tile side length).
/// - **Centre** at the tile corner diagonally *opposite* the outer wall
///   direction.
/// - **Endpoints** exactly at the two tile corners adjacent to the open edges.
///
/// This guarantees that each arc endpoint lands precisely where the adjacent
/// straight tile's wall bar begins, producing a seamless, gap-free boundary.
///
/// | Tile     | Arc centre (relative to cell centre) | Sweep (CCW) |
/// |----------|--------------------------------------|-------------|
/// | CornerNW | (+half, −half) = SE corner           | 90° → 180°  |
/// | CornerNE | (−half, −half) = SW corner           |  0° →  90°  |
/// | CornerSW | (+half, +half) = NE corner           | 180° → 270° |
/// | CornerSE | (−half, +half) = NW corner           | 270° → 360° |
pub fn render_tile_grid(
    commands: &mut Commands,
    grid: &TrackGrid,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<ColorMaterial>>,
) {
    let road_color     = Color::srgb(0.28, 0.28, 0.28);
    let wall_color     = Color::srgb(0.88, 0.88, 0.88);
    let wall_thickness = WALL_THICKNESS;

    let ts   = grid.tile_size;
    let half = ts * 0.5;
    let road_material = materials.add(ColorMaterial::from(road_color));
    let wall_material = materials.add(ColorMaterial::from(wall_color));

    for row in 0..grid.rows() {
        for col in 0..grid.cols() {
            let tile = grid.tile_at(row, col);
            if !tile.is_road() {
                continue;
            }

            let center = grid.cell_center(row, col);

            if tile.is_corner() {
                // Corner tiles: road surface is a quarter-circle sector that
                // matches the curved outer wall.
                let (arc_center, start_deg, end_deg) = corner_arc_params(tile, center, half);
                spawn_corner_surface(
                    commands,
                    meshes,
                    road_material.clone(),
                    center,
                    arc_center,
                    ts,
                    start_deg,
                    end_deg,
                    ARC_SEGMENTS,
                    0.0,
                );

                // Corner tiles: render one continuous quarter-circle arc wall.
                // The arc covers both closed edges with a smooth curve.
                spawn_arc_mesh(
                    commands,
                    meshes,
                    wall_material.clone(),
                    arc_center,
                    ts, // radius = tile_size
                    start_deg,
                    end_deg,
                    ARC_SEGMENTS,
                    wall_thickness,
                    1.0,
                );
            } else {
                // Road surface — fills the full cell.
                commands.spawn((
                    Sprite {
                        color: road_color,
                        custom_size: Some(Vec2::splat(ts)),
                        ..default()
                    },
                    Transform::from_xyz(center.x, center.y, 0.0),
                ));

                // Non-corner tiles: straight wall bars on every closed edge.
                // Adjacent road tiles may produce overlapping sprites on a
                // shared boundary; this is harmless and visually identical to
                // a single wall.
                let (open_n, open_s, open_e, open_w) = tile.open_edges();

                if !open_n {
                    commands.spawn((
                        Sprite {
                            color: wall_color,
                            custom_size: Some(Vec2::new(ts, wall_thickness)),
                            ..default()
                        },
                        Transform::from_xyz(center.x, center.y + half, 1.0),
                    ));
                }

                if !open_s {
                    commands.spawn((
                        Sprite {
                            color: wall_color,
                            custom_size: Some(Vec2::new(ts, wall_thickness)),
                            ..default()
                        },
                        Transform::from_xyz(center.x, center.y - half, 1.0),
                    ));
                }

                if !open_e {
                    commands.spawn((
                        Sprite {
                            color: wall_color,
                            custom_size: Some(Vec2::new(wall_thickness, ts)),
                            ..default()
                        },
                        Transform::from_xyz(center.x + half, center.y, 1.0),
                    ));
                }

                if !open_w {
                    commands.spawn((
                        Sprite {
                            color: wall_color,
                            custom_size: Some(Vec2::new(wall_thickness, ts)),
                            ..default()
                        },
                        Transform::from_xyz(center.x - half, center.y, 1.0),
                    ));
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Arc helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Spawns a filled quarter-circle mesh for a corner tile's road surface.
fn spawn_corner_surface(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<ColorMaterial>,
    tile_center: Vec2,
    arc_center_world: Vec2,
    radius: f32,
    start_deg: f32,
    end_deg: f32,
    segments: usize,
    z: f32,
) {
    let arc_center_local = arc_center_world - tile_center;
    let sweep = end_deg - start_deg;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(segments * 3);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(segments * 3);

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    let center = [arc_center_local.x, arc_center_local.y, 0.0];

    for i in 0..segments {
        let t0 = i as f32 / segments as f32;
        let t1 = (i + 1) as f32 / segments as f32;
        let a0 = (start_deg + t0 * sweep).to_radians();
        let a1 = (start_deg + t1 * sweep).to_radians();
        let p0 = arc_center_local + Vec2::new(radius * a0.cos(), radius * a0.sin());
        let p1 = arc_center_local + Vec2::new(radius * a1.cos(), radius * a1.sin());

        positions.push(center);
        positions.push([p0.x, p0.y, 0.0]);
        positions.push([p1.x, p1.y, 0.0]);

        uvs.push([0.0, 0.0]);
        uvs.push([0.0, 0.0]);
        uvs.push([0.0, 0.0]);
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(material),
        Transform::from_xyz(tile_center.x, tile_center.y, z),
        GlobalTransform::default(),
        Visibility::Visible,
    ));
}

/// Returns the arc parameters for a corner tile: `(arc_center, start_deg, end_deg)`.
///
/// The arc is always a quarter-circle (90° sweep, counter-clockwise).
/// `start_deg` < `end_deg` in all cases so the caller can sweep linearly.
///
/// See the table in [`render_tile_grid`] for the geometry derivation.
fn corner_arc_params(tile: TilePart, cell_center: Vec2, half: f32) -> (Vec2, f32, f32) {
    let cx = cell_center.x;
    let cy = cell_center.y;

    match tile {
        // Outer arc curves around NW. Centre at SE corner of tile.
        TilePart::CornerNW => (Vec2::new(cx + half, cy - half),  90.0, 180.0),
        // Outer arc curves around NE. Centre at SW corner of tile.
        TilePart::CornerNE => (Vec2::new(cx - half, cy - half),   0.0,  90.0),
        // Outer arc curves around SW. Centre at NE corner of tile.
        TilePart::CornerSW => (Vec2::new(cx + half, cy + half), 180.0, 270.0),
        // Outer arc curves around SE. Centre at NW corner of tile.
        TilePart::CornerSE => (Vec2::new(cx - half, cy + half), 270.0, 360.0),
        _ => unreachable!("corner_arc_params called on non-corner tile"),
    }
}

/// Spawns [`segments`] mesh line segments that approximate an arc.
fn spawn_arc_mesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<ColorMaterial>,
    center: Vec2,
    radius: f32,
    start_deg: f32,
    end_deg: f32,
    segments: usize,
    thickness: f32,
    z: f32,
) {
    let sweep = end_deg - start_deg;

    for i in 0..segments {
        let t0 = i as f32 / segments as f32;
        let t1 = (i + 1) as f32 / segments as f32;

        let a0 = (start_deg + t0 * sweep).to_radians();
        let a1 = (start_deg + t1 * sweep).to_radians();

        let p0 = center + Vec2::new(radius * a0.cos(), radius * a0.sin());
        let p1 = center + Vec2::new(radius * a1.cos(), radius * a1.sin());

        spawn_line_segment_mesh(commands, meshes, material.clone(), p0, p1, thickness, z);
    }
}

/// Spawns a single thin rotated mesh line segment from `start` to `end`.
fn spawn_line_segment_mesh(
    commands: &mut Commands,
    meshes: &mut Assets<Mesh>,
    material: Handle<ColorMaterial>,
    start: Vec2,
    end: Vec2,
    thickness: f32,
    z: f32,
) {
    let delta     = end - start;
    let length    = delta.length();
    let midpoint  = (start + end) * 0.5;
    let angle     = delta.y.atan2(delta.x);
    let half_len  = length * 0.5;
    let half_thk  = thickness * 0.5;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(6);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(6);

    let v0 = [-half_len, -half_thk, 0.0];
    let v1 = [ half_len, -half_thk, 0.0];
    let v2 = [ half_len,  half_thk, 0.0];
    let v3 = [-half_len,  half_thk, 0.0];

    positions.extend_from_slice(&[v0, v1, v2, v0, v2, v3]);
    uvs.extend_from_slice(&[[0.0, 0.0]; 6]);

    let mut mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);

    commands.spawn((
        Mesh2d(meshes.add(mesh)),
        MeshMaterial2d(material),
        Transform::from_xyz(midpoint.x, midpoint.y, z)
            .with_rotation(Quat::from_rotation_z(angle)),
        GlobalTransform::default(),
        Visibility::Visible,
    ));
}
