use bevy::prelude::*;

use crate::maps::centerline::{GridDir, TrackCenterline};
use crate::maps::grid::{TrackGrid, render_tile_grid};
use crate::maps::parts::TilePart;
use crate::maps::track::Track;

/// Plugin that spawns the Sepang-inspired circuit.
///
/// The layout is loosely based on the Malaysian Grand Prix circuit:
/// a long asymmetric main straight, a cascading S-curve section on the right
/// that steps inward over three hairpin pairs, a shorter back straight, and an
/// outward hairpin loop on the left before returning to the main straight.
pub struct MonacoPlugin;

impl Plugin for MonacoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_track);
    }
}

/// World-space side length of each grid cell in pixels.
const TILE_SIZE: f32 = 100.0;

/// Builds and spawns the tile grid, derives spawn data, and emits all visual
/// sprites. The resulting `Track` entity is consumed by collision and game systems.
fn spawn_track(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    let tiles = build_tiles();

    let rows = tiles.len();
    let cols = tiles[0].len();

    // Centre the 14×9 grid in the 1600×900 window.
    let origin = Vec2::new(
        -(cols as f32 * TILE_SIZE) * 0.5,
        (rows as f32 * TILE_SIZE) * 0.5,
    );

    let grid = TrackGrid::new(tiles, TILE_SIZE, origin);

    let spawn_cell = grid
        .find_spawn_cell()
        .expect("Track grid must contain exactly one SpawnPoint tile.");

    let (spawn_pos, spawn_rot) = grid
        .find_spawn()
        .expect("Track grid must contain exactly one SpawnPoint tile.");

    let centerline = TrackCenterline::build_closed_loop(&grid, spawn_cell, GridDir::East)
        .expect("Track grid connectivity must form a single closed loop.");

    info!(
        "Sepang track spawned. Grid {}×{}. Car spawn ({:.0},{:.0}) rot {:.2}.",
        grid.cols(),
        grid.rows(),
        spawn_pos.x,
        spawn_pos.y,
        spawn_rot
    );
    info!("Centreline length: {:.0}px.", centerline.total_length());

    render_tile_grid(&mut commands, &grid, &mut meshes, &mut materials);
    render_finish_line(&mut commands, &grid);

    commands.spawn(Track {
        grid,
        spawn_position: spawn_pos,
        spawn_rotation: spawn_rot,
        centerline,
    });
}

/// Defines the Sepang-inspired tile layout on a 14-column × 9-row grid.
///
/// Legend:
/// ```text
///  .  = Empty         NW = CornerNW    NE = CornerNE
///  H  = StraightH     SW = CornerSW    SE = CornerSE
///  V  = StraightV     SP = SpawnPoint
/// ```
///
/// ## Tile grid
///
/// ```text
///       col: 0  1  2  3  4  5  6  7  8  9 10 11 12 13
/// row 0:     .  .  .  .  .  .  .  .  .  .  .  .  .  .
/// row 1:     . NW  H  H SP  H  H  H  H  H  H  H NE  .   ← main straight
/// row 2:     .  V NW  H NE NW  H  H  H NE NW  H SE  .
/// row 3:     . SW SE NW SE SW  H  H NE  V SW  H NE  .
/// row 4:     . NW NE  V NW NE NW NE  V  V NW  H SE  .
/// row 5:     .  V  V SW SE SW SE  V  V  V SW  H NE  .
/// row 6:     .  V SW  H  H  H  H SE  V  V NW NE  V  .
/// row 7:     . SW  H  H  H  H  H  H SE SW SE SW SE  .   ← back straight
/// row 8:     .  .  .  .  .  .  .  .  .  .  .  .  .  .
/// ```
///
/// ## Circuit shape
///
/// The track forms a single closed loop. Key features:
///
/// - **Top straight** (row 1, cols 1–12): 11-tile main straight with spawn.
/// - **Right cascading S-curves** (cols 9–12, rows 2–7): three staircase
///   descents connected by short horizontal runs, ending in a zigzag at
///   the bottom.
/// - **Bottom connection** (row 7, cols 1–12): back straight and zigzag
///   linking left and right halves.
/// - **Left vertical section** (cols 1–2, rows 2–7): two parallel vertical
///   runs with hairpin connections.
/// - **Central serpentine** (cols 3–8, rows 3–6): nested hairpin pairs and
///   internal loops.
///
/// ## Connectivity path (clockwise from spawn)
///
/// ```text
/// SP(1,4) →E H(1,5–11) → NE(1,12)
///   ↓S SE(2,12) ←W H(2,11) ← NW(2,10)
///   ↓S SW(3,10) →E H(3,11) → NE(3,12)
///   ↓S SE(4,12) ←W H(4,11) ← NW(4,10)
///   ↓S SW(5,10) →E H(5,11) → NE(5,12)
///   ↓S V(6,12) ↓S SE(7,12) ←W SW(7,11)
///   ↑N NE(6,11) ←W NW(6,10) ↓S SE(7,10) ←W SW(7,9)
///   ↑N V(6,9) ↑N V(5,9) ↑N V(4,9) ↑N V(3,9)
///   ↑N NE(2,9) ←W H(2,6–8) ← NW(2,5)
///   ↓S SW(3,5) →E H(3,6–7) → NE(3,8)
///   ↓S V(4,8) ↓S V(5,8) ↓S V(6,8)
///   ↓S SE(7,8) ←W H(7,2–7) ← SW(7,1)
///   ↑N V(6,1) ↑N V(5,1) ↑N NW(4,1) →E NE(4,2)
///   ↓S V(5,2) ↓S SW(6,2) →E H(6,3–6) → SE(6,7)
///   ↑N V(5,7) ↑N NE(4,7) ←W NW(4,6)
///   ↓S SE(5,6) ←W SW(5,5) ↑N NE(4,5) ←W NW(4,4)
///   ↓S SE(5,4) ←W SW(5,3) ↑N V(4,3) ↑N NW(3,3) →E SE(3,4)
///   ↑N NE(2,4) ←W H(2,3) ← NW(2,2)
///   ↓S SE(3,2) ←W SW(3,1) ↑N V(2,1) ↑N NW(1,1)
///   →E H(1,2–3) → SP(1,4) — loop closes
/// ```
///
/// Every open edge of every road tile matches its neighbour's open edge.
#[allow(non_snake_case)]
fn build_tiles() -> Vec<Vec<TilePart>> {
    use TilePart::*;

    let E = Empty;
    let NW = CornerNW;
    let NE = CornerNE;
    let SW = CornerSW;
    let SE = CornerSE;
    let H = StraightH;
    let V = StraightV;
    let SP = SpawnPoint;

    //       col: 0   1   2   3   4   5   6   7   8   9  10  11  12  13
    vec![
        vec![E, E, E, E, E, E, E, E, E, E, E, E, E, E], // row 0
        vec![E, NW, H, H, SP, H, H, H, H, H, H, H, NE, E], // row 1
        vec![E, V, NW, H, NE, NW, H, H, H, NE, NW, H, SE, E], // row 2
        vec![E, SW, SE, NW, SE, SW, H, H, NE, V, SW, H, NE, E], // row 3
        vec![E, NW, NE, V, NW, NE, NW, NE, V, V, NW, H, SE, E], // row 4
        vec![E, V, V, SW, SE, SW, SE, V, V, V, SW, H, NE, E], // row 5
        vec![E, V, SW, H, H, H, H, SE, V, V, NW, NE, V, E], // row 6
        vec![E, SW, H, H, H, H, H, H, SE, SW, SE, SW, SE, E], // row 7
        vec![E, E, E, E, E, E, E, E, E, E, E, E, E, E], // row 8
    ]
}

/// Renders the start/finish line as a white vertical stripe.
///
/// Placed at the western boundary of column 3 — the first `StraightH` tile
/// after the `CornerNW`. The `SpawnPoint` occupies column 4, giving exactly
/// one full tile (100 px) of clear road between the line and the spawn point,
/// so the car never overlaps the finish line on spawn or collision reset.
fn render_finish_line(commands: &mut Commands, grid: &TrackGrid) {
    let finish_col = 4usize;
    let finish_row = 1usize;
    let tile_center = grid.cell_center(finish_row, finish_col);

    // Western edge of col 3 = eastern edge of col 2 (the NW corner).
    let x = tile_center.x - grid.tile_size * 0.5;
    let y = tile_center.y;

    commands.spawn((
        Sprite {
            color: Color::srgb(1.0, 1.0, 1.0),
            custom_size: Some(Vec2::new(5.0, grid.tile_size)),
            ..default()
        },
        Transform::from_xyz(x, y, 2.0),
    ));
}
