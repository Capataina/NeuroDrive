# NeuroDrive Architecture

## Repository Overview

NeuroDrive is a brain-inspired AI research project built around a 2D top-down racing environment.
The codebase is written in Rust using Bevy for simulation and rendering.

## Directory Structure

```
NeuroDrive/
├── src/
│   ├── main.rs              # Application entry point
│   ├── game/
│   │   ├── mod.rs           # Game module exports
│   │   ├── car.rs           # Car component and control systems
│   │   ├── collision.rs     # Collision detection and reset logic
│   │   └── plugin.rs        # GamePlugin bundling all game systems
│   └── maps/
│       ├── mod.rs           # Maps module exports
│       ├── grid.rs          # TrackGrid definition and tile rendering
│       ├── track.rs         # Track component
│       ├── parts/
│       │   └── mod.rs       # TilePart enum and connectivity rules
│       └── monaco.rs        # Sepang-inspired circuit definition
├── context/
│   └── ARCHITECTURE.md      # This file
├── Cargo.toml
└── README.md                # Project vision (immutable)
```

## Current Implementation Status

### Implemented

- Modular Bevy application structure
- 2D car entity with physics (velocity, drag, rotation)
- Keyboard controls: W (thrust), A (turn left), D (turn right)
- Grid-based tile track system (`TrackGrid` + `TilePart`)
- Tile types: straights (H/V), corners (NW/NE/SW/SE), T-junctions, crossroads, spawn point
- Connectivity rules per tile type via `open_edges()`
- Sepang-inspired circuit layout on a 14×9 tile grid
- Tile rendering: filled road surfaces, straight wall bars on closed edges, quarter-circle arc walls on corner tiles
- Corner road surfaces rendered as clipped quarter-circle meshes to prevent surface leakage past curved walls
- Collision detection via `grid.is_road_at()` (replaces old polygon-based point-in-polygon)
- Automatic reset on barrier collision
- Car spawns at tile-grid-derived start position
- Start/finish line marker

### Not Yet Implemented

- Raycast sensor system
- Progress measurement along centerline
- Neural brain system
- Learning mechanisms
- Multiple tracks
- Telemetry/observability

## Subsystem Responsibilities

### main.rs

- Bevy app initialisation
- Window configuration
- Plugin registration order (track before game)

### game/car.rs

- `Car` component with velocity, thrust, drag, rotation_speed
- `spawn_car()` function for creating car entities
- `car_control_system()` for handling W/A/D input

### game/collision.rs

- `CollisionEvent` for signalling barrier contact
- `collision_detection_system()` checks car position against track grid
- `handle_collision_system()` respawns car on collision
- `point_in_polygon()` ray-casting algorithm (legacy, may be removed)

### game/plugin.rs

- `GamePlugin` bundles all game systems
- Handles system ordering (control → collision → reset)
- Spawns camera and car on startup

### maps/parts/mod.rs

- `TilePart` enum defining all tile types and the `Empty` sentinel
- `open_edges()` returns `(north, south, east, west)` connectivity per tile
- `is_road()` and `is_corner()` classification helpers
- Connectivity reference table in doc comments

### maps/grid.rs

- `TrackGrid` struct: row-major tile storage, origin, tile_size
- `cell_center()`, `world_to_cell()`, `is_road_at()` for spatial queries
- `find_spawn()` locates the `SpawnPoint` tile
- `render_tile_grid()` spawns all visual sprites:
  - Road surfaces (flat squares for straights, quarter-circle meshes for corners)
  - Straight wall bars on every closed edge of non-corner tiles
  - Quarter-circle arc walls on corner tiles
- Arc geometry helpers (`corner_arc_params`, `spawn_arc_mesh`, `spawn_line_segment_mesh`)

### maps/track.rs

- `Track` component carrying the grid, spawn position, and spawn heading
- Queried by collision and game systems

### maps/monaco.rs

- `MonacoPlugin` spawns the Sepang-inspired track on startup
- `build_tiles()` defines the 14×9 tile grid layout
- `render_finish_line()` places the start/finish marker
- Derives spawn position and heading from the grid's `SpawnPoint` tile

## Execution Flow

1. `MonacoPlugin` runs first (Startup): builds tile grid, renders sprites, spawns Track entity
2. `GamePlugin` runs second (Startup): spawns Camera and Car at track spawn position
3. Each frame (Update):
   - `car_control_system`: reads input, updates velocity and rotation
   - `collision_detection_system`: checks if car is off-track via `grid.is_road_at()`
   - `handle_collision_system`: respawns car if collision detected

## Dependencies

- `bevy 0.18.0` — Game engine for ECS, rendering, input handling
