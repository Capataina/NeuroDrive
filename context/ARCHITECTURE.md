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
│       ├── track.rs         # Track component and rendering utilities
│       └── monaco.rs        # Monaco-inspired F1 circuit
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
- Track system with inner/outer boundary polygons
- Simplified Monaco-inspired loop with non-overlapping boundaries
- Collision detection using point-in-polygon algorithm
- Automatic reset on barrier collision
- Car spawns at track start position

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
- `collision_detection_system()` checks car position against track boundaries
- `handle_collision_system()` respawns car on collision
- `point_in_polygon()` ray-casting algorithm

### game/plugin.rs

- `GamePlugin` bundles all game systems
- Handles system ordering (control → collision → reset)
- Spawns camera and car on startup

### maps/track.rs

- `Track` component storing boundary polygons and spawn point
- `render_track_boundaries()` draws line segments
- Boundary rendering as connected sprites

### maps/monaco.rs

- `MonacoPlugin` spawns the Monaco track
- `generate_outer_boundary()` and `generate_inner_boundary()` define circuit shape
- Start/finish line marker

## Execution Flow

1. `MonacoPlugin` runs first (Startup): spawns Track entity and renders boundaries
2. `GamePlugin` runs second (Startup): spawns Camera and Car at track spawn position
3. Each frame (Update):
   - `car_control_system`: reads input, updates velocity and rotation
   - `collision_detection_system`: checks if car is off-track
   - `handle_collision_system`: respawns car if collision detected

## Dependencies

- `bevy 0.18.0` — Game engine for ECS, rendering, input handling
