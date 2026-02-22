use bevy::prelude::*;
use crate::maps::track::{render_track_boundaries, Track};

/// Monaco Grand Prix circuit — loose approximation of the real F1 street circuit.
///
/// The layout keeps the iconic overall flow while avoiding self-intersection,
/// so the inner/outer boundaries remain stable for collisions and rendering.
pub struct MonacoPlugin;

impl Plugin for MonacoPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_monaco_track);
    }
}

/// Half the track width. The full driveable surface is twice this value.
/// Track width (distance between inner and outer boundaries).
const TRACK_WIDTH: f32 = 52.0;

fn spawn_monaco_track(mut commands: Commands) {
    let outer = generate_outer_boundary();
    let inner = generate_inner_boundary();

    info!(
        "Spawning Monaco track. Outer points: {}, inner points: {}.",
        outer.len(),
        inner.len(),
    );

    let spawn_position = (outer[0] + inner[0]) * 0.5;
    let spawn_dir = (outer[1] - outer[0]).normalize_or_zero();
    let spawn_rotation = spawn_dir.y.atan2(spawn_dir.x);

    let track = Track {
        outer_boundary: outer.clone(),
        inner_boundary: inner.clone(),
        spawn_position,
        spawn_rotation, // Aligned with the first straight
    };

    commands.spawn(track);

    let outer_color = Color::srgb(0.85, 0.85, 0.85);
    let inner_color = Color::srgb(0.65, 0.65, 0.65);
    render_track_boundaries(&mut commands, &outer, &inner, outer_color, inner_color);

    spawn_start_line(&mut commands, &outer, &inner);
}

fn generate_outer_boundary() -> Vec<Vec2> {
    vec![
        // ===== START / FINISH STRAIGHT (south side, heading east) =====
        Vec2::new(-420.0, -260.0),
        Vec2::new(260.0, -260.0),

        // ===== SAINTE DÉVOTE / CLIMB (east to north) =====
        Vec2::new(330.0, -230.0),
        Vec2::new(390.0, -160.0),
        Vec2::new(410.0, -70.0),
        Vec2::new(400.0, 30.0),
        Vec2::new(360.0, 120.0),
        Vec2::new(300.0, 190.0),
        Vec2::new(220.0, 240.0),

        // ===== CASINO SQUARE (top) =====
        Vec2::new(120.0, 270.0),
        Vec2::new(0.0, 275.0),
        Vec2::new(-120.0, 265.0),
        Vec2::new(-220.0, 235.0),

        // ===== MIRABEAU / HAIRPIN (left side tight section) =====
        Vec2::new(-300.0, 190.0),
        Vec2::new(-340.0, 130.0),
        Vec2::new(-360.0, 60.0),
        Vec2::new(-360.0, -10.0),
        Vec2::new(-340.0, -80.0),
        Vec2::new(-300.0, -130.0),
        Vec2::new(-230.0, -170.0),
        Vec2::new(-150.0, -195.0),

        // ===== PORTIER / TUNNEL (back to straight) =====
        Vec2::new(-60.0, -215.0),
        // → polygon closes back to first point (-420, -260)
    ]
}

fn generate_inner_boundary() -> Vec<Vec2> {
    vec![
        // ===== START / FINISH STRAIGHT (south side, heading east) =====
        Vec2::new(-380.0, -220.0),
        Vec2::new(210.0, -220.0),

        // ===== SAINTE DÉVOTE / CLIMB (east to north) =====
        Vec2::new(280.0, -200.0),
        Vec2::new(320.0, -140.0),
        Vec2::new(335.0, -70.0),
        Vec2::new(325.0, 20.0),
        Vec2::new(290.0, 100.0),
        Vec2::new(240.0, 160.0),
        Vec2::new(180.0, 200.0),

        // ===== CASINO SQUARE (top) =====
        Vec2::new(100.0, 225.0),
        Vec2::new(0.0, 230.0),
        Vec2::new(-100.0, 220.0),
        Vec2::new(-190.0, 195.0),

        // ===== MIRABEAU / HAIRPIN (left side tight section) =====
        Vec2::new(-250.0, 160.0),
        Vec2::new(-285.0, 115.0),
        Vec2::new(-300.0, 60.0),
        Vec2::new(-300.0, -5.0),
        Vec2::new(-285.0, -65.0),
        Vec2::new(-255.0, -100.0),
        Vec2::new(-200.0, -125.0),
        Vec2::new(-130.0, -145.0),

        // ===== PORTIER / TUNNEL (back to straight) =====
        Vec2::new(-60.0, -160.0),
        // → polygon closes back to first point (-380, -220)
    ]
}

/// Spawns a white start/finish line perpendicular to the track at the start
/// of the pit straight.
fn spawn_start_line(commands: &mut Commands, outer: &[Vec2], inner: &[Vec2]) {
    if outer.len() < 2 || inner.len() < 1 {
        return;
    }

    let start = (outer[0] + inner[0]) * 0.5;
    let next = outer[1];
    let dir = (next - start).normalize_or_zero();
    let perp_angle = dir.y.atan2(dir.x) + std::f32::consts::FRAC_PI_2;

    commands.spawn((
        Sprite {
            color: Color::srgb(1.0, 1.0, 1.0),
            custom_size: Some(Vec2::new(4.0, TRACK_WIDTH)),
            ..default()
        },
        Transform::from_xyz(start.x, start.y, 2.0)
            .with_rotation(Quat::from_rotation_z(perp_angle)),
    ));
}
