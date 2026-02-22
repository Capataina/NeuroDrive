use bevy::prelude::*;

/// Component representing a race track with inner and outer boundaries.
#[derive(Component)]
pub struct Track {
    pub outer_boundary: Vec<Vec2>,
    pub inner_boundary: Vec<Vec2>,
    pub spawn_position: Vec2,
    pub spawn_rotation: f32,
}

/// Plugin trait for tracks to spawn themselves.
pub trait TrackPlugin: Plugin {
    fn track_name(&self) -> &'static str;
}

/// Renders track boundaries as line segments.
pub fn render_track_boundaries(
    commands: &mut Commands,
    outer: &[Vec2],
    inner: &[Vec2],
    outer_color: Color,
    inner_color: Color,
) {
    // Render outer boundary
    render_boundary_lines(commands, outer, outer_color);
    // Render inner boundary
    render_boundary_lines(commands, inner, inner_color);
}

/// Renders a closed polygon as connected line sprites.
fn render_boundary_lines(commands: &mut Commands, points: &[Vec2], color: Color) {
    let n = points.len();
    if n < 2 {
        return;
    }

    for i in 0..n {
        let start = points[i];
        let end = points[(i + 1) % n];

        let midpoint = (start + end) / 2.0;
        let direction = end - start;
        let length = direction.length();
        let angle = direction.y.atan2(direction.x);

        commands.spawn((
            Sprite {
                color,
                custom_size: Some(Vec2::new(length, 3.0)),
                ..default()
            },
            Transform::from_xyz(midpoint.x, midpoint.y, 1.0)
                .with_rotation(Quat::from_rotation_z(angle)),
        ));
    }
}
