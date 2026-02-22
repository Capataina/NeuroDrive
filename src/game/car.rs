use bevy::prelude::*;

/// Marker component identifying the player's car entity.
#[derive(Component)]
pub struct Car {
    pub velocity: Vec2,
    pub rotation_speed: f32,
    pub thrust: f32,
    pub drag: f32,
}

impl Default for Car {
    fn default() -> Self {
        Self {
            velocity: Vec2::ZERO,
            rotation_speed: 4.0,
            thrust: 1500.0,
            drag: 0.985,
        }
    }
}

/// Car dimensions for collision detection and rendering.
pub const CAR_WIDTH: f32 = 12.0;
pub const CAR_HEIGHT: f32 = 6.0;

/// Spawns the car entity at a given position and rotation.
pub fn spawn_car(commands: &mut Commands, position: Vec2, rotation: f32) {
    info!(
        "Spawn car entity at ({:.1}, {:.1}) rot {:.2}.",
        position.x, position.y, rotation
    );
    commands.spawn((
        Sprite {
            color: Color::srgb(0.9, 0.2, 0.2),
            custom_size: Some(Vec2::new(CAR_WIDTH, CAR_HEIGHT)),
            ..default()
        },
        Transform::from_xyz(position.x, position.y, 10.0)
            .with_rotation(Quat::from_rotation_z(rotation)),
        Car::default(),
    ));
}

/// Handles keyboard input to control the car.
pub fn car_control_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut query: Query<(&mut Transform, &mut Car)>,
) {
    let dt = time.delta_secs();

    for (mut transform, mut car) in query.iter_mut() {
        // Rotation: A = left, D = right
        if keyboard.pressed(KeyCode::KeyA) {
            transform.rotate_z(car.rotation_speed * dt);
        }
        if keyboard.pressed(KeyCode::KeyD) {
            transform.rotate_z(-car.rotation_speed * dt);
        }

        // Thrust: W = forward
        if keyboard.pressed(KeyCode::KeyW) {
            let forward = transform.rotation * Vec3::X;
            let thrust = car.thrust;
            car.velocity += Vec2::new(forward.x, forward.y) * thrust * dt;
        }

        // Apply drag
        let drag = car.drag;
        car.velocity *= drag;

        // Update position
        transform.translation.x += car.velocity.x * dt;
        transform.translation.y += car.velocity.y * dt;
    }
}
