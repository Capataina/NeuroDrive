# Raycast Observation Systems

## 1. What Is This Pattern?

A raycast observation system gives an agent a simplified model of its surroundings by casting virtual rays outward from the agent and measuring the distance to the nearest obstacle along each ray. The resulting distance vector becomes the agent's sensory input — analogous to whiskers, sonar pings, or lidar beams.

Raycasting decouples perception from world complexity. The agent never sees raw geometry; it sees a fixed-size vector of distances. This makes the observation dimensionally stable regardless of environment detail, which is essential for neural network inputs.

## 2. When To Use This Pattern

**Good for:**
- RL agents that need a fixed-size observation vector from a variable environment
- 2D or simple 3D worlds where full rendering is unnecessary
- Systems requiring deterministic, reproducible sensor readings
- Environments where obstacle geometry is implicit (e.g., grid lookups rather than mesh intersection)

**Not good for:**
- Agents that need texture, colour, or semantic information (use vision/CNN instead)
- Very dense environments where thousands of rays are needed for adequate coverage
- Scenarios requiring exact collision normals or surface properties

## 3. Core Concept

### Two-Phase Raycasting

Naively testing every point along a ray at fine resolution is expensive. A two-phase approach provides accuracy efficiently:

**Phase 1 — Coarse ray marching:** Step along the ray direction at a fixed interval. At each step, test whether the point is inside an obstacle. Stop at the first hit.

**Phase 2 — Binary-search refinement (bisection):** Once a hit is found between step *n-1* (clear) and step *n* (hit), bisect the interval repeatedly to narrow down the exact boundary.

### Worked Example

A ray cast at 45° from position `(100, 200)`, step size `3.0`, max range `375`:

**Direction vector:** `(cos(45°), sin(45°)) = (0.707, 0.707)`

**Phase 1 — March:**
- Step 0: `(100, 200)` → clear
- Step 1: `(102.1, 202.1)` → clear
- Step 2: `(104.2, 204.2)` → clear
- ...
- Step 40: `(184.9, 284.9)` → clear
- Step 41: `(187.0, 287.0)` → **hit!**

**Phase 2 — Bisect** between distances `120.0` (clear) and `123.0` (hit):
- Iteration 1: test `121.5` → clear → narrow to `[121.5, 123.0]`
- Iteration 2: test `122.25` → hit → narrow to `[121.5, 122.25]`
- Iteration 3: test `121.875` → clear → narrow to `[121.875, 122.25]`

After 8 iterations, the interval width is `3.0 / 2^8 ≈ 0.012` — sub-pixel precision from a coarse starting step.

### Observation Vector Construction

Raw ray distances are normalised to `[0, 1]` by dividing by `max_range`. A hit at distance 122 with max range 375 becomes `122 / 375 = 0.325`. A miss (no hit within range) becomes `1.0`. Additional state variables (speed, heading, angular velocity) are appended with their own normalisation scales.

## 4. Key Design Decisions

| Decision | Option A | Option B |
|---|---|---|
| Ray count | Few rays (fast, coarse perception) | Many rays (rich perception, slower) |
| Angular spread | Narrow (focused forward) | Wide (peripheral awareness) |
| Refinement method | None (fast, ~step-size error) | Binary search (precise, log₂ more tests) |
| Miss encoding | max_range (numerically large) | 1.0 after normalisation (bounded) |

**Key trade-off:** Step size vs performance. Smaller steps find boundaries more precisely but cost more per ray. The two-phase approach decouples these concerns: use a large step for speed, then bisect for precision.

## 5. Simplified Example Implementation

```python
from math import cos, sin, radians, sqrt

def cast_ray(origin, angle_deg, step, max_range, is_obstacle):
    dx = cos(radians(angle_deg))
    dy = sin(radians(angle_deg))
    # Phase 1: coarse march
    prev_dist = 0.0
    dist = 0.0
    while dist <= max_range:
        x = origin[0] + dx * dist
        y = origin[1] + dy * dist
        if is_obstacle(x, y):
            break
        prev_dist = dist
        dist += step
    else:
        return max_range  # miss

    # Phase 2: binary search refinement
    lo, hi = prev_dist, dist
    for _ in range(8):
        mid = (lo + hi) / 2.0
        x = origin[0] + dx * mid
        y = origin[1] + dy * mid
        if is_obstacle(x, y):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0

def build_observation(origin, heading, speed, ang_vel,
                      ray_angles, step, max_range, is_obstacle):
    obs = []
    for angle_offset in ray_angles:
        d = cast_ray(origin, heading + angle_offset,
                     step, max_range, is_obstacle)
        obs.append(d / max_range)  # normalise to [0, 1]
    obs.append(speed / 1200.0)     # normalise speed
    obs.append(ang_vel / 10.0)     # normalise angular velocity
    return obs
```

## 6. How NeuroDrive Implements This

NeuroDrive's sensor system casts **11 rays** spanning **−150° to +150°** in 30° increments relative to the car's heading. Each ray uses `step_size = 3.0` and `max_range = 375.0`.

**Two-phase pipeline:** The system is split into two distinct stages:
1. `SensorReadings` — raw ray distances (11 floats) produced by the raycast system.
2. `ObservationVector` — a normalised 14-dimensional vector consumed by the neural network.

**Observation vector composition:**
- Dimensions 0–10: ray distances normalised by `max_range` (375.0)
- Dimension 11: speed normalised by 1200.0
- Dimension 12: heading error (signed angle between car heading and ideal track direction)
- Dimension 13: angular velocity normalised by 10.0

**Heading error calculation:** Uses `signed_angle_between`, which computes the angular difference via `atan2(cross, dot)`. This correctly handles wrapping across the ±π boundary, producing a value in `[−π, π]` that is then normalised by `π` to `[−1, 1]`.

**Precision:** 8 bisection iterations on a step size of 3.0 yield precision of `3.0 / 256 ≈ 0.012` world units — well below the car's physical size (≈30 units) and sufficient for learning.

**Performance:** 11 rays × ~125 march steps + 8 bisection steps = ~1,463 grid lookups per tick. Because `is_road_at()` is O(1) via the grid, this completes in microseconds.

## 7. Variations

- **Fan-cast / cone-cast:** Cast a wedge and return the nearest hit within the cone. Smoother but more expensive.
- **Multi-bounce raycasting:** Rays reflect off surfaces; used in rendering (ray tracing) but rarely needed for RL perception.
- **Depth-image observation:** Render a 1D depth buffer from a virtual camera. More realistic but requires a rendering pipeline.
- **Lidar-style 360° sweep:** Uniform angular spacing across the full circle. Common in autonomous-vehicle simulation (CARLA, AirSim).

## 8. Common Pitfalls

- **Step size larger than thin obstacles:** A ray can step entirely over a thin wall. Ensure step size is smaller than the thinnest obstacle or use swept tests.
- **Unnormalised observations:** Feeding raw pixel/unit distances into a neural network creates scale-dependent learning. Always normalise to a bounded range.
- **Asymmetric ray layouts:** If rays are not symmetric about the heading, the agent develops a turning bias. Use symmetric angular offsets unless asymmetry is intentional.
- **Ignoring the "miss" case:** Encoding a miss as `0.0` (same as "obstacle at zero distance") confuses the network. Use `1.0` (maximum normalised distance) for misses.
- **Heading error discontinuity:** Naive angle subtraction without `atan2` wrapping produces jumps at ±180°. Always use the signed-angle formula.

## 9. Projects That Use This Pattern

- **OpenAI Gym CarRacing-v2:** Uses pixel rendering, but many custom wrappers replace it with raycast observations for faster training.
- **Unity ML-Agents RayPerceptionSensor:** Built-in ray-based sensor component configurable with ray count, spread, and tag filtering.
- **CARLA Simulator:** Provides lidar sensor actors that generate point clouds — the 3D equivalent of 2D raycasting.

## 10. Glossary

| Term | Definition |
|---|---|
| **Ray marching** | Stepping along a ray at fixed intervals to find the first intersection |
| **Binary-search refinement** | Bisecting the interval between a clear point and a hit point to localise the boundary |
| **Observation vector** | The fixed-size numeric array an agent receives as its sensory input each tick |
| **Normalisation** | Scaling raw values to a bounded range (typically [0, 1] or [−1, 1]) |
| **Heading error** | Signed angular difference between the agent's facing direction and the desired direction |
| **Step size** | The distance increment used during the coarse march phase |
| **Max range** | The furthest distance a ray can travel before returning a "miss" |

## 11. Recommended Materials

- **"Ray Casting in 2D Game Engines"** (Lode's Computer Graphics Tutorial) — Clear walkthrough of 2D raycasting with diagrams, originally for Wolfenstein-style renderers.
- **"Artificial Intelligence: A Modern Approach"** by Russell & Norvig, Chapter 26 (Robotics: Perception) — Covers sensor models including range finders and their probabilistic treatment.
- **Unity ML-Agents Documentation: RayPerceptionSensor** — Practical reference for configuring ray-based observations in a production RL framework.
