# NeuroDrive's Progress and Lap System

The progress and lap system converts the car's world-space position into a scalar measure of how far it has travelled along the track. This is a *measurement* system, not an *observation* system — its outputs are used by the environment for reward shaping and episode management, but are deliberately excluded from the agent's observation vector to prevent privileged information leakage.

> **Prerequisites:** [Observation System](observation_system.md), [Car Physics Model](car_physics_model.md)

**Code locations:** `src/maps/centerline.rs`, `src/game/progress.rs`, `src/game/episode.rs`

---

## Centreline Construction

**Code location:** `src/maps/centerline.rs`

The centreline is the reference path along the middle of the track. It is constructed automatically from the tile grid during map loading — no manual waypoint placement is required.

### Tile-Grid Traversal

The algorithm begins at the start tile and traverses the grid's connectivity using a DFS-like walk with a visited set. At each tile, it examines the `TilePart` variant to determine the tile's geometric contribution:

- **Straight tiles** contribute a line segment along the tile's centreline (horizontal or vertical, depending on orientation).
- **Corner tiles** (inner and outer arcs) contribute a quarter-circle arc. Each arc is discretised into 8 evenly-spaced sample points, converting the smooth curve into a polyline approximation.

The visited set prevents the traversal from revisiting tiles, and ambiguous branches (junctions where multiple unvisited neighbours exist) are rejected — the current track design assumes a single non-branching loop.

### Closed Polyline

The traversal produces an ordered sequence of 2D points forming a closed polyline that approximates the track's centreline. Adjacent points are connected by line segments, and the final point connects back to the first.

### Cumulative Arc-Length Prefix Sums

After constructing the polyline, a prefix-sum array of cumulative arc lengths is computed. For segment `i`, the cumulative length is the sum of all segment lengths from segment 0 through segment `i-1`. The total track length is the sum of all segment lengths.

This prefix array enables O(1) conversion from segment index to arc-length position: given that a point lies on segment `i` at parameter `t ∈ [0,1]`, its arc-length position is `prefix[i] + t × segment_length[i]`.

---

## Projection: Position to Progress

**Code location:** `src/game/progress.rs`

Given the car's world-space position `p`, the system must find the closest point on the centreline and compute the corresponding arc-length distance.

### Brute-Force Closest-Point Search

The projection uses a brute-force O(N) search over all centreline segments. For each segment defined by endpoints `a` and `b`:

1. Compute the direction vector `d = b - a`.
2. Compute the projection parameter `t = clamp(dot(p - a, d) / |d|², 0, 1)`.
3. Compute the closest point on the segment: `q = a + t × d`.
4. Compute the squared distance `|p - q|²`.

The segment with the smallest squared distance wins. The clamping of `t` to `[0, 1]` ensures the closest point stays within the segment rather than extrapolating beyond its endpoints.

### Arc-Length and Fraction

Once the nearest segment `i` and parameter `t` are found:

- **Arc-length position:** `s = prefix[i] + t × segment_length[i]`
- **Progress fraction:** `fraction = s / total_length`, yielding a value in `[0, 1)` representing how far around the loop the car has travelled.

The progress fraction is the value used for reward computation and lap detection. It increases as the car moves forward along the track and wraps from near-1.0 back to near-0.0 when completing a lap.

### Performance Note

The O(N) brute-force search is acceptable because N (the number of centreline segments) is small — typically a few hundred for a standard track. At 60 Hz, a few hundred dot products per tick is negligible compared to the ray casting and neural network forward pass. If tracks grew to thousands of segments, a spatial acceleration structure (grid bucketing or bounding-volume hierarchy) could replace the brute-force search without changing the interface.

---

## Lap Detection

**Code location:** `src/game/episode.rs`

Lap completion is detected using an **arm/trigger hysteresis** mechanism to prevent false positives from oscillation near the start/finish line.

### The Problem

Progress fraction wraps from near-1.0 to near-0.0 when the car crosses the start/finish line. If lap completion were detected simply by checking for a decrease in progress, any small backward movement or jitter would falsely register a lap. Conversely, a car driving in reverse past the start line should not count as a lap.

### Arm/Trigger Hysteresis

The mechanism uses two thresholds:

1. **Arm threshold (25%):** The lap detector becomes *armed* when the car's progress fraction exceeds 25% of the track. Before this point, no lap can be triggered. This ensures the car has made meaningful forward progress before a lap completion is possible.

2. **Trigger condition:** Once armed, a lap is triggered when the progress fraction wraps from `≥ 85%` to `≤ 15%` between consecutive ticks. This large gap (85% → 15%) can only occur when the car crosses the start/finish line in the forward direction. Small jitter or backward movement cannot produce a wrap of this magnitude.

After triggering, the arm flag resets, requiring the car to travel past 25% again before the next lap can count. This prevents double-counting if the car oscillates near the start/finish region.

### Why These Thresholds

The 25% arm threshold is conservative — it guarantees that at least a quarter of the track has been traversed. The 85%/15% trigger window is wide enough to reject noise (the car would need to teleport 70% of the track backward to produce a false trigger) but narrow enough to capture legitimate crossings at any reasonable speed.

---

## Measurement vs. Observation

A critical design distinction: TrackProgress is **measurement**, not **observation**.

- **Measurement** is environment-internal truth used for reward computation, episode termination, and analytics. It requires global knowledge of the track layout (the centreline, the total length, the start/finish position).
- **Observation** is what the agent perceives through its sensors. The observation vector contains only locally-observable quantities: ray distances, speed, heading error, and angular velocity.

If TrackProgress were included in the observation vector, the agent could learn to optimise a trivially correlated signal (increase progress fraction → get reward) without developing genuine driving competence. The resulting policy would overfit to the specific track layout and fail to transfer. By keeping progress as a measurement that shapes the reward but never enters the observation, the agent must learn to drive from perception alone.

This separation mirrors the real-world distinction between what a GPS tells the race control system (absolute position) and what a driver sees through the windshield (the road ahead). The agent is the driver, not the race control system.

---

## Integration with Other Systems

The progress system feeds into several downstream consumers:

- **Reward function:** Progress delta (change in arc-length position per tick) is a primary reward signal, incentivising forward movement.
- **Episode termination:** Episode boundaries can be defined by lap completion or timeout.
- **Analytics:** Best progress per episode, lap completion rates, and crash positions relative to track fraction are recorded for training diagnostics.

> **See also:** [Observation System](observation_system.md) for why TrackProgress is excluded from observations, [Analytics Pipeline](analytics_pipeline.md) for how progress data is recorded, [Architecture Decisions](../architecture_decisions.md) for the tile-grid track design that enables this system.
