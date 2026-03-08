# Grid-Based Spatial Systems

## 1. What Is This Pattern?

A grid-based spatial system discretises a continuous 2D or 3D world into a regular lattice of cells (tiles). Each cell stores metadata — terrain type, walkability, cost, ownership — and spatial queries reduce to simple arithmetic on grid indices rather than expensive geometric intersection tests.

Grids are one of the oldest spatial data structures in computing, found everywhere from early tile-based games to modern robotics occupancy maps. They trade geometric fidelity for constant-time queries and trivial implementation.

## 2. When To Use This Pattern

**Good for:**
- Environments with regular, repeating geometry (race tracks, buildings, terrain)
- Queries that must be O(1) or amortised-constant (collision checks at high frequency)
- Systems where perfect geometric accuracy is unnecessary (agent-level perception, not CAD)
- Deterministic simulations where spatial results must be reproducible

**Not good for:**
- Highly irregular or curved geometry requiring sub-pixel accuracy
- Very large worlds where memory cost of a dense grid is prohibitive
- Scenarios needing continuous collision normals or contact manifolds

## 3. Core Concept

### Coordinate Transforms

Every grid system rests on two transforms:

**World → Grid (discretisation):**

```
grid_x = floor((world_x - origin_x) / tile_size)
grid_y = floor((world_y - origin_y) / tile_size)
```

**Grid → World (reconstruction, returns tile centre):**

```
world_x = origin_x + (grid_x + 0.5) * tile_size
world_y = origin_y + (grid_y + 0.5) * tile_size
```

**Worked example:** Given `origin = (0, 0)`, `tile_size = 100`, a world point `(250, 370)` maps to grid cell `(2, 3)`. The centre of cell `(2, 3)` reconstructs to `(250, 350)`.

### O(1) Point-in-Area Queries

To test whether a world point lies on a particular terrain type:
1. Convert to grid coordinates — two divisions, two floors.
2. Index into the tile array — one array lookup.
3. Read the tile metadata — done.

No iteration over geometry. No bounding-volume hierarchies. The cost is constant regardless of map size.

### Tile Representation

Tiles are typically represented as an enum (or integer code). Common variants: `Empty`, `Straight`, `Corner`, `Intersection`, `Wall`. Each variant carries metadata about traversability, orientation, or shape.

### Boundary Testing

**Straight tiles:** Axis-aligned inset checks. A point is "on the road" if it falls within the tile bounds minus a wall-thickness margin on each side.

**Curved tiles:** Arc-distance checks. Compute the distance from the query point to the arc centre (a tile corner). The point is on the road if that distance falls between `inner_radius` and `outer_radius`, where these are derived from `tile_size` minus wall margins.

## 4. Key Design Decisions

| Decision | Option A | Option B |
|---|---|---|
| Grid resolution | Large tiles (fast, low memory) | Small tiles (accurate, more memory) |
| Storage | Dense 2D array (simple, O(1)) | Sparse hashmap (memory-efficient for large empty worlds) |
| Tile boundaries | Hard inset (fast, approximate) | Smooth curves (accurate, more maths) |
| Coordinate origin | Bottom-left (maths convention) | Top-left (screen convention) |

**Key trade-off:** Grid resolution vs accuracy. A coarser grid is faster and uses less memory, but introduces quantisation error at boundaries. For most agent-based simulations, a resolution matching the agent's perception granularity is sufficient.

## 5. Simplified Example Implementation

```python
from enum import Enum
from math import floor, sqrt

class Tile(Enum):
    EMPTY = 0
    STRAIGHT_H = 1
    STRAIGHT_V = 2
    CORNER_NE = 3

class Grid:
    def __init__(self, width, height, tile_size, origin=(0.0, 0.0)):
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.origin = origin
        self.cells = [[Tile.EMPTY] * width for _ in range(height)]

    def world_to_grid(self, wx, wy):
        gx = int(floor((wx - self.origin[0]) / self.tile_size))
        gy = int(floor((wy - self.origin[1]) / self.tile_size))
        return gx, gy

    def grid_to_world(self, gx, gy):
        wx = self.origin[0] + (gx + 0.5) * self.tile_size
        wy = self.origin[1] + (gy + 0.5) * self.tile_size
        return wx, wy

    def is_road_at(self, wx, wy, margin=10.0):
        gx, gy = self.world_to_grid(wx, wy)
        if not (0 <= gx < self.width and 0 <= gy < self.height):
            return False
        tile = self.cells[gy][gx]
        if tile == Tile.EMPTY:
            return False
        # Local position within tile
        lx = wx - (self.origin[0] + gx * self.tile_size)
        ly = wy - (self.origin[1] + gy * self.tile_size)
        if tile in (Tile.STRAIGHT_H, Tile.STRAIGHT_V):
            return margin < lx < self.tile_size - margin
        if tile == Tile.CORNER_NE:
            dist = sqrt(lx**2 + ly**2)
            return margin < dist < self.tile_size - margin
        return False
```

## 6. How NeuroDrive Implements This

NeuroDrive uses a `TrackGrid` — a 14×9 grid with `tile_size = 100.0` and origin `(-100.0, -100.0)`.

**Tile representation:** A `TilePart` enum with variants `Empty`, `StraightH`, `StraightV`, and four corner orientations (`CornerNE`, `CornerNW`, `CornerSE`, `CornerSW`). The track layout is hardcoded as a Monaco-inspired circuit.

**`is_road_at()` method:** Converts a world position to grid indices, retrieves the tile type, then:
- For straight tiles: checks whether the local offset falls within the tile bounds minus a wall-thickness inset (approximately 8 units on each side).
- For corner tiles: computes the distance from the local offset to the arc centre (the appropriate corner of the tile). The point is on road if `distance > margin` and `distance < tile_size - margin`.

**Why grid over continuous geometry?** The track is checked at every raycast step (11 rays × ~125 steps each = ~1,375 lookups per tick). O(1) grid queries make this feasible without spatial indexing overhead. The grid also ensures bitwise-deterministic results — no floating-point polygon-intersection edge cases.

## 7. Variations

- **Hierarchical grids (quadtrees):** Subdivide only occupied cells; reduces memory for sparse worlds.
- **Hex grids:** Six neighbours instead of four/eight; uniform distance to all neighbours; common in strategy games.
- **Weighted grids:** Each cell carries a traversal cost; used with A* or Dijkstra pathfinding.
- **Occupancy grids:** Probabilistic cell values (0.0–1.0) for robotics SLAM.

## 8. Common Pitfalls

- **Off-by-one at boundaries:** Forgetting that `floor()` can produce negative indices for points below the origin. Always bounds-check before indexing.
- **Coordinate convention mismatch:** Mixing y-up (maths) with y-down (screen) silently flips the map. Establish one convention and enforce it.
- **Quantisation aliasing:** A moving entity can "teleport" between tiles if its speed exceeds `tile_size` per tick. Ensure movement is sub-tile per step or use swept queries.
- **Assuming tiles are square:** Non-square tiles break symmetry in distance calculations. Keep tiles square unless you have a compelling reason.

## 9. Projects That Use This Pattern

- **Minecraft:** 16×256×16 chunk-based voxel grid. Demonstrates hierarchical chunked grids at massive scale.
- **ROS OccupancyGrid:** The standard map representation in robotics. Laser scans produce probabilistic occupancy grids for navigation.
- **Factorio:** Tile-based factory layouts with pathfinding, logistics, and pollution simulation all operating on the same grid.

## 10. Glossary

| Term | Definition |
|---|---|
| **Tile** | One cell in the grid, carrying a type and metadata |
| **Tile size** | The side length of one square cell in world units |
| **Origin** | The world-space position of the grid's (0,0) corner |
| **Inset / margin** | Distance from tile edge to the road boundary (wall thickness) |
| **Arc centre** | The corner point used as the centre for curved-tile distance checks |
| **Quantisation** | The loss of precision when mapping continuous positions to discrete cells |

## 11. Recommended Materials

- **"Red Blob Games: Grids and Graphs"** (redblobgames.com) — The definitive interactive tutorial on grid coordinate systems, hex grids, and pathfinding. Start here.
- **"Game Programming Patterns — Spatial Partition"** by Robert Nystrom — Covers grids alongside other spatial structures with clear trade-off analysis.
- **"Probabilistic Robotics"** by Thrun, Burgard & Fox, Chapter 9 — Occupancy grid mapping: the robotics perspective on grid-based spatial systems.
