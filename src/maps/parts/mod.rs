/// Individual tile types that compose a grid-based race track.
///
/// Each tile occupies one square cell in the grid and defines which edges are
/// open (connected to an adjacent road tile). All other edges are walls.
///
/// **Corner naming** describes the quadrant the *outer wall* curves around.
/// For example, `CornerNW` has its outer arc facing north-west, so its two
/// open edges face east and south.
///
/// **T-junction naming** describes the single closed edge (the stem of the T).
/// For example, `TJunctionN` is open on east, south, west — the stem points
/// north and closes on that edge.
///
/// **Connectivity reference:**
/// ```text
/// Tile          | Open edges (N, S, E, W)
/// --------------|------------------------
/// StraightH     | _, _, E, W
/// StraightV     | N, S, _, _
/// CornerNW      | _, S, E, _   (outer arc curves NW)
/// CornerNE      | _, S, _, W   (outer arc curves NE)
/// CornerSW      | N, _, E, _   (outer arc curves SW)
/// CornerSE      | N, _, _, W   (outer arc curves SE)
/// TJunctionN    | _, S, E, W   (stem closes N)
/// TJunctionS    | N, _, E, W   (stem closes S)
/// TJunctionE    | N, S, _, W   (stem closes E)
/// TJunctionW    | N, S, E, _   (stem closes W)
/// Crossroads    | N, S, E, W   (fully open)
/// SpawnPoint    | _, _, E, W   (same as StraightH, marks spawn cell)
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TilePart {
    /// No road surface. The car is off-track if it occupies this cell.
    Empty,

    /// Horizontal straight. Open west and east.
    StraightH,

    /// Vertical straight. Open north and south.
    StraightV,

    /// Corner whose outer arc curves around the north-west quadrant.
    /// Open east and south.
    CornerNW,

    /// Corner whose outer arc curves around the north-east quadrant.
    /// Open west and south.
    CornerNE,

    /// Corner whose outer arc curves around the south-west quadrant.
    /// Open north and east.
    CornerSW,

    /// Corner whose outer arc curves around the south-east quadrant.
    /// Open north and west.
    CornerSE,

    /// T-junction. Open south, east, west. The closed (stem) edge faces north.
    /// Enables tracks that branch or loop back; needed for figure-9 layouts.
    TJunctionN,

    /// T-junction. Open north, east, west. Closed edge faces south.
    TJunctionS,

    /// T-junction. Open north, south, west. Closed edge faces east.
    TJunctionE,

    /// T-junction. Open north, south, east. Closed edge faces west.
    TJunctionW,

    /// Four-way intersection. All four edges open. Needed for figure-8 layouts
    /// where two loops cross.
    Crossroads,

    /// Functionally identical to `StraightH` (open west and east).
    ///
    /// Marks the single cell where the car spawns. The car is placed at the
    /// exact centre of this tile facing east, with the finish line rendered
    /// at the tile boundary one position to the west.
    SpawnPoint,
}

impl TilePart {
    /// Returns `true` if this tile is part of the driveable road surface.
    ///
    /// Only `Empty` returns `false`.
    pub fn is_road(self) -> bool {
        !matches!(self, TilePart::Empty)
    }

    /// Returns which edges are open (connected to an adjacent road tile).
    ///
    /// Tuple order: `(north, south, east, west)`.
    /// An open edge means the road continues in that direction; a closed edge
    /// is a wall boundary.
    pub fn open_edges(self) -> (bool, bool, bool, bool) {
        // (north, south, east, west)
        match self {
            TilePart::Empty       => (false, false, false, false),
            TilePart::StraightH   => (false, false, true,  true ),
            TilePart::SpawnPoint  => (false, false, true,  true ),
            TilePart::StraightV   => (true,  true,  false, false),
            TilePart::CornerNW    => (false, true,  true,  false),
            TilePart::CornerNE    => (false, true,  false, true ),
            TilePart::CornerSW    => (true,  false, true,  false),
            TilePart::CornerSE    => (true,  false, false, true ),
            TilePart::TJunctionN  => (false, true,  true,  true ),
            TilePart::TJunctionS  => (true,  false, true,  true ),
            TilePart::TJunctionE  => (true,  true,  false, true ),
            TilePart::TJunctionW  => (true,  true,  true,  false),
            TilePart::Crossroads  => (true,  true,  true,  true ),
        }
    }

    /// Returns `true` if this tile type uses a curved arc wall when rendered.
    ///
    /// Corner tiles render a quarter-circle outer arc instead of two straight
    /// wall bars. All other road tiles use straight wall sprites.
    pub fn is_corner(self) -> bool {
        matches!(
            self,
            TilePart::CornerNW
                | TilePart::CornerNE
                | TilePart::CornerSW
                | TilePart::CornerSE
        )
    }
}
