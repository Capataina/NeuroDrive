# Idea — Visual Overhaul

## Vision

NeuroDrive runs on Bevy — a proper game engine — but currently looks like a debug prototype. The simulation is functional and informative, but it doesn't leverage what the engine can do. A visual overhaul would make the simulation genuinely pleasant to watch, which matters because the entire project philosophy is "watchable learning".

This isn't about making a racing game. It's about making the learning process visually compelling and the track environment clear and readable.

## Why This Matters

- "Watchable real-time learning" is a core design principle — the visuals should support that, not fight it
- Better track rendering makes it easier to visually assess car behaviour (are they taking racing lines? cutting corners? drifting?)
- Visual polish increases motivation to work on the project and share results
- Bevy 0.18 has substantial rendering capabilities that are currently unused

## Potential Areas (brainstorm — not committed)

### Track
- Textured road surface instead of flat-coloured tiles
- Kerb markings and rumble strips at corners
- Track boundary lines (white/red striping)
- Grass/gravel off-track areas with distinct visual treatment
- Start/finish line marking
- Sector markers visible on track

### Cars
- Proper car sprites or simple geometric car shapes with direction indicators
- Exhaust/trail effects that show the path taken
- Speed-dependent visual effects (motion blur, stretch)
- Colour-coded glow or outline for best/worst car (building on existing ranking)
- Ghost trail showing the best lap trajectory

### Camera
- Smooth camera follow options (follow best car, follow selected car, overview)
- Zoom controls
- Picture-in-picture for a second view

### Environment
- Background grid or subtle reference pattern
- Ambient lighting / colour grading
- Minimap showing full track with car positions

### HUD / Overlays
- More polished HUD design (the current blue-accent panel is functional but basic)
- Animated transitions when data updates
- Sensor rays with gradient colouring (green = far from wall, red = close)
- Progress bar overlay on the track itself
- Real-time mini-charts in the HUD (reward curve, progress curve)

### Learning Visualisation
- Network activation heatmap overlay (what is the actor "seeing"?)
- Action distribution visualisation (show the Gaussian as a live widget)
- Value function landscape overlay on the track (colour the road by predicted value)
- Reward particles (small visual pops where reward is earned)

## Technical Considerations

- Bevy's 2D rendering pipeline supports sprites, shaders, and post-processing
- Custom materials / shaders could be used for track effects
- Performance budget matters — visual improvements must not compromise the 60 FPS target
- Should be implementable incrementally (each visual improvement is independent)

## Open Questions

- What's the minimum visual improvement that makes the biggest difference to watchability?
- Should we prioritise track rendering or car rendering first?
- How much of this can be done with Bevy's built-in 2D features vs. custom shaders?
- Should the visual overhaul wait until after performance optimisation, or can some lightweight improvements happen earlier?

## Status

Idea stage. Low priority relative to PPO optimisation and performance work, but worth revisiting once the core learning problem is solved and the simulation runs smoothly.
