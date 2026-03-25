# Notation Guide

## Common Symbols In This Archive

| Symbol | Meaning |
|---|---|
| `s_t` | state or practical observation at time `t` |
| `a_t` | action at time `t` |
| `r_t` | reward at time `t` |
| `G_t` | discounted return from time `t` onward |
| `V(s_t)` | value estimate for state `s_t` |
| `A_t` | advantage estimate at time `t` |
| `gamma` | discount factor |
| `lambda` | GAE or trace-style decay parameter depending on context |
| `delta_t` | temporal-difference error |

## Practical Reading Note

In this archive, `state` may sometimes mean the full formal RL notion, while the actual repository often works with an `ObservationVector`. Where the distinction matters, the project files call it out explicitly.
