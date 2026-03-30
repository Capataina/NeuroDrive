use crate::analytics::metrics::sparkline::{ascii_bar, sparkline};
use crate::profiling::config::ProfilingConfig;
use crate::profiling::timers::FrameRecord;

use std::collections::HashMap;
use std::fmt::Write;
use std::fs;
use std::path::Path;

const SPARKLINE_WIDTH: usize = 40;
const BAR_WIDTH: usize = 20;

/// Frame budget at 60 Hz in microseconds.
const BUDGET_US: u64 = 16_667;

/// Stutter threshold: 2x the frame budget.
const STUTTER_US: u64 = BUDGET_US * 2;

// ── System → SimSet mapping ──────────────────────────────────────────────
//
// Hardcoded because the profiling layer does not have access to Bevy's
// runtime schedule metadata. Must be kept in sync with the instrument!()
// registrations in mod.rs.

const INPUT_SYSTEMS: &[(&str, &str)] = &[
    ("PPO Action Selection", "Runs the neural network forward pass for all cars to choose steering and throttle"),
    ("Keyboard Input", "Reads keyboard state for manual car control"),
    ("Action Smoothing", "Applies low-pass filter between desired and applied actions"),
];

const PHYSICS_SYSTEMS: &[(&str, &str)] = &[
    ("Car Physics", "Applies steering rotation and throttle force to update car transform and velocity"),
    ("Action Stats Capture", "Records per-car applied steering and throttle values for analytics"),
];

const COLLISION_SYSTEMS: &[(&str, &str)] = &[
    ("Collision Detection", "Checks all car corners against the road occupancy grid for off-road detection"),
];

const MEASUREMENT_SYSTEMS: &[(&str, &str)] = &[
    ("Track Progress", "Projects each car onto the centreline to compute cumulative forward distance"),
    ("Episode Loop (Reward + Reset)", "Computes velocity-projection and centreline rewards, checks crash/timeout, resets episodes"),
    ("Sensor Raycasting", "Casts 11 rays per car against the track boundary grid and computes kinematics"),
    ("Observation Vector", "Normalises sensor readings and lookahead data into the 43-dim observation vector"),
    ("Trace Capture", "Records per-tick trajectory data (position, velocity, drift, policy confidence) for analytics"),
    ("Trace Snapshot", "Snapshots the accumulated tick trace when an episode completes"),
    ("Action Stats Snapshot", "Snapshots the accumulated action statistics when an episode completes"),
    ("PPO Reward Collection", "Appends reward and done flag to the rollout buffer for all cars"),
    ("PPO Epoch (Training)", "Processes a 128-sample chunk of the prepared PPO update (gradient step)"),
    ("HUD Stats", "Updates the live debug HUD with current per-car metrics"),
    ("HUD Episode Capture", "Captures episode-end metrics for the HUD quarter-summary tables"),
];

/// All timed system groups in pipeline order.
const ALL_GROUPS: &[(&str, &[(&str, &str)])] = &[
    ("Input", INPUT_SYSTEMS),
    ("Physics", PHYSICS_SYSTEMS),
    ("Collision", COLLISION_SYSTEMS),
    ("Measurement", MEASUREMENT_SYSTEMS),
];

/// Exports a rich, explanatory Markdown performance report from captured
/// frame timing data.
///
/// The report is designed to be readable by someone unfamiliar with the
/// codebase — every section includes context about what the numbers mean
/// and why they matter.
pub fn export_performance_markdown(
    frames: &[FrameRecord],
    _config: &ProfilingConfig,
    context_header: &str,
    filepath: &str,
) {
    if frames.is_empty() {
        return;
    }

    let path = Path::new(filepath);
    if let Some(parent) = path.parent() {
        let _ = fs::create_dir_all(parent);
    }

    let mut md = String::with_capacity(16384);

    // ── Pre-compute aggregates ───────────────────────────────────────────
    let n = frames.len();
    let nf = n as f64;

    let mean_frame_us = frames.iter().map(|f| f.total_us as f64).sum::<f64>() / nf;
    let mean_frame_ms = mean_frame_us / 1000.0;
    let budget_ms = BUDGET_US as f64 / 1000.0;
    let budget_util_pct = mean_frame_ms / budget_ms * 100.0;

    let frames_over = frames.iter().filter(|f| f.total_us > BUDGET_US).count();
    let frames_over_pct = frames_over as f64 / nf * 100.0;
    let stutter_count = frames.iter().filter(|f| f.total_us > STUTTER_US).count();

    let mut sorted_totals: Vec<u64> = frames.iter().map(|f| f.total_us).collect();
    sorted_totals.sort_unstable();
    let min_us = sorted_totals[0];
    let max_us = sorted_totals[n - 1];
    let p50_us = percentile(&sorted_totals, 50);
    let p95_us = percentile(&sorted_totals, 95);
    let p99_us = percentile(&sorted_totals, 99);

    // Per-set means/maxes
    let input_mean = mean_of(frames, |f| f.input_us);
    let physics_mean = mean_of(frames, |f| f.physics_us);
    let collision_mean = mean_of(frames, |f| f.collision_us);
    let measurement_mean = mean_of(frames, |f| f.measurement_us);

    let input_max = frames.iter().map(|f| f.input_us).max().unwrap_or(0);
    let physics_max = frames.iter().map(|f| f.physics_us).max().unwrap_or(0);
    let collision_max = frames.iter().map(|f| f.collision_us).max().unwrap_or(0);
    let measurement_max = frames.iter().map(|f| f.measurement_us).max().unwrap_or(0);

    // Per-system aggregates across all frames
    let sys_agg = compute_system_aggregates(frames);

    // ── Title ────────────────────────────────────────────────────────────
    md.push_str("# NeuroDrive — Performance Report\n\n");
    md.push_str(context_header);
    md.push_str("\n\n");

    // ── How to Read This Report ──────────────────────────────────────────
    md.push_str("## How to Read This Report\n\n");
    md.push_str(
        "This report profiles the **fixed-update simulation loop** which runs at 60 Hz \
         (16.67ms budget per tick).\n\
         The simulation pipeline processes four stages in order each tick:\n\n\
         1. **Input** — Controller decisions: the PPO brain selects actions for all cars, \
         keyboard input is captured, actions are smoothed.\n\
         2. **Physics** — Car dynamics: applies steering and throttle to update car positions \
         and velocities.\n\
         3. **Collision** — Boundary detection: checks whether any car has left the road surface.\n\
         4. **Measurement** — Everything else: track progress computation, reward calculation, \
         episode resets, sensor raycasting (11 rays × N cars), observation vector construction, \
         PPO training (gradient updates), analytics capture, and HUD updates.\n\n\
         A \"stutter\" is any tick that takes more than 2× the budget (>33.3ms). \
         Stutters cause visible jank.\n\n",
    );

    // ── Overall Verdict ──────────────────────────────────────────────────
    md.push_str("## Overall Verdict\n\n");
    md.push_str(&generate_verdict(
        budget_util_pct,
        frames_over_pct,
        stutter_count,
        mean_frame_ms,
        &sys_agg,
        mean_frame_us,
    ));
    md.push_str("\n\n");

    // ── Frame Budget ─────────────────────────────────────────────────────
    md.push_str("## Frame Budget\n\n");
    md.push_str("| Metric | Value |\n");
    md.push_str("|--------|------:|\n");
    let _ = writeln!(md, "| Total ticks captured | {n} |");
    let _ = writeln!(md, "| Mean frame time | {mean_frame_ms:.3} ms |");
    let _ = writeln!(md, "| Frame budget (60 Hz) | {budget_ms:.2} ms |");
    let _ = writeln!(md, "| Budget utilisation | {budget_util_pct:.1}% |");
    let _ = writeln!(
        md,
        "| Frames over budget | {frames_over} ({frames_over_pct:.1}%) |"
    );
    let _ = writeln!(md, "| Stutter count (>2× budget) | {stutter_count} |");
    md.push('\n');

    // ── Frame Time Distribution ──────────────────────────────────────────
    md.push_str("### Frame Time Distribution\n\n");
    let total_us_vals: Vec<f32> = frames.iter().map(|f| f.total_us as f32).collect();
    let _ = writeln!(
        md,
        "**Frame times:** `{}`\n",
        sparkline(&total_us_vals, SPARKLINE_WIDTH)
    );

    md.push_str("| Stat | Value (ms) |\n");
    md.push_str("|------|----------:|\n");
    let _ = writeln!(md, "| Min | {:.3} |", us_to_ms(min_us));
    let _ = writeln!(md, "| P50 | {:.3} |", us_to_ms(p50_us));
    let _ = writeln!(md, "| P95 | {:.3} |", us_to_ms(p95_us));
    let _ = writeln!(md, "| P99 | {:.3} |", us_to_ms(p99_us));
    let _ = writeln!(md, "| Max | {:.3} |", us_to_ms(max_us));
    md.push('\n');

    // Histogram
    let buckets: &[(&str, u64, u64)] = &[
        ("<1ms", 0, 1_000),
        ("1–5ms", 1_000, 5_000),
        ("5–10ms", 5_000, 10_000),
        ("10–16ms", 10_000, 16_000),
        ("16–33ms", 16_000, 33_000),
        (">33ms", 33_000, u64::MAX),
    ];
    let bucket_counts: Vec<(&str, usize)> = buckets
        .iter()
        .map(|(label, lo, hi)| {
            let count = frames
                .iter()
                .filter(|f| f.total_us >= *lo && f.total_us < *hi)
                .count();
            (*label, count)
        })
        .collect();

    md.push_str("```text\n");
    for (label, count) in &bucket_counts {
        let pct = *count as f32 / n as f32 * 100.0;
        let _ = writeln!(
            md,
            "{:<8} {} {:.0}% ({})",
            label,
            ascii_bar(pct, 100.0, BAR_WIDTH),
            pct,
            count,
        );
    }
    md.push_str("```\n\n");

    // Distribution interpretation
    md.push_str("### What This Tells Us\n\n");
    md.push_str(&interpret_distribution(
        min_us, max_us, p50_us, p95_us, p99_us, stutter_count, n,
    ));
    md.push_str("\n\n");

    // ── Pipeline Breakdown ───────────────────────────────────────────────
    md.push_str("## Pipeline Breakdown\n\n");
    md.push_str(
        "This shows how much time each stage of the simulation pipeline takes on average.\n\n",
    );

    let set_data = [
        ("Input", input_mean, input_max, &frames.iter().map(|f| f.input_us as f32).collect::<Vec<_>>()),
        ("Physics", physics_mean, physics_max, &frames.iter().map(|f| f.physics_us as f32).collect::<Vec<_>>()),
        ("Collision", collision_mean, collision_max, &frames.iter().map(|f| f.collision_us as f32).collect::<Vec<_>>()),
        ("Measurement", measurement_mean, measurement_max, &frames.iter().map(|f| f.measurement_us as f32).collect::<Vec<_>>()),
    ];

    md.push_str("| Stage | Mean (ms) | Max (ms) | % of frame | Sparkline |\n");
    md.push_str("|-------|----------:|---------:|-----------:|----------:|\n");
    for (name, mean, max, vals) in &set_data {
        let pct = if mean_frame_us > 0.0 {
            mean / mean_frame_us * 100.0
        } else {
            0.0
        };
        let _ = writeln!(
            md,
            "| {name} | {:.3} | {:.3} | {pct:.1}% | `{}` |",
            us_to_ms_f64(*mean),
            us_to_ms(*max),
            sparkline(vals, SPARKLINE_WIDTH),
        );
    }
    md.push('\n');

    // Pipeline interpretation
    md.push_str("### What This Tells Us\n\n");
    let dominant_set = set_data
        .iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();
    let dom_pct = if mean_frame_us > 0.0 {
        dominant_set.1 / mean_frame_us * 100.0
    } else {
        0.0
    };
    let _ = writeln!(
        md,
        "**{}** dominates the frame at {:.1}% ({:.3} ms mean). ",
        dominant_set.0,
        dom_pct,
        us_to_ms_f64(dominant_set.1),
    );
    if dominant_set.0 == "Measurement" {
        md.push_str(
            "This is expected — Measurement contains the most systems including sensor raycasting, \
             reward computation, PPO training, and analytics capture. \
             See the per-system detail below to identify which specific systems are expensive.\n",
        );
    } else {
        let _ = writeln!(
            md,
            "This is unusual — {} typically should not dominate. \
             Investigate the per-system detail below.",
            dominant_set.0,
        );
    }
    md.push('\n');

    // ── Per-System Detail ────────────────────────────────────────────────
    md.push_str("## Per-System Detail\n\n");
    md.push_str(
        "These are the individual systems within each pipeline stage, \
         showing exactly what code is expensive.\n\n",
    );

    for (group_name, systems) in ALL_GROUPS {
        let _ = writeln!(md, "### {group_name} Systems\n");

        // Compute the stage total for % of stage
        let stage_total_mean: f64 = systems
            .iter()
            .filter_map(|(name, _)| sys_agg.get(*name))
            .map(|a| a.mean_us)
            .sum();

        md.push_str(
            "| System | What It Does | Mean (ms) | Max (ms) | % of stage |\n",
        );
        md.push_str(
            "|--------|-------------|----------:|---------:|-----------:|\n",
        );

        for (sys_name, description) in *systems {
            if let Some(agg) = sys_agg.get(*sys_name) {
                let pct_stage = if stage_total_mean > 0.0 {
                    agg.mean_us / stage_total_mean * 100.0
                } else {
                    0.0
                };
                let _ = writeln!(
                    md,
                    "| {sys_name} | {description} | {:.3} | {:.3} | {pct_stage:.1}% |",
                    us_to_ms_f64(agg.mean_us),
                    us_to_ms(agg.max_us),
                );
            } else {
                let _ = writeln!(
                    md,
                    "| {sys_name} | {description} | — | — | — |",
                );
            }
        }
        md.push('\n');
    }

    // Per-system interpretation
    md.push_str("### What This Tells Us\n\n");
    md.push_str(&interpret_systems(&sys_agg, mean_frame_us));
    md.push('\n');

    // ── Stutter Analysis ─────────────────────────────────────────────────
    md.push_str("## Stutter Analysis\n\n");

    if stutter_count > 0 {
        let _ = writeln!(
            md,
            "Stutters occur when a tick exceeds {:.1}ms (2× the {:.2}ms budget). \
             **{stutter_count}** stutters were detected in this run.\n",
            us_to_ms(STUTTER_US),
            budget_ms,
        );

        // Worst ticks with full per-system timings
        md.push_str("### Worst Ticks\n\n");
        let mut worst_indices: Vec<usize> = (0..n).collect();
        worst_indices.sort_by(|&a, &b| frames[b].total_us.cmp(&frames[a].total_us));
        worst_indices.truncate(5);

        // Collect all system names that appear in worst ticks
        let mut all_sys_names: Vec<String> = Vec::new();
        for &i in &worst_indices {
            for (name, _) in &frames[i].system_timings {
                if !all_sys_names.contains(name) {
                    all_sys_names.push(name.clone());
                }
            }
        }
        // Sort by pipeline order using the hardcoded groups
        let order: Vec<&str> = ALL_GROUPS
            .iter()
            .flat_map(|(_, systems)| systems.iter().map(|(name, _)| *name))
            .collect();
        all_sys_names.sort_by_key(|name| {
            order
                .iter()
                .position(|&o| o == name.as_str())
                .unwrap_or(usize::MAX)
        });

        // Header
        md.push_str("| Tick | Total (ms) |");
        for name in &all_sys_names {
            // Abbreviate long names for column headers
            let short = abbreviate_system_name(name);
            let _ = write!(md, " {short} |");
        }
        md.push_str(" Buffer | × Budget |\n");

        md.push_str("|-----:|-----------:|");
        for _ in &all_sys_names {
            md.push_str("-------:|");
        }
        md.push_str("-------:|---------:|\n");

        for &i in &worst_indices {
            let f = &frames[i];
            let sys_map: HashMap<&str, u64> = f
                .system_timings
                .iter()
                .map(|(name, us)| (name.as_str(), *us))
                .collect();
            let times_over = f.total_us as f64 / BUDGET_US as f64;
            let _ = write!(md, "| {} | {:.3} |", f.tick, us_to_ms(f.total_us));
            for name in &all_sys_names {
                let us = sys_map.get(name.as_str()).copied().unwrap_or(0);
                let _ = write!(md, " {:.3} |", us_to_ms(us));
            }
            let _ = writeln!(md, " {} | {times_over:.1}× |", f.rollout_buffer_len);
        }
        md.push('\n');

        // Stutter correlation with PPO Epoch
        md.push_str("### Stutter Correlation\n\n");
        let stutter_frames: Vec<&FrameRecord> =
            frames.iter().filter(|f| f.total_us > STUTTER_US).collect();
        let normal_frames: Vec<&FrameRecord> =
            frames.iter().filter(|f| f.total_us <= STUTTER_US).collect();

        let ppo_epoch_name = "PPO Epoch (Training)";
        let avg_ppo_stutter = avg_system_timing(&stutter_frames, ppo_epoch_name);
        let avg_ppo_normal = avg_system_timing(&normal_frames, ppo_epoch_name);

        if avg_ppo_stutter > 0.0 && avg_ppo_normal > 0.0 {
            let ratio = avg_ppo_stutter / avg_ppo_normal;
            if ratio > 2.0 {
                let _ = writeln!(
                    md,
                    "**Strong PPO correlation detected.** The PPO Epoch system averages \
                     {:.3}ms during stutters vs {:.3}ms normally ({:.1}× higher). \
                     Stutters are likely caused by PPO gradient updates landing on these ticks.",
                    us_to_ms_f64(avg_ppo_stutter),
                    us_to_ms_f64(avg_ppo_normal),
                    ratio,
                );
            } else if ratio > 1.3 {
                let _ = writeln!(
                    md,
                    "**Moderate PPO correlation.** The PPO Epoch system averages \
                     {:.3}ms during stutters vs {:.3}ms normally ({:.1}× higher). \
                     PPO updates contribute to but may not fully explain stutters.",
                    us_to_ms_f64(avg_ppo_stutter),
                    us_to_ms_f64(avg_ppo_normal),
                    ratio,
                );
            } else {
                let _ = writeln!(
                    md,
                    "**No strong PPO correlation.** The PPO Epoch system averages \
                     {:.3}ms during stutters vs {:.3}ms normally. \
                     Stutters are likely caused by other systems.",
                    us_to_ms_f64(avg_ppo_stutter),
                    us_to_ms_f64(avg_ppo_normal),
                );
            }
        } else {
            md.push_str(
                "Insufficient PPO Epoch data to determine stutter correlation.\n",
            );
        }

        // Also check buffer levels
        let avg_stutter_buffer = stutter_frames
            .iter()
            .map(|f| f.rollout_buffer_len as f64)
            .sum::<f64>()
            / stutter_frames.len().max(1) as f64;
        let avg_buffer = frames
            .iter()
            .map(|f| f.rollout_buffer_len as f64)
            .sum::<f64>()
            / nf;
        if avg_stutter_buffer > avg_buffer * 1.5 {
            let _ = writeln!(
                md,
                "\nStutter frames have elevated rollout buffer levels ({:.0} vs {:.0} avg), \
                 confirming they coincide with PPO update preparation.",
                avg_stutter_buffer, avg_buffer,
            );
        }
        md.push('\n');

        md.push_str("### What This Tells Us\n\n");
        md.push_str(&interpret_stutters(
            &stutter_frames,
            &sys_agg,
            avg_ppo_stutter,
            avg_ppo_normal,
        ));
        md.push('\n');
    } else {
        md.push_str(
            "No stutters detected — all ticks completed within 2× the frame budget (33.3ms). \
             This is healthy.\n\n",
        );
    }

    // ── Buffer & Memory Pressure ─────────────────────────────────────────
    md.push_str("## Buffer & Memory Pressure\n\n");

    md.push_str(
        "**PPO Rollout Buffer** — holds transitions until the PPO update fires. \
         When it hits the horizon (512), it's frozen and the training loop \
         begins processing it in chunks.\n\n",
    );
    let rollout_vals: Vec<f32> = frames.iter().map(|f| f.rollout_buffer_len as f32).collect();
    let max_rollout = frames.iter().map(|f| f.rollout_buffer_len).max().unwrap_or(0);
    let _ = writeln!(
        md,
        "`{}` max: {}\n",
        sparkline(&rollout_vals, SPARKLINE_WIDTH),
        max_rollout,
    );

    md.push_str(
        "**Analytics Trace Count** — number of completed episode traces held in memory \
         for the exit report.\n\n",
    );
    let trace_vals: Vec<f32> = frames.iter().map(|f| f.trace_count as f32).collect();
    let max_traces = frames.iter().map(|f| f.trace_count).max().unwrap_or(0);
    let _ = writeln!(
        md,
        "`{}` max: {}\n",
        sparkline(&trace_vals, SPARKLINE_WIDTH),
        max_traces,
    );

    // ── Recommendations ──────────────────────────────────────────────────
    md.push_str("## Recommendations\n\n");
    md.push_str(
        "Based on this profiling run, here are the ranked optimisation opportunities:\n\n",
    );
    let recs = generate_recommendations(&sys_agg, mean_frame_us, stutter_count, frames);
    for (i, rec) in recs.iter().enumerate() {
        let _ = writeln!(md, "{}. {rec}", i + 1);
    }

    let _ = fs::write(filepath, md);
}

// ── Data structures ─────────────────────────────────────────────────────

/// Aggregated timing statistics for a single named system.
struct SystemAggregate {
    mean_us: f64,
    max_us: u64,
    /// Total microseconds across all frames.
    #[allow(dead_code)]
    total_us: u64,
    /// Number of frames where this system reported a timing.
    #[allow(dead_code)]
    frame_count: usize,
}

/// Computes per-system aggregate statistics from all frame records.
fn compute_system_aggregates(frames: &[FrameRecord]) -> HashMap<String, SystemAggregate> {
    let mut totals: HashMap<String, (u64, u64, usize)> = HashMap::new(); // (sum, max, count)

    for frame in frames {
        for (name, us) in &frame.system_timings {
            let entry = totals.entry(name.clone()).or_insert((0, 0, 0));
            entry.0 += us;
            entry.1 = entry.1.max(*us);
            entry.2 += 1;
        }
    }

    totals
        .into_iter()
        .map(|(name, (sum, max, count))| {
            let mean = if count > 0 {
                sum as f64 / count as f64
            } else {
                0.0
            };
            (
                name,
                SystemAggregate {
                    mean_us: mean,
                    max_us: max,
                    total_us: sum,
                    frame_count: count,
                },
            )
        })
        .collect()
}

/// Computes the average timing for a named system across a set of frames.
fn avg_system_timing(frames: &[&FrameRecord], system_name: &str) -> f64 {
    let mut sum = 0u64;
    let mut count = 0usize;
    for f in frames {
        for (name, us) in &f.system_timings {
            if name == system_name {
                sum += us;
                count += 1;
            }
        }
    }
    if count > 0 {
        sum as f64 / count as f64
    } else {
        0.0
    }
}

// ── Interpretation generators ───────────────────────────────────────────

fn generate_verdict(
    budget_util_pct: f64,
    frames_over_pct: f64,
    stutter_count: usize,
    mean_frame_ms: f64,
    sys_agg: &HashMap<String, SystemAggregate>,
    mean_frame_us: f64,
) -> String {
    let health = if budget_util_pct < 30.0 && frames_over_pct < 1.0 {
        "The simulation is **well within budget**"
    } else if budget_util_pct < 70.0 && frames_over_pct < 5.0 {
        "The simulation is **comfortable**"
    } else if budget_util_pct < 100.0 && frames_over_pct < 15.0 {
        "The simulation is **marginal**"
    } else {
        "The simulation is **over budget**"
    };

    // Find the most expensive system
    let top_system = sys_agg
        .iter()
        .max_by(|a, b| a.1.mean_us.partial_cmp(&b.1.mean_us).unwrap());

    let mut verdict = format!(
        "{health} at {mean_frame_ms:.3}ms mean frame time ({budget_util_pct:.1}% of the \
         16.67ms budget).",
    );

    if let Some((name, agg)) = top_system {
        let pct = if mean_frame_us > 0.0 {
            agg.mean_us / mean_frame_us * 100.0
        } else {
            0.0
        };
        if pct > 30.0 {
            let _ = write!(
                verdict,
                " The most expensive system is **{name}** at {:.3}ms ({pct:.1}% of frame time).",
                us_to_ms_f64(agg.mean_us),
            );
        }
    }

    if stutter_count > 0 {
        let _ = write!(
            verdict,
            " There were **{stutter_count} stutters** (ticks exceeding 33.3ms) \
             which would cause visible jank in real-time playback.",
        );
    }

    verdict
}

fn interpret_distribution(
    min_us: u64,
    max_us: u64,
    p50_us: u64,
    p95_us: u64,
    _p99_us: u64,
    stutter_count: usize,
    n: usize,
) -> String {
    let mut text = String::new();

    let spread_ratio = if min_us > 0 {
        max_us as f64 / min_us as f64
    } else {
        0.0
    };
    let p95_p50_ratio = if p50_us > 0 {
        p95_us as f64 / p50_us as f64
    } else {
        0.0
    };

    if spread_ratio > 10.0 {
        let _ = write!(
            text,
            "The distribution has a **wide spread** (max is {spread_ratio:.0}× the min), \
             indicating significant variability between ticks. ",
        );
    } else if spread_ratio > 3.0 {
        let _ = write!(
            text,
            "The distribution shows **moderate variability** \
             (max is {spread_ratio:.1}× the min). ",
        );
    } else {
        text.push_str("The distribution is **tight and consistent**. ");
    }

    if p95_p50_ratio > 3.0 {
        let _ = write!(
            text,
            "The P95/P50 ratio of {p95_p50_ratio:.1}× suggests a **bimodal pattern** — \
             most ticks are fast but a subset are significantly slower, \
             likely corresponding to PPO training ticks. ",
        );
    } else if p95_p50_ratio > 1.5 {
        text.push_str(
            "The tail is moderately elevated, suggesting occasional heavier ticks. ",
        );
    }

    if stutter_count > 0 {
        let pct = stutter_count as f64 / n as f64 * 100.0;
        let _ = write!(
            text,
            "Stutters affect {pct:.2}% of ticks — ",
        );
        if pct < 0.5 {
            text.push_str("rare enough to be acceptable for training but noticeable during observation.");
        } else {
            text.push_str("frequent enough to warrant investigation.");
        }
    }

    text
}

fn interpret_systems(sys_agg: &HashMap<String, SystemAggregate>, mean_frame_us: f64) -> String {
    let mut text = String::new();

    // Sort systems by mean time descending
    let mut sorted: Vec<(&String, &SystemAggregate)> = sys_agg.iter().collect();
    sorted.sort_by(|a, b| b.1.mean_us.partial_cmp(&a.1.mean_us).unwrap());

    // Top 3 most expensive
    let top3: Vec<_> = sorted.iter().take(3).collect();
    if !top3.is_empty() {
        text.push_str("The most expensive systems in descending order:\n\n");
        for (name, agg) in &top3 {
            let pct = if mean_frame_us > 0.0 {
                agg.mean_us / mean_frame_us * 100.0
            } else {
                0.0
            };
            let _ = writeln!(
                text,
                "- **{}** — {:.3}ms mean ({:.1}% of frame)",
                name,
                us_to_ms_f64(agg.mean_us),
                pct,
            );
        }

        // Specific insights
        if let Some(sensor) = sys_agg.get("Sensor Raycasting") {
            let pct = sensor.mean_us / mean_frame_us * 100.0;
            if pct > 20.0 {
                let _ = writeln!(
                    text,
                    "\nSensor Raycasting at {pct:.1}% is a significant cost centre. \
                     Each car casts 11 rays against the track grid every tick. \
                     A spatial acceleration structure could reduce this.",
                );
            }
        }
        if let Some(ppo) = sys_agg.get("PPO Epoch (Training)") {
            if ppo.max_us > BUDGET_US {
                let _ = writeln!(
                    text,
                    "\nPPO Epoch (Training) peaks at {:.3}ms — exceeding a full frame budget \
                     on its own. This is expected since the training step involves matrix \
                     multiplications across the full minibatch.",
                    us_to_ms(ppo.max_us),
                );
            }
        }
    }

    text
}

fn interpret_stutters(
    stutter_frames: &[&FrameRecord],
    _sys_agg: &HashMap<String, SystemAggregate>,
    avg_ppo_stutter: f64,
    avg_ppo_normal: f64,
) -> String {
    let mut text = String::new();

    if avg_ppo_stutter > avg_ppo_normal * 2.0 {
        text.push_str(
            "Stutters are **strongly correlated with PPO training**. The PPO Epoch system \
             runs a gradient update on a 128-sample chunk from the rollout buffer, which \
             involves forward and backward passes through the actor-critic network. ",
        );
        text.push_str(
            "Consider: (a) reducing chunk size to spread the work across more ticks, \
             (b) profiling the matrix multiplication hot path for SIMD opportunities, \
             or (c) accepting the stutters since they only affect visual smoothness during training.\n",
        );
    } else {
        // Identify the dominant system during stutters
        let mut sys_totals: HashMap<&str, u64> = HashMap::new();
        for f in stutter_frames {
            for (name, us) in &f.system_timings {
                *sys_totals.entry(name.as_str()).or_insert(0) += us;
            }
        }
        if let Some((name, _)) = sys_totals.iter().max_by_key(|(_, v)| **v) {
            let _ = write!(
                text,
                "The dominant system during stutters is **{name}**. \
                 Focus optimisation efforts there.\n",
            );
        }
    }

    text
}

fn generate_recommendations(
    sys_agg: &HashMap<String, SystemAggregate>,
    mean_frame_us: f64,
    stutter_count: usize,
    frames: &[FrameRecord],
) -> Vec<String> {
    let mut recs: Vec<(f64, String)> = Vec::new(); // (priority_score, text)

    // Check for a single system dominating >50% of frame time
    for (name, agg) in sys_agg {
        let pct = if mean_frame_us > 0.0 {
            agg.mean_us / mean_frame_us * 100.0
        } else {
            0.0
        };
        if pct > 50.0 {
            recs.push((
                100.0,
                format!(
                    "**Optimise {name}** — This single system consumes {pct:.1}% of every frame \
                     ({:.3}ms mean). Any improvement here directly reduces frame time.",
                    us_to_ms_f64(agg.mean_us),
                ),
            ));
        }
    }

    // PPO Epoch stutter correlation
    if stutter_count > 0 {
        let stutter_frames: Vec<&FrameRecord> =
            frames.iter().filter(|f| f.total_us > STUTTER_US).collect();
        let normal_frames: Vec<&FrameRecord> =
            frames.iter().filter(|f| f.total_us <= STUTTER_US).collect();
        let avg_ppo_stutter = avg_system_timing(&stutter_frames, "PPO Epoch (Training)");
        let avg_ppo_normal = avg_system_timing(&normal_frames, "PPO Epoch (Training)");
        if avg_ppo_stutter > avg_ppo_normal * 2.0 {
            recs.push((
                90.0,
                format!(
                    "**Reduce PPO training stutter** — PPO Epoch averages {:.3}ms during \
                     stutter ticks vs {:.3}ms normally. Consider reducing chunk size below 128, \
                     amortising further, or using batched matrix operations with SIMD intrinsics.",
                    us_to_ms_f64(avg_ppo_stutter),
                    us_to_ms_f64(avg_ppo_normal),
                ),
            ));
        }
    }

    // Sensor raycasting >20%
    if let Some(sensor) = sys_agg.get("Sensor Raycasting") {
        let pct = if mean_frame_us > 0.0 {
            sensor.mean_us / mean_frame_us * 100.0
        } else {
            0.0
        };
        if pct > 20.0 {
            recs.push((
                80.0,
                format!(
                    "**Optimise Sensor Raycasting** — At {pct:.1}% of frame time ({:.3}ms), \
                     raycasting is expensive. The current implementation checks each ray step \
                     against a flat grid. A spatial hash or bounding-volume hierarchy could \
                     reduce ray traversal cost significantly.",
                    us_to_ms_f64(sensor.mean_us),
                ),
            ));
        }
    }

    // Analytics capture >10%
    let analytics_systems = ["Trace Capture", "Trace Snapshot", "Action Stats Capture", "Action Stats Snapshot"];
    let analytics_total: f64 = analytics_systems
        .iter()
        .filter_map(|name| sys_agg.get(*name))
        .map(|a| a.mean_us)
        .sum();
    let analytics_pct = if mean_frame_us > 0.0 {
        analytics_total / mean_frame_us * 100.0
    } else {
        0.0
    };
    if analytics_pct > 10.0 {
        recs.push((
            60.0,
            format!(
                "**Reduce analytics overhead** — Analytics capture systems collectively \
                 consume {analytics_pct:.1}% of frame time ({:.3}ms). Consider reducing \
                 trace detail or sampling frequency.",
                us_to_ms_f64(analytics_total),
            ),
        ));
    }

    // Occasional spikes but healthy average
    let frames_over_pct =
        frames.iter().filter(|f| f.total_us > BUDGET_US).count() as f64 / frames.len() as f64
            * 100.0;
    let budget_util_pct = mean_frame_us / BUDGET_US as f64 * 100.0;
    if budget_util_pct < 70.0 && frames_over_pct > 1.0 {
        recs.push((
            50.0,
            format!(
                "**Focus on spike causes, not average** — Mean budget utilisation is only \
                 {budget_util_pct:.1}% but {frames_over_pct:.1}% of frames exceed budget. \
                 The per-system worst-tick table above shows what fires during spikes.",
            ),
        ));
    }

    // If well within budget, say so
    if recs.is_empty() {
        recs.push((
            0.0,
            "**No critical optimisations needed.** The simulation is running well within \
             budget. Continue monitoring as car count or observation complexity increases."
                .to_string(),
        ));
    }

    // Sort by priority descending
    recs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    recs.into_iter().map(|(_, text)| text).collect()
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn us_to_ms(us: u64) -> f64 {
    us as f64 / 1000.0
}

fn us_to_ms_f64(us: f64) -> f64 {
    us / 1000.0
}

fn mean_of(frames: &[FrameRecord], f: impl Fn(&FrameRecord) -> u64) -> f64 {
    if frames.is_empty() {
        return 0.0;
    }
    frames.iter().map(|fr| f(fr) as f64).sum::<f64>() / frames.len() as f64
}

fn percentile(sorted: &[u64], pct: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (sorted.len() * pct / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// Abbreviates a system name for use as a compact table column header.
fn abbreviate_system_name(name: &str) -> String {
    match name {
        "PPO Action Selection" => "PPO Act".to_string(),
        "Keyboard Input" => "KBD".to_string(),
        "Action Smoothing" => "Smooth".to_string(),
        "Car Physics" => "Phys".to_string(),
        "Action Stats Capture" => "AStats".to_string(),
        "Collision Detection" => "Coll".to_string(),
        "Track Progress" => "Prog".to_string(),
        "Episode Loop (Reward + Reset)" => "EpLoop".to_string(),
        "Sensor Raycasting" => "Rays".to_string(),
        "Observation Vector" => "Obs".to_string(),
        "Trace Capture" => "Trace".to_string(),
        "Trace Snapshot" => "TrSnap".to_string(),
        "Action Stats Snapshot" => "ASSnap".to_string(),
        "PPO Reward Collection" => "PPO Rwd".to_string(),
        "PPO Epoch (Training)" => "PPO Ep".to_string(),
        "HUD Stats" => "HUD".to_string(),
        "HUD Episode Capture" => "HUDEp".to_string(),
        _ => {
            // Fallback: take first 6 chars
            name.chars().take(6).collect()
        }
    }
}
