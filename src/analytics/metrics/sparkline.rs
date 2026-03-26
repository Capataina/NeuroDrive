/// ASCII visual helpers for inline report charts.
///
/// These functions produce compact Unicode representations of numeric series
/// suitable for embedding directly in markdown reports. They are intentionally
/// simple — no external dependencies, no colour, pure text.

const SPARK_CHARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Renders a sparkline from values using Unicode block characters ▁▂▃▄▅▆▇█.
///
/// If `values.len() > width`, the series is downsampled by averaging bins of
/// roughly equal size. An empty input produces an empty string.
pub fn sparkline(values: &[f32], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    let binned = downsample(values, width);
    let min = binned.iter().copied().fold(f32::INFINITY, f32::min);
    let max = binned.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(f32::EPSILON);

    binned
        .iter()
        .map(|&v| {
            let idx = ((v - min) / range * 7.0).round().clamp(0.0, 7.0) as usize;
            SPARK_CHARS[idx]
        })
        .collect()
}

/// Renders a horizontal bar chart segment: "████████░░░░ 234".
///
/// The filled portion uses `█` and the empty portion uses `░`. The raw numeric
/// value is appended after a space. If `max` is zero or negative, the bar is
/// fully empty.
pub fn ascii_bar(value: f32, max: f32, width: usize) -> String {
    if width == 0 {
        return format!("{value:.0}");
    }

    let ratio = if max > 0.0 {
        (value / max).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let filled = (ratio * width as f32).round() as usize;
    let empty = width.saturating_sub(filled);

    format!(
        "{}{} {value:.0}",
        "█".repeat(filled),
        "░".repeat(empty),
    )
}

/// Renders a single-row heatmap — one block character per value, no
/// downsampling. Uses the same ▁–█ scale as `sparkline`.
pub fn heatmap_row(values: &[f32]) -> String {
    if values.is_empty() {
        return String::new();
    }

    let min = values.iter().copied().fold(f32::INFINITY, f32::min);
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(f32::EPSILON);

    values
        .iter()
        .map(|&v| {
            let idx = ((v - min) / range * 7.0).round().clamp(0.0, 7.0) as usize;
            SPARK_CHARS[idx]
        })
        .collect()
}

/// Downsamples `values` into at most `width` bins by averaging.
fn downsample(values: &[f32], width: usize) -> Vec<f32> {
    if values.len() <= width {
        return values.to_vec();
    }

    let mut bins = Vec::with_capacity(width);
    for i in 0..width {
        let start = values.len() * i / width;
        let end = values.len() * (i + 1) / width;
        let slice = &values[start..end];
        if slice.is_empty() {
            bins.push(0.0);
        } else {
            bins.push(slice.iter().sum::<f32>() / slice.len() as f32);
        }
    }
    bins
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparkline_empty() {
        assert_eq!(sparkline(&[], 10), "");
    }

    #[test]
    fn sparkline_single_value() {
        let s = sparkline(&[5.0], 10);
        assert_eq!(s.chars().count(), 1);
    }

    #[test]
    fn sparkline_ascending() {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let s = sparkline(&values, 8);
        assert_eq!(s, "▁▂▃▄▅▆▇█");
    }

    #[test]
    fn sparkline_downsamples() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let s = sparkline(&values, 10);
        assert_eq!(s.chars().count(), 10);
    }

    #[test]
    fn ascii_bar_full() {
        let bar = ascii_bar(100.0, 100.0, 10);
        assert!(bar.starts_with("██████████"));
        assert!(bar.contains("100"));
    }

    #[test]
    fn ascii_bar_half() {
        let bar = ascii_bar(50.0, 100.0, 10);
        assert!(bar.contains('█'));
        assert!(bar.contains('░'));
    }

    #[test]
    fn ascii_bar_zero_max() {
        let bar = ascii_bar(5.0, 0.0, 10);
        assert!(bar.starts_with("░░░░░░░░░░"));
    }

    #[test]
    fn heatmap_row_basic() {
        let values = vec![0.0, 0.5, 1.0];
        let h = heatmap_row(&values);
        assert_eq!(h.chars().count(), 3);
    }

    #[test]
    fn heatmap_row_empty() {
        assert_eq!(heatmap_row(&[]), "");
    }
}
