use colored::*;
use serde::{Deserialize, Serialize};

use std::{
    fmt::Display,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct TimingReporter {
    timer: Instant,
    durations: Vec<(String, Duration)>,
    print_debug: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingReport {
    pub total_duration: Duration,
    pub step_durations: Vec<(String, Duration)>,
}

impl TimingReporter {
    pub fn start() -> Self {
        let timer = Instant::now();
        let durations = vec![];
        let print_debug = std::env::var("DEBUG_TIMING").unwrap_or_else(|_| "0".to_string()) != "0";

        Self {
            timer,
            durations,
            print_debug,
        }
    }

    pub fn elapsed(&mut self, label: &str) {
        let label = label.to_string();
        let duration = self.timer.elapsed();

        if self.print_debug {
            let rel_duration = duration.saturating_sub(
                *self
                    .durations
                    .last()
                    .map(|(_, x)| x)
                    .unwrap_or(&Duration::ZERO),
            );

            println!("EVENT: {label} in {rel_duration:.2?}");
        }

        self.durations.push((label, duration));
    }

    pub fn finish(&self) -> TimingReport {
        let absolute_duration = self.timer.elapsed();
        let mut relative_durations = vec![];
        let mut prev_duration = Duration::ZERO;
        for (label, duration) in self.durations.iter() {
            let rel_duration = duration.saturating_sub(prev_duration);
            relative_durations.push((label.to_string(), rel_duration));
            prev_duration = *duration;
        }
        TimingReport {
            total_duration: absolute_duration,
            step_durations: relative_durations,
        }
    }
}

impl Display for TimingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Calculate width of label and duration columns
        let (label_widths, duration_widths): (Vec<usize>, Vec<usize>) = self
            .step_durations
            .iter()
            .map(|(label, duration)| (label.len(), format!("{} ms", duration.as_millis()).len()))
            .into_iter()
            .unzip();
        let max_label_width = label_widths.into_iter().max().unwrap_or(0);
        let max_duration_width = duration_widths.into_iter().max().unwrap_or(0);
        let width = max_duration_width;

        let (header_label_raw, _) = self.step_durations[0].clone();
        let header_label = header_label_raw
            .strip_prefix("Table name: ")
            .unwrap_or(&header_label_raw);

        let durations: Vec<Duration> = self
            .step_durations
            .iter()
            .skip(1) // We do not want Table header affecting in our stats.
            .map(|(_, duration)| *duration)
            .collect();

        let num_steps: u32 = self.step_durations.len().try_into().unwrap();
        let average_duration = self.total_duration / (num_steps - 1); // account for header
        let std_dev_duration = std_dev_duration(&durations);
        let mean_duration = {
            let mut durations_mean = durations;
            durations_mean.sort();

            let durations_count = durations_mean.len();

            if durations_count % 2 == 0 {
                (durations_mean[durations_count / 2 - 1] + durations_mean[durations_count / 2]) / 2
            } else {
                durations_mean[durations_count / 2]
            }
        };

        let colored_columns: Vec<(String, ColoredString)> = self
            .step_durations
            .iter()
            .skip(1) // We do not want Table header affecting in our stats.
            .map(|(label, duration)| {
                (
                    label.clone(),
                    colorize(*duration, average_duration, max_duration_width),
                )
            })
            .collect();

        let space = 2;
        // Print header
        writeln!(f, "\n")?; // Add at least one empty line before report
        writeln!(f, "{header_label}")?;
        let ornament = "=".repeat(header_label.len());
        writeln!(f, "{ornament}")?;
        // Print entries
        for (label, duration) in colored_columns {
            let padding =
                String::from_utf8(vec![b' '; max_label_width - label.len() + space]).unwrap();
            writeln!(f, "{label}{padding}{duration}")?;
        }

        writeln!(
            f,
            "{}{}{}",
            String::from_utf8(vec![b'-'; max_label_width]).unwrap(),
            String::from_utf8(vec![b' '; space]).unwrap(),
            // +3 to account for the suffix " ms".
            String::from_utf8(vec![b'-'; width + 3]).unwrap()
        )?;

        let total_label = "Total";
        let total_padding =
            String::from_utf8(vec![b' '; max_label_width - total_label.len() + space]).unwrap();
        writeln!(
            f,
            "{}{}{:width$} ms",
            total_label,
            total_padding,
            self.total_duration.as_millis()
        )?;
        let mean_label = "Mean";
        let mean_padding =
            String::from_utf8(vec![b' '; max_label_width - mean_label.len() + space]).unwrap();
        writeln!(
            f,
            "{}{}{:>width$} ms",
            mean_label,
            mean_padding,
            mean_duration.as_millis()
        )?;
        let average_label = "Average";
        let average_padding =
            String::from_utf8(vec![b' '; max_label_width - average_label.len() + space]).unwrap();
        writeln!(
            f,
            "{}{}{:>width$} ms",
            average_label,
            average_padding,
            average_duration.as_millis()
        )?;
        let std_dev_label = "Std. Dev.";
        let std_dev_padding =
            String::from_utf8(vec![b' '; max_label_width - std_dev_label.len() + space]).unwrap();
        writeln!(
            f,
            "{}{}{:>width$} ms",
            std_dev_label,
            std_dev_padding,
            std_dev_duration.as_millis()
        )
    }
}

fn colorize(duration: Duration, average: Duration, width: usize) -> ColoredString {
    let s = format!("{:width$} ms", duration.as_millis());
    if duration > average {
        s.red()
    } else {
        s.normal()
    }
}

fn std_dev_duration(durations: &[Duration]) -> Duration {
    let micros: Vec<u64> = durations
        .iter()
        .map(|duration| duration.as_micros().try_into().unwrap())
        .collect();
    let count = micros.len() as u64;
    let average: u64 = micros.iter().sum::<u64>() / count;
    let micros_squared: Vec<u64> = micros.iter().map(|x| x * x).collect();

    // avg(x)^2
    let average_squared = average * average;

    // avg(x^2)
    let squared_average = micros_squared.iter().sum::<u64>() / count;

    // sqrt( avg(x^2) - avg(x)^2 )
    let std_dev = (squared_average as f64 - average_squared as f64).sqrt();

    Duration::from_micros(std_dev.trunc() as u64)
}
