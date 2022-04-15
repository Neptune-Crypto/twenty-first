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

            println!("{} in {:.2?}", label, rel_duration);
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
            .map(|(label, duration)| (label.len(), format!("{:.?}", duration).len()))
            .into_iter()
            .unzip();
        let label_width = label_widths.into_iter().max().unwrap_or(0);
        let duration_width = duration_widths.into_iter().max().unwrap_or(0);

        let durations: Vec<Duration> = self
            .step_durations
            .iter()
            .map(|(_, duration)| *duration)
            .collect();

        let num_steps = self.step_durations.len().try_into().unwrap();
        let average_duration = self.total_duration / num_steps;
        let std_dev_duration = std_dev_duration(durations);

        writeln!(
            f,
            "\navg = {:.2?}, std.dev. = {:.2?}\n",
            average_duration, std_dev_duration
        )?;

        let colored_columns: Vec<(String, ColoredString)> = self
            .step_durations
            .iter()
            .map(|(label, duration)| (label.clone(), colorize(*duration, average_duration)))
            .collect();

        let space = 2;
        writeln!(f, "\n")?; // Add at least one empty line before report
        for (label, duration) in colored_columns {
            let padding = String::from_utf8(vec![b' '; label_width - label.len() + space]).unwrap();
            writeln!(f, "{}{}{}", label, padding, duration)?;
        }

        writeln!(
            f,
            "{}{}{}",
            String::from_utf8(vec![b'-'; label_width]).unwrap(),
            String::from_utf8(vec![b' '; space]).unwrap(),
            String::from_utf8(vec![b'-'; duration_width]).unwrap()
        )?;

        let total_label = "total";
        let total_padding =
            String::from_utf8(vec![b' '; label_width - total_label.len() + space]).unwrap();

        writeln!(
            f,
            "{}{}{:.2?}",
            total_label, total_padding, self.total_duration
        )
    }
}

fn colorize(duration: Duration, average: Duration) -> ColoredString {
    let s = format!("{:.2?}", duration);
    if duration > average {
        s.red()
    } else {
        s.normal()
    }
}

fn std_dev_duration(durations: Vec<Duration>) -> Duration {
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
