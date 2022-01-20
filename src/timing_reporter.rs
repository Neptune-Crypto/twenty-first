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
        let columns: Vec<(String, String)> = self
            .step_durations
            .iter()
            .map(|(label, duration)| (label.clone(), format!("{:.2?}", duration)))
            .collect();

        let label_width = columns
            .iter()
            .map(|(label, _)| label.len())
            .max()
            .unwrap_or(0);

        let duration_width = columns
            .iter()
            .map(|(_, duration)| duration.len())
            .max()
            .unwrap_or(0);

        let space = 2;
        writeln!(f, "\n")?; // Add at least one empty line before report
        for (label, duration) in columns {
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
