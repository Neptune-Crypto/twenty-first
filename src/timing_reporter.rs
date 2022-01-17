use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    ops::Sub,
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
        let print_debug = std::env::var("DEBUG_TIMING").unwrap_or("0".to_string()) != "0";

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

        self.durations.push((label.clone(), duration));
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

// format!("{:0>8}", "110")
impl Display for TimingReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = self
            .step_durations
            .iter()
            .map(|(label, _)| label.len())
            .max()
            .unwrap_or(0);

        for (label, duration) in self.step_durations.iter() {
            write!(f, "{}", label)?;
            let padding = String::from_utf8(vec![b' '; width - label.len() + 1]).unwrap();
            write!(f, "{}", padding)?;
            write!(f, "{:.2?}\n", duration)?;
        }

        write!(f, "\n")
    }
}
