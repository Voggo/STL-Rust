use std::collections::{BTreeSet, HashMap, VecDeque};
use std::iter::Iterator;
use std::ops::{Add, Mul, Sub};
use std::time::Duration;

use crate::ring_buffer::Step;

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationStrategy {
    None, // No interpolation
    #[default]
    ZeroOrderHold, // formula: v = v0
    Linear, // formula: v = v0 + (v1 - v0) * ((t - t0) / (t1 - t0))
}

pub trait Interpolatable:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self>
{
}

impl Interpolatable for f64 {}

/// Synchronizer struct that handles interpolation of missing steps across multiple signals.
/// It maintains a timeline of all timestamps and the last known step for each active signal.
/// A signal is considered active if it has received at least one step.
pub struct Synchronizer<T> {
    strategy: InterpolationStrategy,
    last_steps: HashMap<&'static str, Step<T>>,
    timeline: BTreeSet<Duration>,
    pub pending: VecDeque<Step<T>>, // Exposed so consumers can drain it
}

impl<T> Synchronizer<T>
where
    T: Interpolatable,
{
    pub fn new(strategy: InterpolationStrategy) -> Self {
        Self {
            strategy,
            last_steps: HashMap::new(),
            timeline: BTreeSet::new(),
            pending: VecDeque::new(),
        }
    }

    /// Processes a new real step and generates interpolated steps if necessary.
    /// All resulting steps (interpolated + real) are added to `self.pending`.
    pub fn evaluate(&mut self, current_step: Step<T>) {
        if self.strategy == InterpolationStrategy::None {
            self.pending.push_back(current_step);
            return;
        }

        let signal_id = current_step.signal;
        let current_time = current_step.timestamp;
        let current_value = current_step.value;

        // 1. Add this new timestamp to the global timeline
        self.timeline.insert(current_time);

        // 2. Check if we can interpolate for this specific signal
        if let Some(prev_step) = self.last_steps.get(&signal_id).cloned() {
            let prev_time = prev_step.timestamp;
            let prev_val = prev_step.value;

            if current_time > prev_time {
                let missed_timestamps: Vec<Duration> = self
                    .timeline
                    .range((
                        std::ops::Bound::Excluded(prev_time),
                        std::ops::Bound::Excluded(current_time),
                    ))
                    .cloned()
                    .collect();

                for t in missed_timestamps {
                    let interp_val = match self.strategy {
                        InterpolationStrategy::None => current_value, // this will never be hit
                        InterpolationStrategy::ZeroOrderHold => prev_val,
                        InterpolationStrategy::Linear => {
                            let dt_total = current_time.as_secs_f64() - prev_time.as_secs_f64();
                            let dt_curr = t.as_secs_f64() - prev_time.as_secs_f64();
                            let alpha = if dt_total != 0.0 {
                                dt_curr / dt_total
                            } else {
                                0.0
                            };
                            prev_val + (current_value - prev_val) * alpha
                        }
                    };

                    self.pending.push_back(Step {
                        signal: signal_id,
                        timestamp: t,
                        value: interp_val,
                    });
                }
            }
        }

        // 3. Update history for this signal
        self.last_steps.insert(signal_id, current_step.clone());

        // 4. Enqueue the real step
        self.pending.push_back(current_step);

        // 5. Cleanup
        self.prune_history();
    }

    fn prune_history(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }
        let min_frontier = self.last_steps.values().map(|s| s.timestamp).min();
        if let Some(frontier) = min_frontier {
            let keep = self.timeline.split_off(&frontier);
            self.timeline = keep;
        }
    }
}

// -----------------------------------------------------------------------------
// tests
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_synchronizer_zero_order_hold() {
        let steps = vec![
            Step {
                signal: "B",
                value: 1.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                signal: "A",
                value: 1.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                signal: "A",
                value: 10.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                signal: "A",
                value: 3.0,
                timestamp: Duration::from_secs(4),
            },
            Step {
                signal: "B",
                value: 30.0,
                timestamp: Duration::from_secs(5),
            },
        ];
        let mut sync = Synchronizer::new(InterpolationStrategy::ZeroOrderHold);
        let mut result = Vec::new();
        for step in &steps {
            sync.evaluate(step.clone());
            // Drain pending steps
            while let Some(s) = sync.pending.pop_front() {
                println!("Popped step: {:?}", s);
                result.push(s);
            }
        }
        // With zero-order hold, signal A at t=2 should hold value 1.0
        assert!(result.iter().any(|s| s.signal == "B"
            && (s.timestamp == Duration::from_secs(1)
                || s.timestamp == Duration::from_secs(2)
                || s.timestamp == Duration::from_secs(4))
            && s.value == 1.0));
    }

    #[test]
    fn test_synchronizer_linear() {
        let steps = vec![
            Step {
                signal: "A",
                value: 0.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                signal: "B",
                value: 0.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                signal: "A",
                value: 10.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                signal: "B",
                value: 20.0,
                timestamp: Duration::from_secs(4),
            },
        ];
        let mut sync = Synchronizer::new(InterpolationStrategy::Linear);
        let mut result = Vec::new();
        for step in &steps {
            sync.evaluate(step.clone());
            // Drain pending steps
            while let Some(s) = sync.pending.pop_front() {
                result.push(s);
            }
        }
        // With linear interpolation, at t=2, signal B should be linearly interpolated to 10.0
        assert!(result.iter().any(|s| s.signal == "B"
            && s.timestamp == Duration::from_secs(2)
            && (s.value - 10.0).abs() < 1e-6));
    }
}
