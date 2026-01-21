use std::collections::{BTreeSet, HashMap, VecDeque};
use std::iter::Iterator;
use std::ops::{Add, Mul, Sub};
use std::time::Duration;

use crate::ring_buffer::Step;

// -----------------------------------------------------------------------------
// 1. Interpolation Strategy
// -----------------------------------------------------------------------------

/// Defines how values are interpolated when synchronizing signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationStrategy {
    /// Hold the previous value (step function)
    ZeroOrderHold,
    /// Linear interpolation between points
    Linear,
}

impl Default for InterpolationStrategy {
    fn default() -> Self {
        Self::ZeroOrderHold
    }
}

// -----------------------------------------------------------------------------
// 2. Trait Definition
// -----------------------------------------------------------------------------

/// Trait to define how values are interpolated.
/// You can implement this for f32, f64, or custom structs.
pub trait Interpolatable:
    Copy + Add<Output = Self> + Sub<Output = Self> + Mul<f64, Output = Self>
{
}

impl Interpolatable for f64 {}

// -----------------------------------------------------------------------------
// 3. The Iterator Adapter
// -----------------------------------------------------------------------------

/// An iterator that wraps a stream of Steps and injects interpolated steps
/// to ensure that if ANY signal has a value at time T, ALL active signals
/// will produce a value at time T.
pub struct SynchronizedStream<I, T> {
    iter: I,
    strategy: InterpolationStrategy,

    // State
    last_steps: HashMap<&'static str, Step<T>>, // The most recent real point for each signal
    timeline: BTreeSet<Duration>,               // Global union of timestamps seen so far
    pending: VecDeque<Step<T>>,                 // Buffer for steps ready to be emitted

                                                // Optimization: Track if we have seen all expected signals to start pruning?
                                                // Or just prune based on what we have.
}

impl<I, T> SynchronizedStream<I, T>
where
    I: Iterator<Item = Step<T>>,
{
    /// Creates a new synchronized stream with the default interpolation strategy (zero-order hold).
    pub fn new(iter: I) -> Self {
        Self::with_strategy(iter, InterpolationStrategy::default())
    }

    /// Creates a new synchronized stream with a specific interpolation strategy.
    pub fn with_strategy(iter: I, strategy: InterpolationStrategy) -> Self {
        Self {
            iter,
            strategy,
            last_steps: HashMap::new(),
            timeline: BTreeSet::new(),
            pending: VecDeque::new(),
        }
    }

    /// Prunes timestamps from history that are no longer needed by any signal.
    /// This keeps memory usage low (proportional to signal skew, not total time).
    fn prune_history(&mut self) {
        if self.last_steps.is_empty() {
            return;
        }

        // Find the "Global Frontier": the oldest "last_seen" time across all signals.
        // We cannot possibly interpolate before this time for any signal,
        // because we have already moved past it.
        let min_frontier = self.last_steps.values().map(|s| s.timestamp).min();

        if let Some(frontier) = min_frontier {
            // Remove all timestamps strictly older than the frontier
            // split_off returns elements >= frontier, leaving elements < frontier in self.timeline.
            // We want to KEEP >= frontier.
            let keep = self.timeline.split_off(&frontier);
            self.timeline = keep;
        }
    }
}

// -----------------------------------------------------------------------------
// 3. Iterator Implementation
// -----------------------------------------------------------------------------

impl<I, T> Iterator for SynchronizedStream<I, T>
where
    I: Iterator<Item = Step<T>>,
    T: Interpolatable,
{
    type Item = Step<T>;

    fn next(&mut self) -> Option<Self::Item> {
        // 1. If we have pending interpolated events, emit them first.
        if let Some(step) = self.pending.pop_front() {
            return Some(step);
        }

        // 2. Fetch the next real event from the source
        let current_step = self.iter.next()?;

        let signal_id = current_step.signal;
        let current_time = current_step.timestamp;
        let current_value = current_step.value;

        // 3. Add this new timestamp to the global timeline
        self.timeline.insert(current_time);

        // 4. Check if we can interpolate for this specific signal
        if let Some(prev_step) = self.last_steps.get(&signal_id).cloned() {
            let prev_time = prev_step.timestamp;
            let prev_val = prev_step.value;

            // If this step moves time forward
            if current_time > prev_time {
                // Find all timestamps T existing in other signals such that: prev_time < T < current_time
                // We use a range query on the BTreeSet.
                // We collect them to a vector to avoid borrowing conflicts with self.pending
                let missed_timestamps: Vec<Duration> = self
                    .timeline
                    .range((
                        std::ops::Bound::Excluded(prev_time),
                        std::ops::Bound::Excluded(current_time),
                    ))
                    .cloned()
                    .collect();

                // Generate interpolated steps
                for t in missed_timestamps {
                    let interp_val = match self.strategy {
                        InterpolationStrategy::ZeroOrderHold => {
                            // Hold the previous value
                            prev_val
                        }
                        InterpolationStrategy::Linear => {
                            // Linear Interpolation Math
                            let dt_total = current_time.as_secs_f64() - prev_time.as_secs_f64();
                            let dt_curr = t.as_secs_f64() - prev_time.as_secs_f64();

                            // Avoid division by zero (though current_time > prev_time check handles strict inequality)
                            let alpha = if dt_total != 0.0 {
                                dt_curr / dt_total
                            } else {
                                0.0
                            };

                            // v_interp = prev + (next - prev) * alpha
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

        // 5. Update history for this signal
        self.last_steps.insert(signal_id, current_step.clone());

        // 6. Enqueue the real step (it comes after any interpolated ones we just added)
        self.pending.push_back(current_step);

        // 7. Perform efficient cleanup
        // This ensures the BTreeSet doesn't grow indefinitely.
        self.prune_history();

        // 8. Return the next item (which is either the first interpolated one, or the real one)
        self.pending.pop_front()
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
                signal: "A",
                value: 1.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                signal: "B",
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
        let sync_iter = SynchronizedStream::new(steps.into_iter());
        let result: Vec<Step<f64>> = sync_iter.collect();
        println!("Synchronized Steps (Zero-Order Hold):");
        for step in &result {
            println!("{:?}", step);
        }
        // With zero-order hold, signal A at t=2 should hold value 1.0
        assert!(
            result.iter().any(|s| s.signal == "A"
                && s.timestamp == Duration::from_secs(2)
                && s.value == 1.0)
        );
    }

    #[test]
    fn test_synchronizer_linear() {
        let steps = vec![
            Step {
                signal: "A",
                value: 1.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                signal: "B",
                value: 10.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                signal: "A",
                value: 3.0,
                timestamp: Duration::from_secs(4),
            },
        ];
        let sync_iter =
            SynchronizedStream::with_strategy(steps.into_iter(), InterpolationStrategy::Linear);
        let result: Vec<Step<f64>> = sync_iter.collect();
        println!("Synchronized Steps (Linear):");
        for step in &result {
            println!("{:?}", step);
        }
        // With linear interpolation, signal A at t=2 should be 1.0 + (3.0-1.0) * (1/3) ≈ 1.67
        let interp_step = result
            .iter()
            .find(|s| s.signal == "A" && s.timestamp == Duration::from_secs(2));
        assert!(interp_step.is_some());
        let interp_val = interp_step.unwrap().value;
        assert!((interp_val - 1.666666).abs() < 0.01);
    }

    #[test]
    fn test_default_is_zero_order_hold() {
        assert_eq!(
            InterpolationStrategy::default(),
            InterpolationStrategy::ZeroOrderHold
        );
    }

    #[test]
    fn test_timeline_size() {
        // generate random signal 'A' with 1000 steps
        let mut steps = Vec::new();
        for i in 0..1000 {
            steps.push(Step {
                signal: "A",
                value: i as f64,
                timestamp: Duration::from_millis(i * 10),
            });
        }
        

        let mut sync_iter = SynchronizedStream::new(steps.into_iter());
        // Consume the iterator
        while let Some(_step) = sync_iter.next() {
            assert!(sync_iter.timeline.len() <= 1);
        }

        // generate two signals of size 100. A has even timestamps, B has odd timestamps
        let mut steps = Vec::new();
        for i in 0..100 {
            steps.push(Step {
                signal: "A",
                value: i as f64,
                timestamp: Duration::from_millis(i * 2),
            });
            steps.push(Step {
                signal: "B",
                value: i as f64,
                timestamp: Duration::from_millis(i * 2 + 1),
            });
        }
        let mut sync_iter = SynchronizedStream::new(steps.into_iter());
        // Consume the iterator
        while let Some(_step) = sync_iter.next() {
            assert!(sync_iter.timeline.len() <= 2);           
        }
    }
}
