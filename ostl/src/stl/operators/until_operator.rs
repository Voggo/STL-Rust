use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use std::collections::{BTreeMap, BTreeSet, HashSet, VecDeque};
use std::fmt::Display;
use std::time::Duration;

#[derive(Clone)]
pub struct Until<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    interval: TimeInterval,
    left: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    right: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    left_cache: C,
    right_cache: C,
    t_max: Duration,
    last_eval_time: Option<Duration>,
    eval_buffer: BTreeSet<Duration>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    max_lookahead: Duration,
    // New fields for Breach-style optimization
    // Stores minimum robustness of left formula over sliding windows
    left_min_cache: BTreeMap<Duration, VecDeque<(Duration, Y)>>,
    // Stores robustness values for right formula at each time point
    right_values: BTreeMap<Duration, Y>,
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Until<T, C, Y, IS_EAGER, IS_ROSI> {
    pub fn new(
        interval: TimeInterval,
        left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        let max_lookahead = interval.end + left.get_max_lookahead().max(right.get_max_lookahead());
        Until {
            interval,
            left,
            right,
            left_cache: left_cache.unwrap_or_else(|| C::new()),
            right_cache: right_cache.unwrap_or_else(|| C::new()),
            t_max: Duration::ZERO,
            last_eval_time: None,
            eval_buffer: BTreeSet::new(),
            left_signals_set: HashSet::new(),
            right_signals_set: HashSet::new(),
            max_lookahead,
            left_min_cache: BTreeMap::new(),
            right_values: BTreeMap::new(),
        }
    }

    /// Maintains a monotonic deque for efficient windowed minimum computation
    /// Similar to Breach's streaming min/max algorithm
    fn update_min_deque(
        deque: &mut VecDeque<(Duration, Y)>,
        timestamp: Duration,
        value: Y,
        window_start: Duration,
    ) where
        Y: RobustnessSemantics + PartialOrd,
    {
        // Remove elements outside the window
        while let Some(&(t, _)) = deque.front() {
            if t < window_start {
                deque.pop_front();
            } else {
                break;
            }
        }

        // Maintain monotonic property:  remove elements from back that are >= new value
        // This ensures deque always has minimum at front
        while let Some((_, v)) = deque.back() {
            if Y::and(v.clone(), value.clone()) == value {
                // value is smaller or equal, remove larger values
                deque.pop_back();
            } else {
                break;
            }
        }

        deque.push_back((timestamp, value));
    }

    /// Get the minimum robustness value over a time window efficiently
    fn get_windowed_min(&self, t_start: Duration, t_end: Duration) -> Option<Y>
    where
        Y: RobustnessSemantics + Clone,
        C: RingBufferTrait<Value = Option<Y>>,
    {
        if let Some(deque) = self.left_min_cache.get(&t_start) {
            // The front of the deque contains the minimum value for this window
            deque.front().map(|(_, v)| v.clone())
        } else {
            // Fallback: compute from left_cache directly using RingBufferTrait methods
            let mut result: Option<Y> = None;
            for step in self.left_cache.iter() {
                if step.timestamp < t_start {
                    continue;
                }
                if step.timestamp > t_end {
                    break;
                }
                if let Some(v) = step.value.clone() {
                    result = Some(match result {
                        None => v,
                        Some(a) => Y::and(a, v),
                    });
                }
            }
            result
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for Until<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static + std::fmt::Debug + Clone + PartialOrd,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let mut output_robustness = Vec::new();

        // 1. Populate caches with results from children operators
        let left_updates =
            if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
                self.left.update(step)
            } else {
                Vec::new()
            };
        let mut left_updates = left_updates.iter().peekable();

        let right_updates =
            if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
                self.right.update(step)
            } else {
                Vec::new()
            };
        let mut right_updates = right_updates.iter().peekable();

        // Process updates and build optimized caches
        while left_updates.peek().is_some() || right_updates.peek().is_some() {
            match (left_updates.peek(), right_updates.peek()) {
                (Some(l), Some(r)) if l.timestamp == r.timestamp => {
                    let left_update = left_updates.next().unwrap();
                    let right_update = right_updates.next().unwrap();

                    if IS_ROSI {
                        if !self.left_cache.update_step(left_update.clone()) {
                            self.left_cache.add_step(left_update.clone());
                        };
                        if !self.right_cache.update_step(right_update.clone()) {
                            self.right_cache.add_step(right_update.clone());
                        };
                    } else {
                        self.left_cache.add_step(left_update.clone());
                        self.right_cache.add_step(right_update.clone());
                    }

                    // Update optimized data structures
                    if let Some(val) = &left_update.value {
                        // Update min deques for all active evaluation windows
                        let mut windows_to_update = Vec::new();
                        for &t_eval in self.eval_buffer.iter() {
                            if left_update.timestamp >= t_eval {
                                windows_to_update.push(t_eval);
                            }
                        }

                        for t_eval in windows_to_update {
                            let deque = self
                                .left_min_cache
                                .entry(t_eval)
                                .or_insert_with(VecDeque::new);
                            Self::update_min_deque(
                                deque,
                                left_update.timestamp,
                                val.clone(),
                                t_eval,
                            );
                        }
                    }

                    if let Some(val) = &right_update.value {
                        self.right_values
                            .insert(right_update.timestamp, val.clone());
                    }

                    self.eval_buffer.insert(left_update.timestamp);
                }
                (Some(l), Some(r)) if l.timestamp < r.timestamp => {
                    let left_update = left_updates.next().unwrap();
                    if let (Some(last_left), Some(last_right)) = (
                        self.left_cache.iter().last(),
                        self.right_cache.iter().last(),
                    ) && last_left.timestamp < last_right.timestamp
                        && left_update.timestamp > last_right.timestamp
                    {
                        self.left_cache.add_step(Step::new(
                            "interpolated",
                            last_left.value.clone(),
                            last_right.timestamp,
                        ));
                    }

                    if IS_ROSI {
                        if !self.left_cache.update_step(left_update.clone()) {
                            self.left_cache.add_step(left_update.clone());
                        };
                    } else {
                        self.left_cache.add_step(left_update.clone());
                    }

                    // Update min deques
                    if let Some(val) = &left_update.value {
                        let mut windows_to_update = Vec::new();
                        for &t_eval in self.eval_buffer.iter() {
                            if left_update.timestamp >= t_eval {
                                windows_to_update.push(t_eval);
                            }
                        }

                        for t_eval in windows_to_update {
                            let deque = self
                                .left_min_cache
                                .entry(t_eval)
                                .or_insert_with(VecDeque::new);
                            Self::update_min_deque(
                                deque,
                                left_update.timestamp,
                                val.clone(),
                                t_eval,
                            );
                        }
                    }

                    self.eval_buffer.insert(left_update.timestamp);
                }
                (Some(_), None) => {
                    let left_update = left_updates.next().unwrap();
                    if let (Some(last_left), Some(last_right)) = (
                        self.left_cache.iter().last(),
                        self.right_cache.iter().last(),
                    ) && last_left.timestamp < last_right.timestamp
                        && left_update.timestamp > last_right.timestamp
                    {
                        self.left_cache.add_step(Step::new(
                            "interpolated",
                            last_left.value.clone(),
                            last_right.timestamp,
                        ));
                    }

                    if IS_ROSI {
                        if !self.left_cache.update_step(left_update.clone()) {
                            self.left_cache.add_step(left_update.clone());
                        };
                    } else {
                        self.left_cache.add_step(left_update.clone());
                    }

                    // Update min deques
                    if let Some(val) = &left_update.value {
                        let mut windows_to_update = Vec::new();
                        for &t_eval in self.eval_buffer.iter() {
                            if left_update.timestamp >= t_eval {
                                windows_to_update.push(t_eval);
                            }
                        }

                        for t_eval in windows_to_update {
                            let deque = self
                                .left_min_cache
                                .entry(t_eval)
                                .or_insert_with(VecDeque::new);
                            Self::update_min_deque(
                                deque,
                                left_update.timestamp,
                                val.clone(),
                                t_eval,
                            );
                        }
                    }

                    self.eval_buffer.insert(left_update.timestamp);
                }
                (Some(_), Some(_)) | (None, Some(_)) => {
                    // Implies r.timestamp < l.timestamp
                    let right_update = right_updates.next().unwrap();
                    if let (Some(last_right), Some(last_left)) = (
                        self.right_cache.iter().last(),
                        self.left_cache.iter().last(),
                    ) && last_right.timestamp < last_left.timestamp
                        && right_update.timestamp > last_left.timestamp
                    {
                        self.right_cache.add_step(Step::new(
                            "interpolated",
                            last_right.value.clone(),
                            last_left.timestamp,
                        ));
                    }

                    if IS_ROSI {
                        if !self.right_cache.update_step(right_update.clone()) {
                            self.right_cache.add_step(right_update.clone());
                        };
                    } else {
                        self.right_cache.add_step(right_update.clone());
                    }

                    if let Some(val) = &right_update.value {
                        self.right_values
                            .insert(right_update.timestamp, val.clone());
                    }

                    self.eval_buffer.insert(right_update.timestamp);
                }
                (None, None) => break,
            }
        }

        let mut tasks_to_remove = Vec::new();
        let current_time = step.timestamp;
        self.t_max = self
            .left_cache
            .iter()
            .last()
            .map_or(self.t_max, |s| s.timestamp)
            .min(
                self.right_cache
                    .iter()
                    .last()
                    .map_or(self.t_max, |s| s.timestamp),
            );

        if self.left_cache.is_empty() || self.right_cache.is_empty() {
            return output_robustness;
        }

        // 2. Process evaluation buffer - OPTIMIZED VERSION
        for &t_eval in self.eval_buffer.iter() {
            let window_end_t_eval = t_eval + self.interval.end;

            // We can only evaluate up to the data we have
            let effective_end_time = current_time.min(window_end_t_eval);

            // OPTIMIZATION: Instead of nested iteration, we use precomputed values
            // For Until:  ρ(φ U[a,b] ψ)(t) = sup_{t' ∈ [t+a, t+b]} [ ρ(ψ)(t') ∧ inf_{t'' ∈ [t+a, t')} ρ(φ)(t'') ]

            let mut max_robustness = None;
            let mut falsified = false;

            // Iterate over potential satisfaction points for right formula
            for (&t_prime, right_val) in self.right_values.range(t_eval..=effective_end_time) {
                // Get minimum of left formula from t_eval to t_prime
                let left_min = if t_prime == t_eval {
                    // At the boundary, we use the globally identity
                    Some(Y::globally_identity())
                } else {
                    // Use optimized windowed minimum
                    self.get_windowed_min(t_eval, t_prime.saturating_sub(Duration::from_nanos(1)))
                };

                let robustness_t_prime = match &left_min {
                    Some(l_min) => Y::and(right_val.clone(), l_min.clone()),
                    None => {
                        if IS_ROSI {
                            Y::and(right_val.clone(), Y::unknown())
                        } else {
                            continue; // Skip if we don't have data
                        }
                    }
                };

                // EAGER FALSIFICATION CHECK
                if IS_EAGER
                    && left_min
                        .as_ref()
                        .map_or(false, |l| l.clone() == Y::atomic_false())
                    && *right_val != Y::atomic_true()
                    && self.t_max >= t_eval
                {
                    falsified = true;
                    max_robustness = Some(Y::atomic_false());
                    break;
                }

                // Update maximum robustness
                max_robustness = Some(match max_robustness {
                    None => robustness_t_prime,
                    Some(current_max) => Y::or(current_max, robustness_t_prime),
                });

                // EAGER SATISFACTION CHECK
                if IS_EAGER && max_robustness.as_ref() == Some(&Y::atomic_true()) {
                    break;
                }
            }

            let max_robustness_val = match max_robustness {
                Some(val) => val,
                None => continue, // No data to evaluate yet
            };

            // State-based Eager/Strict/ROSI logic
            let final_value: Option<Y>;
            let mut remove_task = false;

            if current_time >= t_eval + self.get_max_lookahead() {
                // Case 1: Full window and child lookaheads have passed
                final_value = Some(max_robustness_val);
                remove_task = true;
            } else if IS_EAGER && max_robustness_val == Y::atomic_true() {
                // Case 2a: Eager short-circuit (Satisfaction)
                final_value = Some(max_robustness_val);
                remove_task = true;
            } else if IS_EAGER && falsified {
                // Case 2b: Eager short-circuit (Falsification)
                final_value = Some(max_robustness_val);
                remove_task = true;
            } else if IS_ROSI {
                // Case 3: Intermediate ROSI - widen with unknown
                let intermediate_value = Y::or(max_robustness_val, Y::unknown());
                final_value = Some(intermediate_value);
            } else {
                // Case 4: Cannot evaluate yet
                break;
            }

            if let Some(val) = final_value {
                output_robustness.push(Step::new("output", Some(val), t_eval));
            }

            if remove_task {
                tasks_to_remove.push(t_eval);
            }
        }

        // 3. Prune caches and remove completed tasks
        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        // Prune optimization caches
        let cutoff_time = current_time.saturating_sub(lookahead);
        self.left_min_cache.retain(|&k, _| k >= cutoff_time);
        self.right_values.retain(|&k, _| k >= cutoff_time);

        for t in tasks_to_remove {
            self.eval_buffer.remove(&t);
            self.left_min_cache.remove(&t);
        }

        if let Some(last_step) = output_robustness.last() {
            self.last_eval_time = Some(last_step.timestamp);
        }

        output_robustness
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for Until<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        let mut ids = std::collections::HashSet::new();
        self.left_signals_set
            .extend(self.left.get_signal_identifiers());
        self.right_signals_set
            .extend(self.right.get_signal_identifiers());
        ids.extend(self.left_signals_set.iter().cloned());
        ids.extend(self.right_signals_set.iter().cloned());
        ids
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Until<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) U[{}, {}] ({})",
            self.left,
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.right
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::{RingBuffer, Step};
    use crate::stl::core::{StlOperatorTrait, TimeInterval};
    use crate::stl::operators::atomic_operators::Atomic;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

    #[test]
    fn until_operator_robustness() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic_left = Atomic::<bool>::new_less_than("x", 10.0);
        let atomic_right = Atomic::<bool>::new_greater_than("x", 5.0);
        let mut until = Until::<f64, RingBuffer<Option<bool>>, bool, false, false>::new(
            interval,
            Box::new(atomic_left),
            Box::new(atomic_right),
            None,
            None,
        );
        until.get_signal_identifiers();
        let signal_values = vec![2.0, 2.0, 2.0, 6.0, 12.0];
        let signal_timestamps = vec![0, 2, 4, 6, 8];

        let signal: Vec<_> = signal_values
            .into_iter()
            .zip(signal_timestamps)
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let expected_outputs = [
            Step::new("output", Some(false), Duration::from_secs(0)),
            Step::new("output", Some(true), Duration::from_secs(2)),
            Step::new("output", Some(true), Duration::from_secs(4)),
        ];

        let mut all_outputs = Vec::new();
        for s in &signal {
            let up = until.update(s);
            println!("Updates at t={:?}:  {:?}", s.timestamp, up);
            all_outputs.extend(up);
        }

        assert_eq!(all_outputs.len(), expected_outputs.len());
        for (output, expected) in all_outputs.iter().zip(expected_outputs.iter()) {
            assert_eq!(output.timestamp, expected.timestamp);
            assert_eq!(output.value, expected.value);
        }
    }

    #[test]
    fn until_operator_interpolation() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic_left = Atomic::<bool>::new_greater_than("x", 5.0);
        let atomic_right = Atomic::<bool>::new_less_than("y", 10.0);
        let mut until = Until::<f64, RingBuffer<Option<bool>>, bool, false, false>::new(
            interval,
            Box::new(atomic_left),
            Box::new(atomic_right),
            None,
            None,
        );
        until.get_signal_identifiers();
        let signal_values_x = vec![6.0, 12.0, 8.0];
        let signal_timestamps_x = vec![0, 2, 4];
        let signal_values_y = vec![15.0, 8.0];
        let signal_timestamps_y = vec![1, 3];

        let signal_x: Vec<_> = signal_values_x
            .into_iter()
            .zip(signal_timestamps_x)
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let signal_y: Vec<_> = signal_values_y
            .into_iter()
            .zip(signal_timestamps_y)
            .map(|(val, ts)| Step::new("y", val, Duration::from_secs(ts)))
            .collect();

        // zip the two signals based on timestamps
        let mut all_steps = Vec::new();
        all_steps.extend(signal_x);
        all_steps.extend(signal_y);
        all_steps.sort_by_key(|s| s.timestamp);

        for s in &all_steps {
            let _ = until.update(s);
            if s.timestamp == Duration::from_secs(0) {
                assert_eq!(until.left_cache.len(), 1);
                assert!(until.left_cache.iter().all(|step| step.value == Some(true)));
                assert_eq!(until.right_cache.len(), 0);
            } else if s.timestamp == Duration::from_secs(1) {
                assert_eq!(until.left_cache.len(), 1);
                assert!(until.left_cache.iter().all(|step| step.value == Some(true)));
                assert_eq!(until.right_cache.len(), 1);
                assert!(
                    until
                        .right_cache
                        .iter()
                        .all(|step| step.value == Some(false))
                );
            } else if s.timestamp == Duration::from_secs(2) {
                assert_eq!(until.left_cache.len(), 3);
                assert_eq!(until.right_cache.len(), 1);
            } else if s.timestamp == Duration::from_secs(3) {
                assert_eq!(until.left_cache.len(), 3);
                assert!(until.left_cache.iter().all(|step| step.value == Some(true)));
                assert_eq!(until.right_cache.len(), 3);
                // first two are false, last is true
                let mut iter = until.right_cache.iter();
                assert_eq!(iter.next().unwrap().value, Some(false));
                assert_eq!(iter.next().unwrap().value, Some(false));
                assert_eq!(iter.next().unwrap().value, Some(true));
            } else if s.timestamp == Duration::from_secs(4) {
                assert_eq!(until.left_cache.len(), 5);
                assert!(until.left_cache.iter().all(|step| step.value == Some(true)));
                assert_eq!(until.right_cache.len(), 3);
                let mut iter = until.right_cache.iter();
                assert_eq!(iter.next().unwrap().value, Some(false));
                assert_eq!(iter.next().unwrap().value, Some(false));
                assert_eq!(iter.next().unwrap().value, Some(true));
            }
        }
    }

    #[test]
    fn until_signal_identifiers() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic_left = Atomic::<bool>::new_greater_than("x", 5.0);
        let atomic_right = Atomic::<bool>::new_less_than("y", 10.0);
        let mut until = Until::<f64, RingBuffer<Option<bool>>, bool, false, false>::new(
            interval,
            Box::new(atomic_left),
            Box::new(atomic_right),
            None,
            None,
        );
        let ids = until.get_signal_identifiers();
        let expected_ids: HashSet<&'static str> = vec!["x", "y"].into_iter().collect();
        assert_eq!(ids, expected_ids);
    }
}
