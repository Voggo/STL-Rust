use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use std::collections::{BTreeSet, HashSet};
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
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for Until<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static + std::fmt::Debug,
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
                    self.eval_buffer.insert(right_update.timestamp);
                }
                (None, None) => break, // Both iterators are empty
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

        // If there is no data in the left cache it cannot be calculated yet
        if self.left_cache.is_empty() || self.right_cache.is_empty() {
            return output_robustness;
        }

        // 2. Process the evaluation buffer for tasks
        for &t_eval in self.eval_buffer.iter() {
            let window_start_t_eval = t_eval;
            let window_end_t_eval = t_eval + self.interval.end;

            // This is the outer `max` (Eventually)
            let mut max_robustness_vec = Vec::new();
            let mut falsified = false;

            // We can only evaluate up to the data we have.
            // We must use the minimum of the current time and the window end.
            let effective_end_time = current_time.min(window_end_t_eval);

            // Iterate over all t' in [window_start_t_eval, effective_end_time].
            // We use the eval_buffer as the source of t' timestamps.
            let t_prime_iter = self
                .eval_buffer
                .iter()
                .copied()
                .skip_while(|s| s < &window_start_t_eval)
                .take_while(|s| s <= &effective_end_time); // Only up to current time
            let mut right_cache_t_prime_iter = self.right_cache.iter().peekable();

            for step_psi_t_prime in t_prime_iter {
                let t_prime = step_psi_t_prime;

                // 2. Get min_{t'' \in [t_eval+a, t')} rho_phi(t'')
                let mut robustness_phi_left = self
                    .left_cache
                    .iter()
                    .skip_while(|s| s.timestamp < t_eval) // t'' >= t_eval+a
                    .take_while(|s| s.timestamp <= t_prime)
                    .filter_map(|s| s.value.clone())
                    .fold(Y::globally_identity(), Y::and);

                // if no data in in left_cache for >t_prime, we need to and with unknown
                if let Some(last_left_step) = self.left_cache.iter().find(|s| s.timestamp > t_prime)
                    && last_left_step.timestamp < t_prime
                {
                    // no data for t'' up to t', need to and with unknown
                    robustness_phi_left = Y::and(robustness_phi_left, Y::unknown());
                }

                // Advance the right_cache iterator to find the step corresponding to t_prime
                while let Some(step) = right_cache_t_prime_iter.peek() {
                    if step.timestamp < t_prime {
                        right_cache_t_prime_iter.next(); // Consume and advance
                    } else {
                        break; // Found t' or passed it
                    }
                }
                let t_prime_right_step =
                    right_cache_t_prime_iter.find(|s| s.timestamp <= self.t_max);
                // 1. Get rho_psi(t')
                let robustness_psi_right = match t_prime_right_step {
                    Some(val) => val.value.clone().unwrap(),
                    None => Y::unknown(), // Cannot compute with None
                };

                // --- EAGER FALSIFICATION CHECK ---
                // If phi has become false, and psi has not *already* made us true,
                // then we are (and will remain) false.
                if IS_EAGER
                    && robustness_phi_left == Y::atomic_false()
                    && robustness_psi_right != Y::atomic_true()
                    && self.t_max >= t_eval
                {
                    falsified = true;
                    max_robustness_vec.push(Y::atomic_false());
                    break;
                    // continue; // Short-circuit: Falsified
                }

                // 3. Combine: min(rho_psi(t'), robustness_phi_left)
                let robustness_t_prime = Y::and(robustness_psi_right, robustness_phi_left);
                max_robustness_vec.push(robustness_t_prime); // For the outer max/sup
            }

            let max_robustness = if max_robustness_vec.is_empty() {
                break; // No data to evaluate yet
            } else {
                max_robustness_vec.into_iter().reduce(Y::or).unwrap()
            };

            // ---
            // **State-based Eager/Strict/ROSI logic**
            // ---
            let final_value: Option<Y>;
            let mut remove_task = false;

            if current_time >= t_eval + self.get_max_lookahead() {
                // Case 1: Full window *and* child lookaheads have passed.
                // This is a final, "closed" value for Strict mode.
                // (This also captures Eager results that were `false` until the end)
                final_value = Some(max_robustness);
                remove_task = true;
            } else if IS_EAGER && max_robustness == Y::atomic_true() {
                // Case 2a: Eager short-circuit (Satisfaction). Found "true" before window closed.
                final_value = Some(max_robustness); // which is Y::atomic_true()
                remove_task = true;
            } else if IS_EAGER && falsified {
                // Case 2b: Eager short-circuit (Falsification). Found "false" before window closed.
                final_value = Some(max_robustness); // which is Y::atomic_false()
                remove_task = true;
            } else if IS_ROSI {
                // Case 3: Intermediate ROSI. Window is still open, no short-circuit.
                // We must account for unknown future contributions. For Until, the
                // outer sup (max_robustness) should be widened with unknown() so
                // that subsequent negations or compositions see the correct
                // refinable bounds (mirrors behavior in Eventually/Globally).
                let intermediate_value = Y::or(max_robustness, Y::unknown());
                final_value = Some(intermediate_value);
                // DO NOT remove task, it's not finished
            } else {
                // Case 4: Cannot evaluate yet (e.g., Strict/Eager bool/f64 and window is still open)
                // Since the buffer is time-ordered, we stop.
                break;
            }

            if let Some(val) = final_value {
                output_robustness.push(Step::new("output", Some(val), t_eval));
            }

            if remove_task {
                tasks_to_remove.push(t_eval);
            }
        }

        // 3. Prune the caches and remove completed tasks from the buffer.
        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        for t in tasks_to_remove {
            self.eval_buffer.remove(&t);
        }

        if let Some(last_step) = output_robustness.last() {
            self.last_eval_time = Some(last_step.timestamp);
        };

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
            println!("Updates at t={:?}: {:?}", s.timestamp, up);
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
                // assert all equal to 1
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