use crate::ring_buffer::{RingBufferTrait, Step, guarded_prune};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use std::collections::{BTreeSet, HashSet};
use std::fmt::{Debug, Display};
use std::time::Duration;

fn pop_dominated_values<C, Y, F>(cache: &mut C, sub_step: &Step<Option<Y>>, combine_op: F)
where
    C: RingBufferTrait<Value = Option<Y>>,
    Y: RobustnessSemantics,
    F: Fn(Y, Y) -> Y,
{
    // lemires sliding min/max optimization
    while let Some(back) = cache.get_back()
        && sub_step.value.is_some()
        && back.value.is_some()
        && combine_op(sub_step.clone().value.unwrap(), back.clone().value.unwrap())
            == sub_step.clone().value.unwrap()
    // this really needs to not use clone()
    {
        // Short-circuit: new value dominates the back of the cache.
        cache.pop_back();
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    interval: TimeInterval,
    operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    cache: C,
    eval_buffer: BTreeSet<Duration>,
    max_lookahead: Duration,
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Eventually<T, C, Y, IS_EAGER, IS_ROSI> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        cache: Option<C>,
        eval_buffer: Option<BTreeSet<Duration>>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        let max_lookahead = interval.end + operand.get_max_lookahead();
        Eventually {
            interval,
            operand,
            cache: cache.unwrap_or_else(|| C::new()),
            eval_buffer: eval_buffer.unwrap_or_default(),
            max_lookahead,
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for Eventually<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + Debug + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let sub_robustness_vec = self.operand.update(step);
        let mut output_robustness = Vec::new();

        // 1. Add new sub-formula results to the cache
        if IS_ROSI {
            for sub_step in sub_robustness_vec {
                self.eval_buffer.insert(sub_step.timestamp);
                if !self.cache.update_step(sub_step.clone()) {
                    // pop_dominated_values(&mut self.cache, &sub_step, Y::or);
                    self.cache.add_step(sub_step);
                }
            }
        } else {
            for sub_step in &sub_robustness_vec {
                self.eval_buffer.insert(sub_step.timestamp);
                pop_dominated_values(&mut self.cache, &sub_step, Y::or);
                self.cache.add_step(sub_step.clone());
            }
        }

        let mut tasks_to_remove = Vec::new();
        let current_time = step.timestamp;

        // 2. Process the evaluation buffer
        for &t_eval in self.eval_buffer.iter() {
            let window_start = t_eval + self.interval.start;

            // The first step in this window is always the max as a result of how the cache is built
            let windowed_max_value = if IS_ROSI {
                let window_end = t_eval + self.interval.end;
                self.cache
                    .iter()
                    .skip_while(|entry| entry.timestamp < window_start)
                    .take_while(|entry| entry.timestamp <= window_end)
                    .map(|entry| entry.value.clone().unwrap())
                    .fold(Y::eventually_identity(), Y::or)
            } else {
                // Standard optimization (valid for f64/bool with append-only data)
                let windowed_max_step = self
                    .cache
                    .iter()
                    .find(|entry| entry.timestamp >= window_start);

                if let Some(entry) = windowed_max_step {
                    entry.value.clone().unwrap()
                } else {
                    Y::eventually_identity()
                }
            };

            let final_value: Option<Y>;
            let mut remove_task = false;

            // state-based logic
            if current_time >= t_eval + self.max_lookahead {
                // Case 1: Full window has passed. This is a final, "closed" value.
                final_value = Some(windowed_max_value);
                remove_task = true;
            } else if IS_EAGER && windowed_max_value == Y::atomic_true() {
                // Case 2: Eager short-circuit. Found "true" before window closed.
                final_value = Some(windowed_max_value);
                remove_task = true;
            } else if IS_ROSI {
                // Case 3: Intermediate ROSI. Window is still open.
                // We must 'or' with the unknown future.
                let intermediate_value = Y::or(windowed_max_value, Y::unknown()); // Use unknown() from your code
                final_value = Some(intermediate_value);
                // DO NOT remove task, it's not finished
            } else {
                // Case 4: Cannot evaluate yet (e.g., bool/f64 and window is still open)
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

        // 3. Prune the cache and buffer
        let protected_ts = self.eval_buffer.first().copied().unwrap_or(Duration::ZERO);
        let lookahead = self.get_max_lookahead();
        guarded_prune(&mut self.cache, lookahead, protected_ts);
        for t in tasks_to_remove {
            self.eval_buffer.remove(&t);
        }

        output_robustness
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for Eventually<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

#[derive(Clone)]
pub struct Globally<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    interval: TimeInterval,
    operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    cache: C,
    eval_buffer: BTreeSet<Duration>,
    max_lookahead: Duration,
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Globally<T, C, Y, IS_EAGER, IS_ROSI> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        cache: Option<C>,
        eval_buffer: Option<BTreeSet<Duration>>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        let max_lookahead = interval.end + operand.get_max_lookahead();
        Globally {
            interval,
            operand,
            cache: cache.unwrap_or_else(|| C::new()),
            eval_buffer: eval_buffer.unwrap_or_default(),
            max_lookahead,
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for Globally<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let sub_robustness_vec = self.operand.update(step);
        let mut output_robustness = Vec::new();

        // 1. Add new sub-formula results to the cache
        if IS_ROSI {
            for sub_step in sub_robustness_vec {
                self.eval_buffer.insert(sub_step.timestamp);
                if !self.cache.update_step(sub_step.clone()) {
                    // pop_dominated_values(&mut self.cache, &sub_step, Y::and);
                    self.cache.add_step(sub_step);
                }
            }
        } else {
            for sub_step in &sub_robustness_vec {
                pop_dominated_values(&mut self.cache, &sub_step, Y::and);
                self.eval_buffer.insert(sub_step.timestamp);
                self.cache.add_step(sub_step.clone());
            }
        }

        let mut tasks_to_remove = Vec::new();
        let current_time = step.timestamp;

        // 2. Process the evaluation buffer
        for &t_eval in self.eval_buffer.iter() {
            let window_start = t_eval + self.interval.start;

            // The first step in this window is always the max as a result of how the cache is built
            let windowed_min_value = if IS_ROSI {
                let window_end = t_eval + self.interval.end;
                self.cache
                    .iter()
                    .skip_while(|entry| entry.timestamp < window_start)
                    .take_while(|entry| entry.timestamp <= window_end)
                    .map(|entry| entry.value.clone().unwrap())
                    .fold(Y::globally_identity(), Y::and)
            } else {
                // Standard optimization (valid for f64/bool with append-only data)
                let windowed_min_step = self
                    .cache
                    .iter()
                    .find(|entry| entry.timestamp >= window_start);

                if let Some(entry) = windowed_min_step {
                    entry.value.clone().unwrap()
                } else {
                    Y::globally_identity()
                }
            };

            let final_value: Option<Y>;
            let mut remove_task = false;

            // state-based logic
            if current_time >= t_eval + self.max_lookahead {
                // Case 1: Full window has passed. This is a final, "closed" value.
                final_value = Some(windowed_min_value);
                remove_task = true;
            } else if IS_EAGER && windowed_min_value == Y::atomic_false() {
                // Case 2: Eager short-circuit. Found "false" before window closed.
                final_value = Some(windowed_min_value);
                remove_task = true;
            } else if IS_ROSI {
                // Case 3: Intermediate ROSI. Window is still open.
                // We must 'and' with the unknown future.
                let intermediate_value = Y::and(windowed_min_value, Y::unknown());
                final_value = Some(intermediate_value);
                // DO NOT remove task, it's not finished
            } else {
                // Case 4: Cannot evaluate yet (e.g., bool/f64 and window is still open)
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

        // 3. Prune the cache and buffer
        let protected_ts = self.eval_buffer.first().copied().unwrap_or(Duration::ZERO);
        let lookahead = self.get_max_lookahead();
        guarded_prune(&mut self.cache, lookahead, protected_ts);
        for t in tasks_to_remove {
            self.eval_buffer.remove(&t);
        }

        output_robustness
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for Globally<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Globally<T, Y, C, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "G[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand
        )
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Eventually<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "F[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand
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
    fn eventually_operator_robustness() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut eventually = Eventually::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            interval,
            Box::new(atomic),
            None,
            None,
        );
        eventually.get_signal_identifiers();

        let signal_values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let signal_timestamps = vec![0, 2, 4, 6, 8];
        let signal: Vec<_> = signal_values
            .into_iter()
            .zip(signal_timestamps)
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let mut all_outputs = Vec::new();
        for s in &signal {
            all_outputs.extend(eventually.update(s));
        }

        let expected_outputs = [
            Step::new("output", Some(5.0), Duration::from_secs(0)),
            Step::new("output", Some(2.0), Duration::from_secs(2)),
            Step::new("output", Some(2.0), Duration::from_secs(4)),
        ];

        assert_eq!(all_outputs.len(), expected_outputs.len());
        for (output, expected) in all_outputs.iter().zip(expected_outputs.iter()) {
            assert_eq!(output.timestamp, expected.timestamp);
            assert!(
                (output.value.unwrap() - expected.value.unwrap()).abs() < 1e-9,
                "left: {}, right: {}",
                output.value.unwrap(),
                expected.value.unwrap()
            );
        }
    }

    #[test]
    fn globally_operator_robustness() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut globally = Globally::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            interval,
            Box::new(atomic),
            None,
            None,
        );
        globally.get_signal_identifiers();

        let signal_values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let signal_timestamps = vec![0, 2, 4, 6, 8];
        let signal: Vec<_> = signal_values
            .into_iter()
            .zip(signal_timestamps)
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let mut all_outputs = Vec::new();
        for s in &signal {
            all_outputs.extend(globally.update(s));
        }

        let expected_outputs = [
            Step::new("output", Some(-2.0), Duration::from_secs(0)),
            Step::new("output", Some(-5.0), Duration::from_secs(2)),
            Step::new("output", Some(-5.0), Duration::from_secs(4)),
        ];

        assert_eq!(all_outputs.len(), expected_outputs.len());
        for (output, expected) in all_outputs.iter().zip(expected_outputs.iter()) {
            assert_eq!(output.timestamp, expected.timestamp);
            assert!(
                (output.value.unwrap() - expected.value.unwrap()).abs() < 1e-9,
                "left: {}, right: {}",
                output.value.unwrap(),
                expected.value.unwrap()
            );
        }
    }

    #[test]
    fn unary_temporal_signal_identifiers() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut globally = Globally::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            interval,
            Box::new(atomic),
            None,
            None,
        );
        let ids = globally.get_signal_identifiers();
        let expected_ids: HashSet<&'static str> = vec!["x"].into_iter().collect();
        assert_eq!(ids, expected_ids);
    }
}
