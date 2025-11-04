use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use crate::stl::monitor::EvaluationMode;
use std::collections::{BTreeSet, HashSet};
use std::fmt::Display;
use std::time::Duration;

// This is the new function for Strict mode
fn process_binary_strict<C, Y, F>(
    left_cache: &mut C,
    right_cache: &mut C,
    combine_op: F,
) -> Vec<Step<Option<Y>>>
where
    C: RingBufferTrait<Value = Option<Y>>,
    Y: RobustnessSemantics,
    F: Fn(Y, Y) -> Y,
{
    let mut output_robustness = Vec::new();

    // In Strict mode, we only process when timestamps align.
    // We cannot use "last_known" values.
    while let (Some(left_step), Some(right_step)) =
        (left_cache.get_front(), right_cache.get_front())
    {
        let new_step = if left_step.timestamp == right_step.timestamp {
            // Timestamps are equal. Combine their current values.
            // This is the only time we produce a Some(value).
            let combined_value = left_step.value.as_ref().and_then(|l_val| {
                right_step
                    .value
                    .as_ref()
                    .map(|r_val| combine_op(l_val.clone(), r_val.clone()))
            });

            let step = left_cache.pop_front().unwrap();
            right_cache.pop_front();
            Step::new("output", combined_value, step.timestamp)
        } else {
            break;
        };
        output_robustness.push(new_step);
    }

    // In strict mode, if one cache has data and the other is empty,
    // we can't compute anything. We must wait.
    output_robustness
}

fn process_binary_eager<C, Y, F>(
    left_cache: &mut C,
    right_cache: &mut C,
    left_last_known: &mut Step<Option<Y>>,
    right_last_known: &mut Step<Option<Y>>,
    combine_op: F,
    default_atomic: Y,
) -> Vec<Step<Option<Y>>>
where
    C: RingBufferTrait<Value = Option<Y>>,
    Y: RobustnessSemantics,
    F: Fn(Y, Y) -> Y,
{
    let mut output_robustness = Vec::new();

    // Only process while both caches have data. This prevents using a stale "last_known"
    // value when one sub-formula has not yet produced a result for a given time.
    while let (Some(left_step), Some(right_step)) =
        (left_cache.get_front(), right_cache.get_front())
    {
        let new_step = if left_step.timestamp < right_step.timestamp {
            // Left is earlier. Combine its value with the last known value from the right.
            let combined_value = left_step.value.as_ref().and_then(|l_val| {
                right_last_known
                    .value
                    .as_ref()
                    .map(|r_val| combine_op(l_val.clone(), r_val.clone()))
            });
            // Update the last known value for the left side and pop the step.
            *left_last_known = left_step.clone();
            let step = left_cache.pop_front().unwrap();
            Step::new("output", combined_value, step.timestamp)
        } else if right_step.timestamp < left_step.timestamp {
            // Right is earlier. Combine its value with the last known value from the left.
            let combined_value = right_step.value.as_ref().and_then(|r_val| {
                left_last_known
                    .value
                    .as_ref()
                    .map(|l_val| combine_op(l_val.clone(), r_val.clone()))
            });
            // Update the last known value for the right side and pop the step.
            *right_last_known = right_step.clone();
            let step = right_cache.pop_front().unwrap();
            Step::new("output", combined_value, step.timestamp)
        } else {
            // Timestamps are equal. Combine their current values.
            let combined_value = left_step.value.as_ref().and_then(|l_val| {
                right_step
                    .value
                    .as_ref()
                    .map(|r_val| combine_op(l_val.clone(), r_val.clone()))
            });
            // Update last known values for both and pop from both caches.
            *left_last_known = left_step.clone();
            *right_last_known = right_step.clone();
            let step = left_cache.pop_front().unwrap();
            right_cache.pop_front();
            Step::new("output", combined_value, step.timestamp)
        };
        output_robustness.push(new_step);
    }

    // 'short-circuit' logic:
    // for and: if either left or right is false, result is false
    // for or: if either left or right is true, result is true
    let mut max_timestamp = right_last_known.timestamp.max(left_last_known.timestamp);
    while let Some(left) = left_cache.get_front() {
        let left_value = left.value.to_owned().unwrap();
        if default_atomic == left_value && max_timestamp <= left.timestamp {
            let value = combine_op(left.value.to_owned().unwrap(), default_atomic.clone());
            let new_step = Step::new("output", Some(value), left.timestamp);
            output_robustness.push(new_step.clone());
            left_cache.pop_front();
            *left_last_known = new_step.clone();
            max_timestamp = max_timestamp.max(new_step.timestamp);
        } else {
            break;
        }
    }
    while let Some(right) = right_cache.get_front() {
        let right_value = right.value.to_owned().unwrap();
        if default_atomic == right_value && max_timestamp <= right.timestamp {
            let value = combine_op(right_value, default_atomic.clone());
            let new_step = Step::new("output", Some(value), right.timestamp);
            output_robustness.push(new_step.clone());
            right_cache.pop_front();
            *right_last_known = new_step.clone();
            max_timestamp = max_timestamp.max(new_step.timestamp);
        } else {
            break;
        }
    }

    output_robustness
}

#[derive(Clone)]
pub struct And<T, C, Y> {
    pub left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub left_cache: C,
    pub right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    mode: EvaluationMode,
}

impl<T, C, Y> And<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
        mode: EvaluationMode,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        And {
            left,
            right,
            left_cache: left_cache.unwrap_or_else(|| C::new()),
            right_cache: right_cache.unwrap_or_else(|| C::new()),
            last_eval_time: None,
            left_last_known: Step::new("", None, Duration::ZERO),
            right_last_known: Step::new("", None, Duration::ZERO),
            left_signals_set: HashSet::new(),
            right_signals_set: HashSet::new(),
            mode,
        }
    }
}

impl<T, C, Y> StlOperatorTrait<T> for And<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.left.update(step);
            for update in left_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.left_cache.add_step(update);
                    }
                } else {
                    self.left_cache.add_step(update);
                }
            }
        }
        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.right.update(step);
            for update in right_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.right_cache.add_step(update);
                    }
                } else {
                    self.right_cache.add_step(update);
                }
            }
        }
        let output = match self.mode {
            EvaluationMode::Strict => {
                process_binary_strict(&mut self.left_cache, &mut self.right_cache, Y::and)
            }
            EvaluationMode::Eager => process_binary_eager(
                &mut self.left_cache,
                &mut self.right_cache,
                &mut self.left_last_known,
                &mut self.right_last_known,
                Y::and,            // The only difference!
                Y::atomic_false(), // Default atomic value for 'and'
            ),
        };

        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        if let Some(last_step) = output.last() {
            self.last_eval_time = Some(last_step.timestamp);
        }

        output
    }
}

impl<T, C, Y> SignalIdentifier for And<T, C, Y> {
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

#[derive(Clone)]
pub struct Or<T, C, Y> {
    pub left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub left_cache: C,
    pub right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    mode: EvaluationMode,
}

impl<T, C, Y> Or<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
        mode: EvaluationMode,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Or {
            left,
            right,
            left_cache: left_cache.unwrap_or_else(|| C::new()),
            right_cache: right_cache.unwrap_or_else(|| C::new()),
            last_eval_time: None,
            left_last_known: Step {
                signal: "",
                value: None,
                timestamp: Duration::ZERO,
            },
            right_last_known: Step {
                signal: "",
                value: None,
                timestamp: Duration::ZERO,
            },
            left_signals_set: HashSet::new(),
            right_signals_set: HashSet::new(),
            mode,
        }
    }
}

impl<T, C, Y> StlOperatorTrait<T> for Or<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.left.update(step);
            for update in left_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.left_cache.add_step(update);
                    }
                } else {
                    self.left_cache.add_step(update);
                }
            }
        }
        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.right.update(step);
            for update in right_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.right_cache.add_step(update);
                    }
                } else {
                    self.right_cache.add_step(update);
                }
            }
        }
        let output = match self.mode {
            EvaluationMode::Strict => {
                process_binary_strict(&mut self.left_cache, &mut self.right_cache, Y::or)
            }
            EvaluationMode::Eager => process_binary_eager(
                &mut self.left_cache,
                &mut self.right_cache,
                &mut self.left_last_known,
                &mut self.right_last_known,
                Y::or,
                Y::atomic_true(),
            ),
        };

        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);
        if let Some(last_step) = output.last() {
            self.last_eval_time = Some(last_step.timestamp);
        }
        output
    }
}

impl<T, C, Y> SignalIdentifier for Or<T, C, Y> {
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

#[derive(Clone)]
pub struct Not<T, Y> {
    pub operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
}

impl<T, Y> Not<T, Y> {
    pub fn new(operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>) -> Self
    where
        T: Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Not { operand }
    }
}

impl<T, Y> StlOperatorTrait<T> for Not<T, Y>
where
    T: Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.operand.get_max_lookahead()
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let operand_updates = self.operand.update(step);

        let output_robustness: Vec<Step<Option<Y>>> = operand_updates
            .into_iter()
            .map(|step| {
                let negated_value = step.value.map(Y::not);
                Step {
                    signal: "output",
                    value: negated_value,
                    timestamp: step.timestamp,
                }
            })
            .collect();

        output_robustness
    }
}

impl<T, Y> SignalIdentifier for Not<T, Y> {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

#[derive(Clone)]
pub struct Implies<T, C, Y> {
    pub antecedent: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub consequent: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub left_cache: C,
    pub right_cache: C,
    last_eval_time: Option<Duration>,
    pub left_last_known: Step<Option<Y>>,
    pub right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    mode: EvaluationMode,
}

impl<T, C, Y> Implies<T, C, Y> {
    pub fn new(
        antecedent: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        consequent: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
        mode: EvaluationMode,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Implies {
            antecedent,
            consequent,
            left_cache: left_cache.unwrap_or_else(|| C::new()),
            right_cache: right_cache.unwrap_or_else(|| C::new()),
            last_eval_time: None,
            left_last_known: Step {
                signal: "",
                value: None,
                timestamp: Duration::ZERO,
            },
            right_last_known: Step {
                signal: "",
                value: None,
                timestamp: Duration::ZERO,
            },
            left_signals_set: HashSet::new(),
            right_signals_set: HashSet::new(),
            mode,
        }
    }
}

// We need to look at this there might be problems with short-circuiting here.
impl<T, C, Y> StlOperatorTrait<T> for Implies<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.antecedent.get_max_lookahead();
        let right_lookahead = self.consequent.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.antecedent.update(step);
            for update in left_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.left_cache.add_step(update);
                    }
                } else {
                    self.left_cache.add_step(update);
                }
            }
        }
        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.consequent.update(step);
            for update in right_updates {
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.right_cache.add_step(update);
                    }
                } else {
                    self.right_cache.add_step(update);
                }
            }
        }
        let output = match self.mode {
            EvaluationMode::Strict => {
                process_binary_strict(&mut self.left_cache, &mut self.right_cache, Y::implies)
            }
            EvaluationMode::Eager => process_binary_eager(
                &mut self.left_cache,
                &mut self.right_cache,
                &mut self.left_last_known,
                &mut self.right_last_known,
                Y::implies,
                Y::atomic_true(), // Default atomic value for 'implies'
            ),
        };

        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        if let Some(last_step) = output.last() {
            self.last_eval_time = Some(last_step.timestamp);
        };
        output
    }
}

impl<T, C, Y> SignalIdentifier for Implies<T, C, Y> {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        let mut ids = std::collections::HashSet::new();
        self.left_signals_set
            .extend(self.antecedent.get_signal_identifiers());
        self.right_signals_set
            .extend(self.consequent.get_signal_identifiers());
        ids.extend(self.left_signals_set.iter().cloned());
        ids.extend(self.right_signals_set.iter().cloned());
        ids
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub cache: C,
    pub eval_buffer: BTreeSet<Duration>,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Eventually<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        cache: Option<C>,
        eval_buffer: Option<BTreeSet<Duration>>,
        mode: EvaluationMode,
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
            eval_buffer: eval_buffer.unwrap_or_else(|| BTreeSet::new()),
            mode,
            max_lookahead,
        }
    }
}

impl<T, C, Y> StlOperatorTrait<T> for Eventually<T, C, Y>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static, // The crucial trait bound
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let sub_robustness_vec = self.operand.update(step);
        let mut output_robustness = Vec::new();

        // 1. Add new sub-formula results to the cache and queue up new evaluation tasks.
        for sub_step in sub_robustness_vec {
            self.cache.add_step(sub_step.clone());
            // Add a task to the evaluation buffer for this new timestamp.
            // Keep track of timestamps we need to evaluate.
            self.eval_buffer
                .insert(sub_step.timestamp);
        }

        // 2. Process the evaluation buffer for tasks that can now be completed.
        while let Some(&t_eval) = self.eval_buffer.first() {

            let window_start = t_eval + self.interval.start;
            let window_end = t_eval + self.interval.end;

            // Find the maximum robustness within the part of the window we have seen so far.
            let max_in_window = self
                .cache
                .iter()
                .filter(|entry| entry.timestamp >= window_start && entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::eventually_identity(), Y::or);

            // A. Check for short-circuiting: if the max value is "true", we have a definitive result.
            if self.mode == EvaluationMode::Eager && max_in_window == Y::atomic_true() {
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            if step.timestamp >= (window_end) {
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // C. If neither condition is met, we can't resolve this task yet.
            // Since the buffer is time-ordered, we stop for this cycle.
            break;
        }

        // 3. Prune the cache to remove old values that are no longer needed.
        // A value at time `t` is only needed as long as it can be in some future window.
        // The latest possible window starts at `step.timestamp`, so we need to keep values
        // back to `step.timestamp + interval.start - interval.end`.
        // A simpler, safe bound is to just keep `interval.end` duration of history.
        self.cache.prune(self.interval.end);

        output_robustness
    }
}

impl<T, C, Y> SignalIdentifier for Eventually<T, C, Y> {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

#[derive(Clone)]
pub struct Globally<T, Y, C> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    pub cache: C,
    pub eval_buffer: BTreeSet<Duration>,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Globally<T, Y, C> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        cache: Option<C>,
        eval_buffer: Option<BTreeSet<Duration>>,
        mode: EvaluationMode,
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
            eval_buffer: eval_buffer.unwrap_or_else(|| BTreeSet::new()),
            mode,
            max_lookahead,
        }
    }
}

impl<T, C, Y> StlOperatorTrait<T> for Globally<T, Y, C>
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

        // 1. Add new sub-formula results to the cache and queue up new evaluation tasks.
        for sub_step in sub_robustness_vec {
            self.cache.add_step(sub_step.clone());
            // Add a task to the evaluation buffer for this new timestamp.
            self.eval_buffer
                .insert(sub_step.timestamp);
        }

        // 2. Process the evaluation buffer for tasks that can now be completed.
        while let Some(&t_eval) = self.eval_buffer.first() {
            let window_start = t_eval + self.interval.start;
            let window_end = t_eval + self.interval.end;

            // Find the maximum robustness within the part of the window we have seen so far.
            let max_in_window = self
                .cache
                .iter()
                .filter(|entry| entry.timestamp >= window_start && entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::globally_identity(), Y::and);

            // A. Check for short-circuiting: if the max value is "false", we have a definitive result.
            if self.mode == EvaluationMode::Eager && max_in_window == Y::atomic_false() {
                self.eval_buffer.pop_first();
                output_robustness.push(Step::new("output", Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            if step.timestamp >= (t_eval + self.max_lookahead) {
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // C. If neither condition is met, we can't resolve this task yet.
            // Since the buffer is time-ordered, we stop for this cycle.
            break;
        }

        // 3. Prune the cache to remove old values that are no longer needed.
        // A value at time `t` is only needed as long as it can be in some future window.
        // The latest possible window starts at `step.timestamp`, so we need to keep values
        // back to `step.timestamp + interval.start - interval.end`.
        // A simpler, safe bound is to just keep `interval.end` duration of history.
        self.cache.prune(self.interval.end);

        output_robustness
    }
}

impl<T, C, Y> SignalIdentifier for Globally<T, C, Y> {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        self.operand.get_signal_identifiers()
    }
}

#[derive(Clone)]
pub struct Until<T, C, Y> {
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    pub right: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    pub left_cache: C,
    pub right_cache: C,
    pub t_max: Duration,
    pub last_eval_time: Option<Duration>,
    pub eval_buffer: BTreeSet<Duration>,
    pub left_signals_set: HashSet<&'static str>,
    pub right_signals_set: HashSet<&'static str>,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Until<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
        mode: EvaluationMode,
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
            mode,
            max_lookahead,
        }
    }
}

impl<T, C, Y> StlOperatorTrait<T> for Until<T, C, Y>
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
        let mut output_robustness = Vec::new();

        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.left.update(step);
            // Add new sub-formula results to the cache and queue up new evaluation tasks.
            for update in left_updates {
                self.left_cache.add_step(update.clone());

                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.eval_buffer.insert(update.timestamp);
                    }
                } else {
                    self.eval_buffer.insert(update.timestamp);
                }
            }
        }
        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.right.update(step);
            for update in right_updates {
                self.right_cache.add_step(update.clone());
                if let Some(last_time) = self.last_eval_time {
                    if update.timestamp > last_time {
                        self.eval_buffer.insert(update.timestamp);
                    }
                } else {
                    self.eval_buffer.insert(update.timestamp);
                }
            }
        }

        // t_max is the latest time we can evaluate up to, based on available data.
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

        // Process the evaluation buffer for tasks that can now be completed.
        while let Some(&t_eval) = self.eval_buffer.first() {
            let window_start = t_eval + self.interval.start;
            let window_end = t_eval + self.interval.end;

            // Find out if the robustness values in the window allow us to conclude the until value.
            let max_in_window = self
                .left_cache
                .iter()
                .filter(|entry| entry.timestamp >= window_start && entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::globally_identity(), Y::and);

            // Short-circuit: if left is false in the window, until is false regardless of right
            if self.mode == EvaluationMode::Eager && max_in_window == Y::atomic_false() {
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(Y::atomic_false()), t_eval));
                continue; // Move to the next task
            }

            // Check if right is true at some point in the window
            let right_in_window = self
                .right_cache
                .iter()
                .filter(|entry| entry.timestamp >= window_start && entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::eventually_identity(), Y::or);

            // Short-circuit: if right is true in the window, until is true
            if self.mode == EvaluationMode::Eager
                && right_in_window == Y::atomic_true()
                && self.t_max >= t_eval
            {
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(Y::atomic_true()), t_eval));
                continue; // Move to the next task
            }

            // Normal completion: if the full time window has passed.
            if step.timestamp >= (t_eval + self.max_lookahead) {
                let until_value = Y::or(max_in_window, right_in_window);
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new("output", Some(until_value), t_eval));
                continue; // Move to the next task
            }
            // If no condition is met, we can't resolve this task yet.
            break;
        }

        // Prune the caches.
        self.left_cache.prune(self.interval.end);
        self.right_cache.prune(self.interval.end);

        // update last eval time to max in output_robustness
        if let Some(last_step) = output_robustness.last() {
            self.last_eval_time = Some(last_step.timestamp);
        }

        output_robustness
    }
}

impl<T, C, Y> SignalIdentifier for Until<T, C, Y> {
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

#[derive(Clone)]
pub enum Atomic<Y> {
    LessThan(&'static str, f64, std::marker::PhantomData<Y>),
    GreaterThan(&'static str, f64, std::marker::PhantomData<Y>),
    True(std::marker::PhantomData<Y>),
    False(std::marker::PhantomData<Y>),
}

impl<Y> Atomic<Y> {
    pub fn new_less_than(signal_name: &'static str, val: f64) -> Self {
        Atomic::LessThan(signal_name, val, std::marker::PhantomData)
    }
    pub fn new_greater_than(signal_name: &'static str, val: f64) -> Self {
        Atomic::GreaterThan(signal_name, val, std::marker::PhantomData)
    }
    pub fn new_true() -> Self {
        Atomic::True(std::marker::PhantomData)
    }
    pub fn new_false() -> Self {
        Atomic::False(std::marker::PhantomData)
    }
}

impl<T, Y> StlOperatorTrait<T> for Atomic<Y>
where
    T: Into<f64> + Clone + 'static,
    Y: RobustnessSemantics + 'static,
{
    type Output = Y;
    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let value = step.value.clone().into();
        let result = match self {
            Atomic::True(_) => Y::atomic_true(),
            Atomic::False(_) => Y::atomic_false(),
            Atomic::GreaterThan(_signal_name, c, _) => Y::atomic_greater_than(value, *c),
            Atomic::LessThan(_signal_name, c, _) => Y::atomic_less_than(value, *c),
        };

        vec![Step {
            signal: "output",
            value: Some(result),
            timestamp: step.timestamp,
        }]
    }

    fn get_max_lookahead(&self) -> Duration {
        Duration::ZERO
    }
}

impl<Y> SignalIdentifier for Atomic<Y> {
    fn get_signal_identifiers(&mut self) -> HashSet<&'static str> {
        let mut ids = std::collections::HashSet::new();
        match self {
            Atomic::LessThan(signal_name, _, _) => {
                ids.insert(*signal_name);
            }
            Atomic::GreaterThan(signal_name, _, _) => {
                ids.insert(*signal_name);
            }
            Atomic::True(_) => {}
            Atomic::False(_) => {}
        }
        ids
    }
}

impl<Y> Display for Atomic<Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atomic::LessThan(signal_name, c, _) => write!(f, "{} < {}", signal_name, c),
            Atomic::GreaterThan(signal_name, c, _) => write!(f, "{} > {}", signal_name, c),
            Atomic::True(_) => write!(f, "True"),
            Atomic::False(_) => write!(f, "False"),
        }
    }
}
impl<T, C, Y> Display for And<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) ∧ ({})",
            self.left.to_string(),
            self.right.to_string()
        )
    }
}

impl<T, C, Y> Display for Until<T, Y, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) U[{}, {}] ({})",
            self.left.to_string(),
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.right.to_string()
        )
    }
}

impl<T, C, Y> Display for Or<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) v ({})",
            self.left.to_string(),
            self.right.to_string()
        )
    }
}

impl<T, C, Y> Display for Globally<T, Y, C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "G[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

impl<T, C, Y> Display for Eventually<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "F[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}
impl<T, Y> Display for Not<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand.to_string())
    }
}
impl<T, C, Y> Display for Implies<T, C, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "({}) → ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::{RingBuffer, Step};
    use crate::stl::core::{StlOperatorTrait, TimeInterval};
    use crate::stl::monitor::EvaluationMode;
    use std::time::Duration;

    // Atomic operators
    #[test]
    fn atomic_greater_than_robustness() {
        let mut atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        atomic.get_signal_identifiers();
        let step1 = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = atomic.update(&step1);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(5.0), Duration::from_secs(5))]
        );

        let step2 = Step::new("x", 8.0, Duration::from_secs(6));
        let robustness2 = atomic.update(&step2);
        assert_eq!(
            robustness2,
            vec![Step::new("output", Some(-2.0), Duration::from_secs(6))]
        );
    }

    #[test]
    fn atomic_less_than_robustness() {
        let mut atomic = Atomic::<f64>::new_less_than("x", 10.0);
        atomic.get_signal_identifiers();
        let step1 = Step::new("x", 5.0, Duration::from_secs(5));
        let robustness = atomic.update(&step1);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(5.0), Duration::from_secs(5))]
        );

        let step2 = Step::new("x", 12.0, Duration::from_secs(6));
        let robustness2 = atomic.update(&step2);
        assert_eq!(
            robustness2,
            vec![Step::new("output", Some(-2.0), Duration::from_secs(6))]
        );
    }

    #[test]
    fn atomic_true_robustness() {
        let mut atomic = Atomic::<f64>::new_true();
        atomic.get_signal_identifiers();
        let step = Step::new("x", 0.0, Duration::from_secs(5));
        let robustness = atomic.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new(
                "output",
                Some(f64::INFINITY),
                Duration::from_secs(5)
            )]
        );
    }

    #[test]
    fn atomic_false_robustness() {
        let mut atomic = Atomic::<f64>::new_false();
        atomic.get_signal_identifiers();
        let step = Step::new("x", 0.0, Duration::from_secs(5));
        let robustness = atomic.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new(
                "output",
                Some(f64::NEG_INFINITY),
                Duration::from_secs(5)
            )]
        );
    }

    #[test]
    fn get_signal_identifiers_atomic_and_composites() {
        // Atomic greater/less
        let mut a_gt = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut a_lt = Atomic::<f64>::new_less_than("y", 5.0);
        let mut a_true = Atomic::<f64>::new_true();
        let mut a_false = Atomic::<f64>::new_false();

        let ids_gt = a_gt.get_signal_identifiers();
        let ids_lt = a_lt.get_signal_identifiers();
        let ids_true = a_true.get_signal_identifiers();
        let ids_false = a_false.get_signal_identifiers();

        let mut expected_gt: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        expected_gt.insert("x");
        let mut expected_lt: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        expected_lt.insert("y");
        let expected_empty: std::collections::HashSet<&'static str> = std::collections::HashSet::new();

        assert_eq!(ids_gt, expected_gt);
        assert_eq!(ids_lt, expected_lt);
        assert_eq!(ids_true, expected_empty);
        assert_eq!(ids_false, expected_empty);

        // Composite: And(x>10, y<5)
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            Box::new(Atomic::<f64>::new_less_than("y", 5.0)),
            None,
            None,
            EvaluationMode::Strict,
        );
        let ids_and = and.get_signal_identifiers();
        let mut expected_and: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        expected_and.insert("x");
        expected_and.insert("y");
        assert_eq!(ids_and, expected_and);

        // Composite with constant: And(True, x>10) -> should report only 'x'
        let mut and2 = And::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(Atomic::<f64>::new_true()),
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            None,
            None,
            EvaluationMode::Strict,
        );
        let ids_and2 = and2.get_signal_identifiers();
        let mut expected_and2: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        expected_and2.insert("x");
        assert_eq!(ids_and2, expected_and2);
    }

    #[test]
    fn get_signal_identifiers_nested() {
        // And(x>10, U(y>5, z<0))
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            Box::new(
                Until::<f64, RingBuffer<Option<f64>>, f64>::new(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(5),
                },
                Box::new(Atomic::<f64>::new_greater_than("y", 5.0)),
                Box::new(Atomic::<f64>::new_less_than("z", 0.0)),
                None,
                None,
                EvaluationMode::Strict,
            )),
            None,
            None,
            EvaluationMode::Strict,
        );
        let mut expected_and: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        let mut expected_and_left: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        let mut expected_and_right: std::collections::HashSet<&'static str> = std::collections::HashSet::new();
        expected_and.insert("x");
        expected_and.insert("y");
        expected_and.insert("z");
        expected_and_left.insert("x");
        expected_and_right.insert("y");
        expected_and_right.insert("z");

        let ids_and = and.get_signal_identifiers();
        assert_eq!(ids_and, expected_and);
        assert_eq!(and.left_signals_set, expected_and_left);
        assert_eq!(and.right_signals_set, expected_and_right);
    }

    // Logical operators
    #[test]
    fn not_operator_robustness() {
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut not = Not::new(Box::new(atomic));
        not.get_signal_identifiers();
        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = not.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(-5.0), Duration::from_secs(5))]
        );
    }

    #[test]
    fn and_operator_robustness_strict() {
        let atomic1 = Atomic::<f64>::new_greater_than("x", 10.0);
        let atomic2 = Atomic::<f64>::new_less_than("x", 20.0);
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
            EvaluationMode::Strict,
        );
        and.get_signal_identifiers();

        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = and.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(5.0), Duration::from_secs(5))]
        );
    }

    #[test]
    fn or_operator_robustness_strict() {
        let atomic1 = Atomic::<f64>::new_greater_than("x", 10.0);
        let atomic2 = Atomic::<f64>::new_less_than("x", 5.0);
        let mut or = Or::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
            EvaluationMode::Strict,
        );
        or.get_signal_identifiers();

        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = or.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(5.0), Duration::from_secs(5))]
        );
    }

    #[test]
    fn implies_operator_robustness_strict() {
        let atomic1 = Atomic::<f64>::new_greater_than("x", 10.0);
        let atomic2 = Atomic::<f64>::new_less_than("x", 20.0);
        let mut implies = Implies::<f64, RingBuffer<Option<f64>>, f64>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
            EvaluationMode::Strict,
        );
        implies.get_signal_identifiers();

        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = implies.update(&step);
        assert_eq!(
            robustness,
            vec![Step::new("output", Some(5.0), Duration::from_secs(5))]
        );
    }

    // Temporal operators
    #[test]
    fn eventually_operator_robustness() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let atomic = Atomic::<f64>::new_greater_than("x", 10.0);
        let mut eventually = Eventually::<f64, RingBuffer<Option<f64>>, f64>::new(
            interval,
            Box::new(atomic),
            None,
            None,
            EvaluationMode::Strict,
        );
        eventually.get_signal_identifiers();

        let signal_values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let signal_timestamps = vec![0, 2, 4, 6, 8];
        let signal: Vec<_> = signal_values
            .into_iter()
            .zip(signal_timestamps.into_iter())
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let mut all_outputs = Vec::new();
        for s in &signal {
            all_outputs.extend(eventually.update(s));
        }

        let expected_outputs = vec![
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
        let mut globally = Globally::<f64, f64, RingBuffer<Option<f64>>>::new(
            interval,
            Box::new(atomic),
            None,
            None,
            EvaluationMode::Strict,
        );
        globally.get_signal_identifiers();

        let signal_values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let signal_timestamps = vec![0, 2, 4, 6, 8];
        let signal: Vec<_> = signal_values
            .into_iter()
            .zip(signal_timestamps.into_iter())
            .map(|(val, ts)| Step::new("x", val, Duration::from_secs(ts)))
            .collect();

        let mut all_outputs = Vec::new();
        for s in &signal {
            all_outputs.extend(globally.update(s));
        }

        let expected_outputs = vec![
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
}
