use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{RobustnessSemantics, StlOperatorTrait, TimeInterval};
use crate::stl::monitor::EvaluationMode;
use std::collections::BTreeSet;
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
        let new_step = if left_step.timestamp < right_step.timestamp {
            // Left is earlier. Right has no value for this time.
            // Strict propagation of None.
            let step = left_cache.pop_front().unwrap();
            Step::new(None, step.timestamp)
        } else if right_step.timestamp < left_step.timestamp {
            // Right is earlier. Left has no value for this time.
            // Strict propagation of None.
            let step = right_cache.pop_front().unwrap();
            Step::new(None, step.timestamp)
        } else {
            // Timestamps are equal. Combine their current values.j
            // This is the only time we produce a Some(value).
            let combined_value = left_step.value.as_ref().and_then(|l_val| {
                right_step
                    .value
                    .as_ref()
                    .map(|r_val| combine_op(l_val.clone(), r_val.clone()))
            });

            let step = left_cache.pop_front().unwrap();
            right_cache.pop_front();
            Step::new(combined_value, step.timestamp)
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
            Step::new(combined_value, step.timestamp)
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
            Step::new(combined_value, step.timestamp)
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
            Step::new(combined_value, step.timestamp)
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
            let new_step = Step::new(Some(value), left.timestamp);
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
            let new_step = Step::new(Some(value), right.timestamp);
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
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub left_cache: C,
    pub right_cache: C,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    mode: EvaluationMode,
}

impl<T, C, Y> And<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
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
            left_last_known: Step::new(None, Duration::ZERO),
            right_last_known: Step::new(None, Duration::ZERO),
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.left.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.right.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
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

        let max_timestamp = self
            .right_last_known
            .timestamp
            .max(self.left_last_known.timestamp);
        self.left_cache.prune(max_timestamp);
        self.right_cache.prune(max_timestamp);

        output
    }
}

#[derive(Clone)]
pub struct Or<T, C, Y> {
    pub left: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub left_cache: C,
    pub right_cache: C,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    mode: EvaluationMode,
}

impl<T, C, Y> Or<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
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
            left_last_known: Step {
                value: None,
                timestamp: Duration::ZERO,
            },
            right_last_known: Step {
                value: None,
                timestamp: Duration::ZERO,
            },
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.left.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.right.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
        }
        let output = match self.mode {
            EvaluationMode::Strict => process_binary_strict(
                &mut self.left_cache,
                &mut self.right_cache,
                Y::or,
            ),
            EvaluationMode::Eager => process_binary_eager(
                &mut self.left_cache,
                &mut self.right_cache,
                &mut self.left_last_known,
                &mut self.right_last_known,
                Y::or,
                Y::atomic_false(),
            ),
        };
        let max_timestamp = self
            .right_last_known
            .timestamp
            .max(self.left_last_known.timestamp);
        self.left_cache.prune(max_timestamp);
        self.right_cache.prune(max_timestamp);
        output
    }
}

#[derive(Clone)]
pub struct Not<T, Y> {
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
}

impl<T, Y> Not<T, Y> {
    pub fn new(operand: Box<dyn StlOperatorTrait<T, Output = Y>>) -> Self
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        self.operand.get_max_lookahead()
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let operand_updates = self.operand.robustness(step);

        let output_robustness: Vec<Step<Option<Y>>> = operand_updates
            .into_iter()
            .map(|step| {
                let negated_value = step.value.map(Y::not);
                Step {
                    value: negated_value,
                    timestamp: step.timestamp,
                }
            })
            .collect();

        output_robustness
    }
}

#[derive(Clone)]
pub struct Implies<T, C, Y> {
    pub antecedent: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub consequent: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub left_cache: C,
    pub right_cache: C,
    pub left_last_known: Step<Option<Y>>,
    pub right_last_known: Step<Option<Y>>,
    mode: EvaluationMode,
}

impl<T, C, Y> Implies<T, C, Y> {
    pub fn new(
        antecedent: Box<dyn StlOperatorTrait<T, Output = Y>>,
        consequent: Box<dyn StlOperatorTrait<T, Output = Y>>,
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
            left_last_known: Step {
                value: None,
                timestamp: Duration::ZERO,
            },
            right_last_known: Step {
                value: None,
                timestamp: Duration::ZERO,
            },
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.antecedent.get_max_lookahead();
        let right_lookahead = self.consequent.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.antecedent.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.consequent.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
        }

        match self.mode {
            EvaluationMode::Strict => process_binary_strict(
                &mut self.left_cache,
                &mut self.right_cache,
                Y::implies,
            ),
            EvaluationMode::Eager => process_binary_eager(
                &mut self.left_cache,
                &mut self.right_cache,
                &mut self.left_last_known,
                &mut self.right_last_known,
                Y::implies,
                Y::atomic_false(), // Default atomic value for 'implies'
            ),
        }
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub cache: C,
    pub eval_buffer: C,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Eventually<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
        cache: Option<C>,
        eval_buffer: Option<C>,
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
            eval_buffer: eval_buffer.unwrap_or_else(|| C::new()),
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let sub_robustness_vec = self.operand.robustness(step);
        let mut output_robustness = Vec::new();

        // 1. Add new sub-formula results to the cache and queue up new evaluation tasks.
        for sub_step in sub_robustness_vec {
            self.cache.add_step(sub_step.clone());
            // Add a task to the evaluation buffer for this new timestamp.
            // The `value` is None because it's just a placeholder for a pending task.
            self.eval_buffer
                .add_step(Step::new(None, sub_step.timestamp));
        }

        // 2. Process the evaluation buffer for tasks that can now be completed.
        while let Some(eval_task) = self.eval_buffer.get_front() {
            let t_eval = eval_task.timestamp;
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
                self.eval_buffer.pop_front(); // Task is done
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            // if step.timestamp >= (t_eval + self.max_lookahead) {
            if step.timestamp >= (window_end) {
                self.eval_buffer.pop_front(); // Task is done
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
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

#[derive(Clone)]
pub struct Globally<T, Y, C> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub cache: C,
    pub eval_buffer: C,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Globally<T, Y, C> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
        cache: Option<C>,
        eval_buffer: Option<C>,
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
            eval_buffer: eval_buffer.unwrap_or_else(|| C::new()),
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let sub_robustness_vec = self.operand.robustness(step);
        let mut output_robustness = Vec::new();

        // 1. Add new sub-formula results to the cache and queue up new evaluation tasks.
        for sub_step in sub_robustness_vec {
            self.cache.add_step(sub_step.clone());
            // Add a task to the evaluation buffer for this new timestamp.
            // The `value` is None because it's just a placeholder for a pending task.
            self.eval_buffer
                .add_step(Step::new(None, sub_step.timestamp));
        }

        // 2. Process the evaluation buffer for tasks that can now be completed.
        while let Some(eval_task) = self.eval_buffer.get_front() {
            let t_eval = eval_task.timestamp;
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
                self.eval_buffer.pop_front();
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            if step.timestamp >= (t_eval + self.max_lookahead) {
                self.eval_buffer.pop_front(); // Task is done
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
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

#[derive(Clone)]
pub struct Until<T, C, Y> {
    pub interval: TimeInterval,
    pub left: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub right: Box<dyn StlOperatorTrait<T, Output = Y> + 'static>,
    pub left_cache: C,
    pub right_cache: C,
    pub t_max: Duration,
    pub last_eval_time: Option<Duration>,
    pub eval_buffer: BTreeSet<Duration>,
    pub mode: EvaluationMode,
    max_lookahead: Duration,
}

impl<T, C, Y> Until<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn get_max_lookahead(&self) -> Duration {
        self.max_lookahead
    }

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.left.robustness(step);
        let right_updates = self.right.robustness(step);
        let mut output_robustness = Vec::new();

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
                output_robustness.push(Step::new(Some(Y::atomic_false()), t_eval));
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
                output_robustness.push(Step::new(Some(Y::atomic_true()), t_eval));
                continue; // Move to the next task
            }

            // Normal completion: if the full time window has passed.
            if step.timestamp >= (t_eval + self.max_lookahead) {
                let until_value = Y::or(max_in_window, right_in_window);
                self.eval_buffer.pop_first(); // Task is done
                output_robustness.push(Step::new(Some(until_value), t_eval));
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

#[derive(Clone)]
pub enum Atomic<Y> {
    LessThan(f64, std::marker::PhantomData<Y>),
    GreaterThan(f64, std::marker::PhantomData<Y>),
    True(std::marker::PhantomData<Y>),
    False(std::marker::PhantomData<Y>),
}

impl<Y> Atomic<Y> {
    pub fn new_less_than(val: f64) -> Self {
        Atomic::LessThan(val, std::marker::PhantomData)
    }
    pub fn new_greater_than(val: f64) -> Self {
        Atomic::GreaterThan(val, std::marker::PhantomData)
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
    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let value = step.value.clone().into();
        let result = match self {
            Atomic::True(_) => Y::atomic_true(),
            Atomic::False(_) => Y::atomic_false(),
            Atomic::GreaterThan(c, _) => Y::atomic_greater_than(value, *c),
            Atomic::LessThan(c, _) => Y::atomic_less_than(value, *c),
        };

        vec![Step {
            value: Some(result),
            timestamp: step.timestamp,
        }]
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn get_max_lookahead(&self) -> Duration {
        Duration::ZERO
    }
}

impl<Y> Display for Atomic<Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Atomic::LessThan(c, _) => write!(f, "x < {}", c),
            Atomic::GreaterThan(c, _) => write!(f, "x > {}", c),
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
    use crate::ring_buffer::RingBuffer;
    use std::time::Duration;
    fn get_signal_1() -> Vec<Step<f64>> {
        let inputs = vec![
            Step {
                value: 0.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                value: 6.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                value: 0.0,
                timestamp: Duration::from_secs(3),
            },
            Step {
                value: 8.0,
                timestamp: Duration::from_secs(4),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(5),
            },
            Step {
                value: 7.0,
                timestamp: Duration::from_secs(6),
            },
        ];
        inputs
    }

    fn get_signal_2() -> Vec<Step<f64>> {
        let inputs = vec![
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                value: 4.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                value: 4.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(3),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(4),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(5),
            },
            Step {
                value: 2.0,
                timestamp: Duration::from_secs(6),
            },
            Step {
                value: 6.0,
                timestamp: Duration::from_secs(7),
            },
            Step {
                value: 6.0,
                timestamp: Duration::from_secs(8),
            },
            Step {
                value: 6.0,
                timestamp: Duration::from_secs(9),
            },
            Step {
                value: 6.0,
                timestamp: Duration::from_secs(10),
            },
        ];
        inputs
    }
    fn get_signal_3() -> Vec<Step<f64>> {
        let inputs = vec![
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(1),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                value: 2.0,
                timestamp: Duration::from_secs(3),
            },
            Step {
                value: 3.0,
                timestamp: Duration::from_secs(4),
            },
            Step {
                value: 4.0,
                timestamp: Duration::from_secs(5),
            },
            Step {
                value: 0.0,
                timestamp: Duration::from_secs(6),
            },
            Step {
                value: 0.0,
                timestamp: Duration::from_secs(7),
            },
            Step {
                value: 0.0,
                timestamp: Duration::from_secs(8),
            },
            Step {
                value: 1.0,
                timestamp: Duration::from_secs(9),
            },
            Step {
                value: 2.0,
                timestamp: Duration::from_secs(10),
            },
        ];
        inputs
    }

    #[test]
    fn test_1() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let atomic = Atomic::<bool>::new_greater_than(5.0);

        let mut op = And::new(
            Box::new(Eventually::new(
                interval.clone(),
                Box::new(atomic.clone()),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
                EvaluationMode::Eager,
            )),
            Box::new(Atomic::<bool>::new_true()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_ev = Eventually::new(
            interval.clone(),
            Box::new(atomic.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_global = Globally::new(
            interval.clone(),
            Box::new(atomic.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_or = Or::new(
            Box::new(Eventually::new(
                interval.clone(),
                Box::new(atomic.clone()),
                Some(RingBuffer::new()),
                Some(RingBuffer::new()),
                EvaluationMode::Eager,
            )),
            Box::new(Atomic::<bool>::new_true()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        println!("STL formula: {}", op.to_string());
        let inputs = get_signal_1();

        for input in inputs.clone() {
            let res_ev = op_ev.robustness(&input);
            println!("Input: {:?}, Output EV: {:?}", input, res_ev);
        }
        println!("\n");
        for input in inputs.clone() {
            let res = op.robustness(&input);
            println!("Input: {:?}, Output AND: {:?}", input, res);
        }
        println!("\n");
        for input in inputs.clone() {
            let res_or = op_or.robustness(&input);
            println!("Input: {:?}, Output OR: {:?}", input, res_or);
        }
        println!("\n");
        for input in inputs {
            let res_global = op_global.robustness(&input);
            println!("Input: {:?}, Output GLOBALLY: {:?}", input, res_global);
        }
    }

    #[test]
    fn test_2() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let atomic_g5 = Atomic::<bool>::new_greater_than(5.0);
        let atomic_g0 = Atomic::<bool>::new_greater_than(0.0);

        let mut op_global = Globally::new(
            interval.clone(),
            Box::new(atomic_g0.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_ev = Eventually::new(
            interval.clone(),
            Box::new(atomic_g5.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_and = And::new(
            Box::new(op_ev.clone()),
            Box::new(op_global.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        println!("STL formula: {}", op_global.to_string());
        let inputs = get_signal_1();

        for input in inputs.clone() {
            let res = op_global.robustness(&input);
            println!("Input: {:?}, Output GLOBALLY: {:?}", input, res);
        }
        println!("\n");
        for input in inputs.clone() {
            let res_ev = op_ev.robustness(&input);
            println!("Input: {:?}, Output EV: {:?}", input, res_ev);
        }
        println!("\n");
        for input in inputs {
            let res_and = op_and.robustness(&input);
            println!("Input: {:?}, Output AND: {:?}", input, res_and);
        }
    }

    #[test]
    fn test_3_until() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(3),
        };
        let atomic_g5 = Atomic::<bool>::new_greater_than(5.0);
        let atomic_g0 = Atomic::<bool>::new_greater_than(0.0);

        let mut op_until = Until::new(
            interval.clone(),
            Box::new(atomic_g0.clone()),
            Box::new(atomic_g5.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        println!("STL formula: {}", op_until.to_string());
        let inputs = get_signal_1();
        for input in inputs {
            let res_until = op_until.robustness(&input);
            println!("Input: {:?}, Output UNTIL: {:?}", input, res_until);
        }
    }

    #[test]
    fn test_4_until() {
        let interval_eventually = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let interval_until = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(6),
        };

        let mut eventually_g3 = Eventually::new(
            interval_eventually.clone(),
            Box::new(Atomic::<bool>::new_greater_than(3.0)),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let atomic_g5 = Atomic::<bool>::new_greater_than(5.0);

        let mut op_until = Until::new(
            interval_until.clone(),
            Box::new(eventually_g3.clone()),
            Box::new(atomic_g5),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let inputs = get_signal_2();

        println!("\n");
        println!("STL formula: {}", eventually_g3.to_string());
        for input in inputs.clone() {
            let res_ev = eventually_g3.robustness(&input);
            println!("Input: {:?}, \nOutput EV: {:?}", input, res_ev);
        }
        println!("\n");

        println!("STL formula: {}", op_until.to_string());
        for input in inputs {
            let res_until = op_until.robustness(&input);
            println!("Input: {:?}, \nOutput UNTIL: {:?}", input, res_until);
        }
    }
    #[test]
    fn test_5_until() {
        let interval_globally = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let interval_eventually = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let interval_until = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(6),
        };
        let mut globally_g0 = Globally::new(
            interval_globally.clone(),
            Box::new(Atomic::<bool>::new_greater_than(0.0)),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut eventually_g3 = Eventually::new(
            interval_eventually.clone(),
            Box::new(Atomic::<bool>::new_greater_than(3.0)),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let mut op_until = Until::new(
            interval_until.clone(),
            Box::new(globally_g0.clone()),
            Box::new(eventually_g3.clone()),
            Some(RingBuffer::new()),
            Some(RingBuffer::new()),
            EvaluationMode::Eager,
        );
        let inputs = get_signal_3();

        println!("\n");
        println!("STL formula: {}", globally_g0.to_string());
        for input in inputs.clone() {
            let res_global = globally_g0.robustness(&input);
            println!("Input: {:?}, \nOutput GLOBALLY: {:?}", input, res_global);
        }
        println!("\n");
        println!("STL formula: {}", eventually_g3.to_string());
        for input in inputs.clone() {
            let res_ev = eventually_g3.robustness(&input);
            println!("Input: {:?}, \nOutput EV: {:?}", input, res_ev);
        }
        println!("\n");
        println!("STL formula: {}", op_until.to_string());
        for input in inputs {
            let res_until = op_until.robustness(&input);
            println!("Input: {:?}, \nOutput UNTIL: {:?}", input, res_until);
        }
    }
}
