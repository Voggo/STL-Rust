use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{RobustnessSemantics, StlOperatorTrait, TimeInterval};
use std::fmt::Display;
use std::time::Duration;

fn process_binary_operator_caches<C, Y, F>(
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
}

impl<T, C, Y> And<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.left.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.right.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
        }
        let output = process_binary_operator_caches(
            &mut self.left_cache,
            &mut self.right_cache,
            &mut self.left_last_known,
            &mut self.right_last_known,
            Y::and,            // The only difference!
            Y::atomic_false(), // Default atomic value for 'and'
        );

        let max_timestamp = self
            .right_last_known
            .timestamp
            .max(self.left_last_known.timestamp);
        self.left_cache.prune(max_timestamp);
        self.right_cache.prune(max_timestamp);

        output
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth)
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
}

impl<T, C, Y> Or<T, C, Y> {
    pub fn new(
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.left.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.right.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
        }
        let output = process_binary_operator_caches(
            &mut self.left_cache,
            &mut self.right_cache,
            &mut self.left_last_known,
            &mut self.right_last_known,
            Y::or,            // The only difference!
            Y::atomic_true(), // Default atomic value for 'or'
        );
        let max_timestamp = self
            .right_last_known
            .timestamp
            .max(self.left_last_known.timestamp);
        self.left_cache.prune(max_timestamp);
        self.right_cache.prune(max_timestamp);
        output
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth)
    }
}

#[derive(Clone)]
pub struct Not<T, Y> {
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    // question: why no cache here ?
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

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth()
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
}

impl<T, C, Y> Implies<T, C, Y> {
    pub fn new(
        antecedent: Box<dyn StlOperatorTrait<T, Output = Y>>,
        consequent: Box<dyn StlOperatorTrait<T, Output = Y>>,
        left_cache: Option<C>,
        right_cache: Option<C>,
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
        }
    }
}

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

    fn robustness(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let left_updates = self.antecedent.robustness(step);
        for update in left_updates {
            self.left_cache.add_step(update);
        }

        let right_updates = self.consequent.robustness(step);
        for update in right_updates {
            self.right_cache.add_step(update);
        }

        process_binary_operator_caches(
            &mut self.left_cache,
            &mut self.right_cache,
            &mut self.left_last_known,
            &mut self.right_last_known,
            Y::implies,        // The only difference!
            Y::atomic_false(), // Default atomic value for 'implies'
        )
    }

    fn get_temporal_depth(&self) -> usize {
        let antecedent_depth = self.antecedent.get_temporal_depth();
        let consequent_depth = self.consequent.get_temporal_depth();
        antecedent_depth.max(consequent_depth)
    }
}

#[derive(Clone)]
pub struct Eventually<T, C, Y> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
    pub cache: C,
    pub eval_buffer: C,
}

impl<T, C, Y> Eventually<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
        cache: Option<C>,
        eval_buffer: Option<C>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Eventually {
            interval,
            operand,
            cache: cache.unwrap_or_else(|| C::new()),
            eval_buffer: eval_buffer.unwrap_or_else(|| C::new()),
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

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        // Use the identity and combining function from the trait
        self.robustness_unary_with(step, Y::eventually_identity(), Y::or)
    }

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth() + self.interval.end.as_secs() as usize
    }
}

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
            if max_in_window == Y::atomic_true() {
                self.eval_buffer.pop_front(); // Task is done
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            if step.timestamp >= window_end {
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
}

impl<T, C, Y> Globally<T, Y, C> {
    pub fn new(
        interval: TimeInterval,
        operand: Box<dyn StlOperatorTrait<T, Output = Y>>,
        cache: Option<C>,
        eval_buffer: Option<C>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Globally {
            interval,
            operand,
            cache: cache.unwrap_or_else(|| C::new()),
            eval_buffer: eval_buffer.unwrap_or_else(|| C::new()),
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

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        self.robustness_unary_with(step, Y::globally_identity(), Y::and)
    }

    fn get_temporal_depth(&self) -> usize {
        self.operand.get_temporal_depth() + self.interval.end.as_secs() as usize
    }
}

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
            if max_in_window == Y::atomic_false() {
                self.eval_buffer.pop_front();
                output_robustness.push(Step::new(Some(max_in_window), t_eval));
                continue; // Move to the next task
            }

            // B. Check for normal completion: if the full time window has passed.
            if step.timestamp >= window_end {
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
    pub cache: C,
    pub eval_buffer: C,
}

impl<T, C, Y> Until<T, C, Y> {
    pub fn new(
        interval: TimeInterval,
        left: Box<dyn StlOperatorTrait<T, Output = Y>>,
        right: Box<dyn StlOperatorTrait<T, Output = Y>>,
        cache: Option<C>,
        eval_buffer: Option<C>,
    ) -> Self
    where
        T: Clone + 'static,
        C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
        Y: RobustnessSemantics + 'static,
    {
        Until {
            interval,
            left,
            right,
            cache: cache.unwrap_or_else(|| C::new()),
            eval_buffer: eval_buffer.unwrap_or_else(|| C::new()),
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

    fn robustness(&mut self, step: &Step<T>) -> Option<Self::Output> {
        let right_robustness = self.right.robustness(step)?;
        self.cache
            .add_step(self.left.robustness(step), step.timestamp);

        // The window of interest for the left operand's past robustness values
        let t = step.timestamp.saturating_sub(self.interval.end);
        let lower_bound_t_prime = t + self.interval.start;
        let upper_bound_t_prime = t + self.interval.end;

        // Ensure we have enough data to evaluate the window
        if self.is_cache_sufficient(lower_bound_t_prime, upper_bound_t_prime, t) {
            let max_robustness = self
                .cache
                .iter()
                .filter(|entry| {
                    entry.timestamp >= lower_bound_t_prime && entry.timestamp <= upper_bound_t_prime
                })
                .map(|entry| {
                    let t_prime = entry.timestamp;
                    let min_left_robustness = self
                        .cache
                        .iter()
                        .filter(|e| e.timestamp >= lower_bound_t_prime && e.timestamp <= t_prime)
                        .filter_map(|e| e.value.clone())
                        .fold(Y::globally_identity(), Y::and);

                    Y::and(right_robustness.clone(), min_left_robustness) // OBS: Using clone() here !!!!!!!!!!!!!!!!!!!!!! Should maybe be changed
                })
                .fold(Y::eventually_identity(), Y::or);

            Some(max_robustness)
        } else {
            None // Not enough data to evaluate
        }
    }

    fn get_temporal_depth(&self) -> usize {
        let left_depth = self.left.get_temporal_depth();
        let right_depth = self.right.get_temporal_depth();
        left_depth.max(right_depth) + self.interval.end.as_secs() as usize
    }
}
impl<T, C, Y> TemporalOperatorBaseTrait<T, C> for Until<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn interval(&self) -> TimeInterval {
        self.interval
    }

    fn cache(&mut self) -> &mut C {
        &mut self.cache
    }
}

impl<T, C, Y> BinaryTemporalOperatorTrait<T, C> for Until<T, C, Y>
where
    T: Clone + 'static,
    Y: Clone + RobustnessSemantics + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
{
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.left
    }

    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Output = Self::Output>> {
        &mut self.right
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
    fn get_temporal_depth(&self) -> usize {
        0
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
    fn get_signal() -> Vec<Step<f64>> {
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

    #[test]
    fn test_1() {
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };
        let atomic = Atomic::<bool>::new_greater_than(5.0);

        let mut op = And {
            left: Box::new(Eventually {
                interval: interval.clone(),
                operand: Box::new(atomic.clone()),
                cache: RingBuffer::new(),
                eval_buffer: RingBuffer::new(),
            }),
            right: Box::new(Atomic::<bool>::new_true()),
            left_cache: RingBuffer::new(),
            right_cache: RingBuffer::new(),
            left_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
            right_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
        };
        let mut op_ev = Eventually {
            interval: interval.clone(),
            operand: Box::new(atomic.clone()),
            cache: RingBuffer::new(),
            eval_buffer: RingBuffer::new(),
        };
        let mut op_global = Globally {
            interval: interval.clone(),
            operand: Box::new(atomic.clone()),
            cache: RingBuffer::new(),
            eval_buffer: RingBuffer::new(),
        };
        let mut op_or = Or {
            left: Box::new(Eventually {
                interval: interval.clone(),
                operand: Box::new(atomic.clone()),
                cache: RingBuffer::new(),
                eval_buffer: RingBuffer::new(),
            }),
            right: Box::new(Atomic::<bool>::new_true()),
            left_cache: RingBuffer::new(),
            right_cache: RingBuffer::new(),
            left_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
            right_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
        };

        let inputs = get_signal();

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

        let mut op_global = Globally {
            interval: interval.clone(),
            operand: Box::new(atomic_g0.clone()),
            cache: RingBuffer::new(),
            eval_buffer: RingBuffer::new(),
        };
        let mut op_ev = Eventually {
            interval: interval.clone(),
            operand: Box::new(atomic_g5.clone()),
            cache: RingBuffer::new(),
            eval_buffer: RingBuffer::new(),
        };
        let mut op_and = And {
            left: Box::new(op_ev.clone()),
            right: Box::new(op_global.clone()),
            left_cache: RingBuffer::new(),
            right_cache: RingBuffer::new(),
            left_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
            right_last_known: Step {
                value: None,
                timestamp: Duration::from_secs(0),
            },
        };

        let inputs = get_signal();

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
}
