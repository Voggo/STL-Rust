use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
    TimeInterval,
};
use std::collections::{BTreeSet, HashSet};
use std::fmt::{Debug, Display};
use std::time::Duration;

/// A unified binary processor that handles Strict, Eager, and Refinable (RoSI) semantics correctly.
fn process_binary<C, Y, F, const IS_EAGER: bool, const IS_ROSI: bool>(
    left_cache: &mut C,
    right_cache: &mut C,
    left_last_known: &mut Step<Option<Y>>,
    right_last_known: &mut Step<Option<Y>>,
    combine_op: F,
    short_circuit_val: Option<Y>,
) -> Vec<Step<Option<Y>>>
where
    C: RingBufferTrait<Value = Option<Y>>,
    Y: RobustnessSemantics + Copy + PartialEq + 'static,
    F: Fn(Y, Y) -> Y,
{
    let mut output_robustness = Vec::new();

    // CASE 1: Refinable Semantics (e.g., RoSI)
    // We simply iterate the intersection of timestamps currently available.
    if IS_ROSI {
        let mut l_iter = left_cache.iter();
        let mut r_iter = right_cache.iter();

        let mut l_curr = l_iter.next();
        let mut r_curr = r_iter.next();

        // We assume strict time alignment for RoSI based on standard semantics.
        // Iterate while both have data.
        while let (Some(l), Some(r)) = (l_curr, r_curr) {
            if l.timestamp == r.timestamp {
                // Timestamps align: Combine and emit
                let combined = l.value.zip(r.value).map(|(lv, rv)| combine_op(lv, rv));
                output_robustness.push(Step::new("output", combined, l.timestamp));

                l_curr = l_iter.next();
                r_curr = r_iter.next();
            } else if l.timestamp < r.timestamp {
                l_curr = l_iter.next();
            } else {
                r_curr = r_iter.next();
            }
        }

        return output_robustness;
    }

    // CASE 2: Non-Refinable Semantics (f64, bool)
    // We act as a consumer: pop data as we process it.
    loop {
        let l_head = left_cache.get_front();
        let r_head = right_cache.get_front();

        match (l_head, r_head) {
            // Both caches have data
            (Some(l), Some(r)) => {
                if l.timestamp == r.timestamp {
                    // 1. Timestamps align
                    let val = l.value.zip(r.value).map(|(lv, rv)| combine_op(lv, rv));
                    let ts = l.timestamp;

                    // Update state
                    if let Some(v) = l.value {
                        *left_last_known = Step::new("", Some(v), ts);
                    }
                    if let Some(v) = r.value {
                        *right_last_known = Step::new("", Some(v), ts);
                    }

                    output_robustness.push(Step::new("output", val, ts));

                    // Consume both
                    left_cache.pop_front();
                    right_cache.pop_front();
                } else if l.timestamp < r.timestamp {
                    if !IS_EAGER {
                        // In Strict, we cannot process mismatched timestamps.
                        // Since L < R, and we need L==R, we might need to wait for R to catch up,
                        // OR if we assume synchronized inputs, L is garbage.
                        // Standard STL usually waits.
                        break;
                    }
                    // 2. Left is earlier
                    let l_ts = l.timestamp;
                    let l_val = l.value;

                    // Eager Short-Circuit Check
                    if IS_EAGER {
                        // Interpolation (Piecewise Constant)
                        if let (Some(lv), Some(rl)) = (l_val, right_last_known.value) {
                            output_robustness.push(Step::new(
                                "output",
                                Some(combine_op(lv, rl)),
                                l_ts,
                            ));
                            *left_last_known = left_cache.pop_front().unwrap();
                            continue;
                        }
                    }

                    *left_last_known = left_cache.pop_front().unwrap();
                } else {
                    // 3. Right is earlier (Symmetric to Left)
                    let r_ts = r.timestamp;
                    let r_val = r.value;

                    if IS_EAGER && let (Some(rv), Some(ll)) = (r_val, left_last_known.value) {
                        output_robustness.push(Step::new("output", Some(combine_op(ll, rv)), r_ts));
                        *right_last_known = right_cache.pop_front().unwrap();
                        continue;
                    }

                    if !IS_EAGER {
                        break;
                    }

                    *right_last_known = right_cache.pop_front().unwrap();
                }
            }
            // Only Left has data
            (Some(l), None) => {
                if IS_EAGER {
                    let l_ts = l.timestamp;
                    let l_val = l.value;

                    // Check Short Circuit
                    if let (Some(sc), Some(lv)) = (short_circuit_val, l_val)
                        && lv == sc
                    {
                        output_robustness.push(Step::new("output", Some(sc), l_ts));
                        *left_last_known = left_cache.pop_front().unwrap();
                        continue;
                    }
                }
                break; // Wait for Right
            }
            // Only Right has data
            (None, Some(r)) => {
                if IS_EAGER {
                    let r_ts = r.timestamp;
                    let r_val = r.value;

                    if let (Some(sc), Some(rv)) = (short_circuit_val, r_val)
                        && rv == sc
                    {
                        output_robustness.push(Step::new("output", Some(sc), r_ts));
                        *right_last_known = right_cache.pop_front().unwrap();
                        continue;
                    }
                }
                break; // Wait for Left
            }
            (None, None) => break,
        }
    }

    output_robustness
}

#[derive(Clone)]
pub struct And<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    pub left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub left_cache: C,
    pub right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> And<T, C, Y, IS_EAGER, IS_ROSI> {
    pub fn new(
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
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for And<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static + Debug + Copy,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let check_relevance = |timestamp: Duration, last_time: Option<Duration>| -> bool {
            match last_time {
                Some(last) => {
                    if IS_ROSI {
                        timestamp >= last // Intervals: allow refinement of current step
                    } else {
                        timestamp > last // Bool/F64: strictly new data only
                    }
                }
                None => true,
            }
        };

        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.left.update(step);
            for update in left_updates {
                if check_relevance(update.timestamp, self.last_eval_time)
                    && !self.left_cache.update_step(update.clone())
                {
                    self.left_cache.add_step(update);
                }
            }
        }

        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.right.update(step);
            for update in right_updates {
                if check_relevance(update.timestamp, self.last_eval_time)
                    && !self.right_cache.update_step(update.clone())
                {
                    self.right_cache.add_step(update);
                }
            }
        }

        let mut output = process_binary::<C, Y, _, IS_EAGER, IS_ROSI>(
            &mut self.left_cache,
            &mut self.right_cache,
            &mut self.left_last_known,
            &mut self.right_last_known,
            Y::and,
            Some(Y::atomic_false()),
        );

        // Ensure we don't emit stale timestamps for non-refinable types.
        if !IS_ROSI && let Some(last_time) = self.last_eval_time {
            output.retain(|step| step.timestamp > last_time);
        }

        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        // Update last_eval_time based on strictness semantics
        if let Some(eval_time) = if IS_ROSI {
            // For intervals, we track the *start* of the batch because we might re-evaluate it
            output.first().map(|step| step.timestamp)
        } else {
            // For strict/bool, we track the *end* because everything before is finalized
            output.last().map(|step| step.timestamp)
        } {
            self.last_eval_time = Some(eval_time);
        }

        output
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for And<T, C, Y, IS_EAGER, IS_ROSI>
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

#[derive(Clone)]
pub struct Or<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    pub left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub left_cache: C,
    pub right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Or<T, C, Y, IS_EAGER, IS_ROSI> {
    pub fn new(
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
        }
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> StlOperatorTrait<T>
    for Or<T, C, Y, IS_EAGER, IS_ROSI>
where
    T: Clone + 'static,
    C: RingBufferTrait<Value = Option<Y>> + Clone + 'static,
    Y: RobustnessSemantics + 'static + Debug + Copy,
{
    type Output = Y;

    fn get_max_lookahead(&self) -> Duration {
        let left_lookahead = self.left.get_max_lookahead();
        let right_lookahead = self.right.get_max_lookahead();
        left_lookahead.max(right_lookahead)
    }

    fn update(&mut self, step: &Step<T>) -> Vec<Step<Option<Self::Output>>> {
        let check_relevance = |timestamp: Duration, last_time: Option<Duration>| -> bool {
            match last_time {
                Some(last) => {
                    if IS_ROSI {
                        timestamp >= last // Intervals: allow refinement of current step
                    } else {
                        timestamp > last // Bool/F64: strictly new data only
                    }
                }
                None => true,
            }
        };

        if self.left_signals_set.contains(&step.signal) || self.left_signals_set.is_empty() {
            let left_updates = self.left.update(step);
            for update in left_updates {
                if check_relevance(update.timestamp, self.last_eval_time)
                    && !self.left_cache.update_step(update.clone())
                {
                    self.left_cache.add_step(update);
                }
            }
        }

        if self.right_signals_set.contains(&step.signal) || self.right_signals_set.is_empty() {
            let right_updates = self.right.update(step);
            for update in right_updates {
                if check_relevance(update.timestamp, self.last_eval_time)
                    && !self.right_cache.update_step(update.clone())
                {
                    self.right_cache.add_step(update);
                }
            }
        }

        let mut output = process_binary::<C, Y, _, IS_EAGER, IS_ROSI>(
            &mut self.left_cache,
            &mut self.right_cache,
            &mut self.left_last_known,
            &mut self.right_last_known,
            Y::or,
            Some(Y::atomic_true()),
        );

        // Ensure we don't emit stale timestamps for non-refinable types.
        if !IS_ROSI && let Some(last_time) = self.last_eval_time {
            output.retain(|step| step.timestamp > last_time);
        }

        let lookahead = self.get_max_lookahead();
        self.left_cache.prune(lookahead);
        self.right_cache.prune(lookahead);

        // Update last_eval_time based on strictness semantics
        if let Some(eval_time) = if IS_ROSI {
            // For intervals, we track the *start* of the batch because we might re-evaluate it
            output.first().map(|step| step.timestamp)
        } else {
            // For strict/bool, we track the *end* because everything before is finalized
            output.last().map(|step| step.timestamp)
        } {
            self.last_eval_time = Some(eval_time);
        }

        output
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> SignalIdentifier
    for Or<T, C, Y, IS_EAGER, IS_ROSI>
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
pub struct Eventually<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    pub cache: C,
    pub eval_buffer: BTreeSet<Duration>,
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
                    self.cache.add_step(sub_step);
                }
            }
        } else {
            for sub_step in sub_robustness_vec {
                self.eval_buffer.insert(sub_step.timestamp);
                self.cache.add_step(sub_step); // sub_step is moved
            }
        }

        let mut tasks_to_remove = Vec::new();
        let current_time = step.timestamp;

        // 2. Process the evaluation buffer
        for &t_eval in self.eval_buffer.iter() {
            let window_start = t_eval + self.interval.start;
            let window_end = t_eval + self.interval.end;

            // Use `skip_while` and `take_while` to iterate only over
            // the relevant part of the cache, not the whole thing.
            let windowed_max_value = self
                .cache
                .iter()
                .skip_while(|entry| entry.timestamp < window_start)
                .take_while(|entry| entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::eventually_identity(), Y::or); // Use the identity from your code

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
        self.cache.prune(self.interval.end);
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
    pub interval: TimeInterval,
    pub operand: Box<dyn StlOperatorAndSignalIdentifier<T, Y> + 'static>,
    pub cache: C,
    pub eval_buffer: BTreeSet<Duration>,
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
                    self.cache.add_step(sub_step);
                }
            }
        } else {
            for sub_step in sub_robustness_vec {
                self.eval_buffer.insert(sub_step.timestamp);
                self.cache.add_step(sub_step);
            }
        }

        let mut tasks_to_remove = Vec::new();
        let current_time = step.timestamp;

        // 2. Process the evaluation buffer
        for &t_eval in self.eval_buffer.iter() {
            let window_start = t_eval + self.interval.start;
            let window_end = t_eval + self.interval.end;

            // Use `skip_while` and `take_while` to iterate only over
            // the relevant part of the cache, not the whole thing.
            let windowed_min_value = self
                .cache
                .iter()
                .skip_while(|entry| entry.timestamp < window_start)
                .take_while(|entry| entry.timestamp <= window_end)
                .filter_map(|entry| entry.value.clone())
                .fold(Y::globally_identity(), Y::and); // Use the identity from your code

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
        self.cache.prune(self.interval.end);
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

#[derive(Clone)]
pub struct Until<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
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
            let window_start_t_eval = t_eval + self.interval.start;
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
                    .take_while(|s| s.timestamp <= t_prime) // t'' < t' : strong until - t'' <= t' weak until
                    .filter_map(|s| s.value.clone())
                    .fold(Y::globally_identity(), Y::and);

                // if no data in in left_cache for >t_prime, we need to and with unknown
                if let Some(last_left_step) = self
                    .left_cache
                    .iter()
                    .skip_while(|s| s.timestamp < t_eval) // t'' >= t_eval+a
                    .take_while(|s| s.timestamp <= t_prime) // t'' < t' : strong until - t'' <= t' weak until
                    .last()
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
                // Case 2: Eager short-circuit (Satisfaction). Found "true" before window closed.
                final_value = Some(max_robustness); // which is Y::atomic_true()
                remove_task = true;
            } else if IS_EAGER && falsified {
                // **FIX 2: Add Eager short-circuit (Falsification).**
                final_value = Some(max_robustness); // which is Y::atomic_false()
                remove_task = true;
            } else if IS_ROSI {
                // Case 3: Intermediate ROSI. Window is still open, no short-circuit.
                // let intermediate_value = Y::or(max_robustness, Y::unknown());
                final_value = Some(max_robustness);
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

#[derive(Clone)]
pub enum Atomic<Y> {
    LessThan(&'static str, f64, std::marker::PhantomData<Y>),
    GreaterThan(&'static str, f64, std::marker::PhantomData<Y>),
    LessThanSignal(&'static str, &'static str, std::marker::PhantomData<Y>),
    GreaterThanSignal(&'static str, &'static str, std::marker::PhantomData<Y>),
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
    pub fn new_less_than_signal(
        signal_name: &'static str,
        signal_to_compare: &'static str,
    ) -> Self {
        Atomic::LessThanSignal(signal_name, signal_to_compare, std::marker::PhantomData)
    }
    pub fn new_greater_than_signal(
        signal_name: &'static str,
        signal_to_compare: &'static str,
    ) -> Self {
        Atomic::GreaterThanSignal(signal_name, signal_to_compare, std::marker::PhantomData)
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
            Atomic::GreaterThanSignal(_signal_name, signal_to_compare, _) => {
                Y::atomic_greater_than_signal(value, signal_to_compare)
            }
            Atomic::LessThanSignal(_signal_name, signal_to_compare, _) => {
                Y::atomic_less_than_signal(value, signal_to_compare)
            }
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
            Atomic::LessThan(signal_name, c, _) => write!(f, "{signal_name} < {c}"),
            Atomic::GreaterThan(signal_name, c, _) => write!(f, "{signal_name} > {c}"),
            Atomic::True(_) => write!(f, "True"),
            Atomic::False(_) => write!(f, "False"),
        }
    }
}
impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for And<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) ∧ ({})", self.left, self.right)
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Until<T, Y, C, IS_EAGER, IS_ROSI>
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

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Or<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) v ({})", self.left, self.right)
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
impl<T, Y> Display for Not<T, Y> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "¬({})", self.operand)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::{RingBuffer, Step};
    use crate::stl::core::{StlOperatorTrait, TimeInterval};
    use pretty_assertions::assert_eq;
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

        let mut expected_gt: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        expected_gt.insert("x");
        let mut expected_lt: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        expected_lt.insert("y");
        let expected_empty: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();

        assert_eq!(ids_gt, expected_gt);
        assert_eq!(ids_lt, expected_lt);
        assert_eq!(ids_true, expected_empty);
        assert_eq!(ids_false, expected_empty);

        // Composite: And(x>10, y<5)
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            Box::new(Atomic::<f64>::new_less_than("y", 5.0)),
            None,
            None,
        );
        let ids_and = and.get_signal_identifiers();
        let mut expected_and: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        expected_and.insert("x");
        expected_and.insert("y");
        assert_eq!(ids_and, expected_and);

        // Composite with constant: And(True, x>10) -> should report only 'x'
        let mut and2 = And::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(Atomic::<f64>::new_true()),
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            None,
            None,
        );
        let ids_and2 = and2.get_signal_identifiers();
        let mut expected_and2: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        expected_and2.insert("x");
        assert_eq!(ids_and2, expected_and2);
    }

    #[test]
    fn get_signal_identifiers_nested() {
        // And(x>10, U(y>5, z<0))
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(Atomic::<f64>::new_greater_than("x", 10.0)),
            Box::new(
                Until::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
                    TimeInterval {
                        start: Duration::from_secs(0),
                        end: Duration::from_secs(5),
                    },
                    Box::new(Atomic::<f64>::new_greater_than("y", 5.0)),
                    Box::new(Atomic::<f64>::new_less_than("z", 0.0)),
                    None,
                    None,
                ),
            ),
            None,
            None,
        );
        let mut expected_and: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        let mut expected_and_left: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
        let mut expected_and_right: std::collections::HashSet<&'static str> =
            std::collections::HashSet::new();
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
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
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
        let mut or = Or::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
        );
        or.get_signal_identifiers();

        let step = Step::new("x", 15.0, Duration::from_secs(5));
        let robustness = or.update(&step);
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
}
