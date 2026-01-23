use crate::ring_buffer::{RingBufferTrait, Step, guarded_prune};
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorAndSignalIdentifier, StlOperatorTrait,
};
use std::collections::HashSet;
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
    if IS_ROSI {
        let mut l_iter = left_cache.iter();
        let mut r_iter = right_cache.iter();

        let mut l_curr = l_iter.next();
        let mut r_curr = r_iter.next();

        while let (Some(l), Some(r)) = (l_curr, r_curr) {
            if l.timestamp == r.timestamp {
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

                    if let Some(v) = l.value {
                        *left_last_known = Step::new("", Some(v), ts);
                    }
                    if let Some(v) = r.value {
                        *right_last_known = Step::new("", Some(v), ts);
                    }

                    output_robustness.push(Step::new("output", val, ts));

                    left_cache.pop_front();
                    right_cache.pop_front();
                } else if l.timestamp < r.timestamp {
                    // 2. Left is earlier
                    let l_ts = l.timestamp;
                    let l_val = l.value;

                    // Eager Short-Circuit Check
                    if IS_EAGER {
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

                    // In Strict mode (and Eager fall-through), we discard the lagging step
                    // because it can never be matched with a future step from Right.
                    *left_last_known = left_cache.pop_front().unwrap();
                } else {
                    // 3. Right is earlier
                    let r_ts = r.timestamp;
                    let r_val = r.value;

                    if IS_EAGER && let (Some(rv), Some(ll)) = (r_val, left_last_known.value) {
                        output_robustness.push(Step::new("output", Some(combine_op(ll, rv)), r_ts));
                        *right_last_known = right_cache.pop_front().unwrap();
                        continue;
                    }

                    // Discard lagging step
                    *right_last_known = right_cache.pop_front().unwrap();
                }
            }
            // Only Left has data - we must wait for Right to potentially match or exceed Left's timestamp
            (Some(l), None) => {
                if IS_EAGER {
                    let l_ts = l.timestamp;
                    let l_val = l.value;
                    if let (Some(sc), Some(lv)) = (short_circuit_val, l_val)
                        && lv == sc
                    {
                        output_robustness.push(Step::new("output", Some(sc), l_ts));
                        *left_last_known = left_cache.pop_front().unwrap();
                        continue;
                    }
                }
                break;
            }
            // Only Right has data - we must wait for Left
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
                break;
            }
            (None, None) => break,
        }
    }

    output_robustness
}

#[derive(Clone)]
pub struct And<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> {
    left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    left_cache: C,
    right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    max_lookahead: Duration,
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
        let max_lookahead = left.get_max_lookahead().max(right.get_max_lookahead());
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
            max_lookahead,
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
        self.max_lookahead
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
        let protected_ts = self
            .left_cache
            .get_front()
            .into_iter()
            .chain(self.right_cache.get_front())
            .map(|s| s.timestamp)
            .min()
            .unwrap_or(Duration::ZERO);

        guarded_prune(&mut self.left_cache, lookahead, protected_ts);
        guarded_prune(&mut self.right_cache, lookahead, protected_ts);

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
    left: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    right: Box<dyn StlOperatorAndSignalIdentifier<T, Y>>,
    left_cache: C,
    right_cache: C,
    last_eval_time: Option<Duration>,
    left_last_known: Step<Option<Y>>,
    right_last_known: Step<Option<Y>>,
    left_signals_set: HashSet<&'static str>,
    right_signals_set: HashSet<&'static str>,
    max_lookahead: Duration,
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
        let max_lookahead = left.get_max_lookahead().max(right.get_max_lookahead());
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
            max_lookahead,
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
        self.max_lookahead
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

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for And<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) ∧ ({})", self.left, self.right)
    }
}

impl<T, C, Y, const IS_EAGER: bool, const IS_ROSI: bool> Display
    for Or<T, C, Y, IS_EAGER, IS_ROSI>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}) v ({})", self.left, self.right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::{RingBuffer, Step};
    use crate::stl::core::StlOperatorTrait;
    use crate::stl::operators::atomic_operators::Atomic;
    use pretty_assertions::assert_eq;
    use std::time::Duration;

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

    #[test]
    fn binary_operators_signal_identifiers() {
        let atomic1 = Atomic::<f64>::new_greater_than("x", 10.0);
        let atomic2 = Atomic::<f64>::new_less_than("y", 5.0);
        let mut and = And::<f64, RingBuffer<Option<f64>>, f64, false, false>::new(
            Box::new(atomic1),
            Box::new(atomic2),
            None,
            None,
        );
        let ids = and.get_signal_identifiers();
        let expected: HashSet<&'static str> = ["x", "y"].iter().cloned().collect();
        assert_eq!(ids, expected);
    }
}
