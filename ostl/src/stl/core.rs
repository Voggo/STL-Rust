use crate::ring_buffer::{RingBufferTrait, Step};
use dyn_clone::{DynClone, clone_trait_object};
use std::fmt::Display;
use std::time::Duration;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// stloperator trait
pub trait StlOperatorTrait<T: Clone, Y>: DynClone + Display {
    // added DynClone for cloning trait objects
    fn robustness(&mut self, step: &Step<T>) -> Option<Y>;
}

clone_trait_object!(<T: Clone, Y> StlOperatorTrait<T, Y>);

pub trait LogicalOperatorTrait<T: Clone, Y> {
    fn left(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;
    fn right(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;

    fn robustness_with<F>(&mut self, step: &Step<T>, f: F) -> Option<Y>
    where
        F: FnOnce(Y, Y) -> Y,
        Y: Clone,
        T: Clone,
        Self: StlOperatorTrait<T, Y>,
    {
        self.left()
            .as_mut()
            .robustness(step)
            .zip(self.right().as_mut().robustness(step))
            .map(|(l, r)| f(l, r))
    }
}

// pub trait TemporalOperatorTrait<T, Y, C>
// where
//     T: Clone,
//     Y: Clone,
//     C: RingBufferTrait<Value = Option<Y>>,
// {
//     fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;
//     fn interval(&self) -> TimeInterval;
//     fn cache(&mut self) -> &mut C;

//     fn robustness_with<F>(&mut self, step: &Step<T>, initial: Y, f: F) -> Option<Y>
//     where
//         F: Fn(Y, Y) -> Y,
//     {
//         // Add the operand's latest robustness value to the cache
//         let robustness = self.operand().robustness(step);
//         self.cache().add_step(robustness, step.timestamp);

//         let t = step.timestamp.saturating_sub(self.interval().end);

//         // The time window for past robustness values is [t - end, t - start]
//         let lower_bound = t + self.interval().start;
//         let upper_bound = t + self.interval().end;

//         // We can only compute a result if the oldest data in our cache is older than
//         // or at least as old as the start of the time window.
//         if self.cache().is_empty()
//             || upper_bound - lower_bound
//                 <= self
//                     .cache()
//                     .get_back()
//                     .map_or(Duration::ZERO, |entry| entry.timestamp - t)
//         {
//             let robustness = self
//                 .cache()
//                 .iter()
//                 .filter(|entry| entry.timestamp >= lower_bound && entry.timestamp <= upper_bound)
//                 .filter_map(|entry| entry.value.clone()) // Get the f64 from Option<f64>
//                 .fold(initial, f);

//             Some(robustness)
//         } else {
//             None // Not enough historical data to compute robustness
//         }
//     }
// }

// Base Trait for all Temporal Operators (Minimal shared components)
pub trait TemporalOperatorBaseTrait<T, Y, C>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait,
{
    fn interval(&self) -> TimeInterval;
    fn cache(&mut self) -> &mut C;

    fn is_cache_sufficient(
        &mut self,
        lower_bound: Duration,
        upper_bound: Duration,
        t: Duration,
    ) -> bool {
        self.cache().is_empty()
            || upper_bound - lower_bound
                <= self
                    .cache()
                    .get_back()
                    .map_or(Duration::ZERO, |entry| entry.timestamp - t)
    }
}

// 2. Unary Temporal Trait (for Eventually and Globally)
// This trait exposes the single operand and adds the shared robustness helper.
pub trait UnaryTemporalOperatorTrait<T, Y, C>: TemporalOperatorBaseTrait<T, Y, C>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>>,
{
    fn operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;

    // Helper for unary aggregation (remains the same)
    fn robustness_unary_with<F>(&mut self, step: &Step<T>, initial: Y, f: F) -> Option<Y>
    where
        F: Fn(Y, Y) -> Y,
    {
        let robustness = self.operand().robustness(step);
        self.cache().add_step(robustness, step.timestamp);

        let t = step.timestamp.saturating_sub(self.interval().end);
        let lower_bound = t + self.interval().start;
        let upper_bound = t + self.interval().end;

        if self.is_cache_sufficient(lower_bound, upper_bound, t) {
            let robustness = self
                .cache()
                .iter()
                .filter(|entry| entry.timestamp >= lower_bound && entry.timestamp <= upper_bound)
                .filter_map(|entry| entry.value.clone())
                .fold(initial, f);

            Some(robustness)
        } else {
            None
        }
    }
}

// 3. Binary Temporal Trait (for Until)
// This trait exposes both left and right operands.
pub trait BinaryTemporalOperatorTrait<T, Y, C>: TemporalOperatorBaseTrait<T, Y, C>
where
    T: Clone,
    Y: Clone,
    C: RingBufferTrait<Value = Option<Y>>,
{
    fn left_operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;
    fn right_operand(&mut self) -> &mut Box<dyn StlOperatorTrait<T, Y>>;

    // NOTE: No shared helper here, as Until's logic is unique.
}
