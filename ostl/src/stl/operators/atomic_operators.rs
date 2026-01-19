use crate::ring_buffer::Step;
use crate::stl::core::{
    RobustnessSemantics, SignalIdentifier, StlOperatorTrait
};
use std::collections::HashSet;
use std::fmt::Display;
use std::time::Duration;


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
            Atomic::LessThan(signal_name, c, _) => write!(f, "{signal_name} < {c}"),
            Atomic::GreaterThan(signal_name, c, _) => write!(f, "{signal_name} > {c}"),
            Atomic::True(_) => write!(f, "True"),
            Atomic::False(_) => write!(f, "False"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring_buffer::Step;
    use crate::stl::core::StlOperatorTrait;
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
    fn atomic_signal_identifiers() {
        let mut atomic_gt = Atomic::<f64>::new_greater_than("x", 10.0);
        let ids_gt = atomic_gt.get_signal_identifiers();
        assert_eq!(ids_gt.len(), 1);
        assert!(ids_gt.contains("x"));  
        let mut atomic_lt = Atomic::<f64>::new_less_than("y", 5.0);
        let ids_lt = atomic_lt.get_signal_identifiers();
        assert_eq!(ids_lt.len(), 1);
        assert!(ids_lt.contains("y"));
        let mut atomic_true = Atomic::<f64>::new_true();
        let ids_true = atomic_true.get_signal_identifiers();
        assert_eq!(ids_true.len(), 0);
        let mut atomic_false = Atomic::<f64>::new_false();
        let ids_false = atomic_false.get_signal_identifiers();
        assert_eq!(ids_false.len(), 0);     
    }
}