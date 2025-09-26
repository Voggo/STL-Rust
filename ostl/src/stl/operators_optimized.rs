use crate::ring_buffer::{RingBuffer, RingBufferTrait, Step};
use std::time::Duration;

// Time interval type
#[derive(Debug, Copy, Clone)]
pub struct TimeInterval {
    pub start: Duration,
    pub end: Duration,
}

// A generic representation of an STL formula.
pub trait StlOperator<T: Clone> {
    fn robustness(&self, step: Step<T>) -> f64;
    fn to_string(&self) -> String;
}

struct And<T> {
    left: Box<dyn StlOperator<T>>,
    right: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for And<T> {
    fn robustness(&self, step: Step<T>) -> f64 {
        return self
            .left
            .robustness(step.clone())
            .min(self.right.robustness(step.clone()));
    }
    fn to_string(&self) -> String {
        format!("({}) ∧ ({})", self.left.to_string(), self.right.to_string())
    }
}

struct Or<T> {
    left: Box<dyn StlOperator<T>>,
    right: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Or<T> {
    fn robustness(&self, step: Step<T>) -> f64 {
        return self
            .left
            .robustness(step.clone())
            .max(self.right.robustness(step.clone()));
    }
    fn to_string(&self) -> String {
        format!("({}) v ({})", self.left.to_string(), self.right.to_string())
    }
}

struct Not<T> {
    operand: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Not<T> {
    fn robustness(&self, step: Step<T>) -> f64 {
        return -self.operand.robustness(step);
    }
    fn to_string(&self) -> String {
        format!("¬({})", self.operand.to_string())
    }
}

struct Eventually<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    interval: TimeInterval,
    operand: Box<dyn StlOperator<T>>,
    cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Eventually<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: Step<T>) -> f64 {
        todo!("Implement robustness calculation for Eventually operator")
    }
    fn to_string(&self) -> String {
        format!(
            "F[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

struct Globally<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    interval: TimeInterval,
    operand: Box<dyn StlOperator<T>>,
    cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Globally<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: Step<T>) -> f64 {
        todo!("Implement robustness calculation for Always operator")
    }
    fn to_string(&self) -> String {
        format!(
            "G[{}, {}]({})",
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.operand.to_string()
        )
    }
}

struct Until<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    interval: TimeInterval,
    left: Box<dyn StlOperator<T>>,
    right: Box<dyn StlOperator<T>>,
    cache: C,
}

impl<T: Clone, C: RingBufferTrait<Value = f64>> StlOperator<T> for Until<T, C>
where
    C: RingBufferTrait<Value = f64>,
{
    fn robustness(&self, step: Step<T>) -> f64 {
        todo!("Implement robustness calculation for Until operator")
    }
    fn to_string(&self) -> String {
        format!(
            "({}) U[{}, {}] ({})",
            self.left.to_string(),
            self.interval.start.as_secs_f64(),
            self.interval.end.as_secs_f64(),
            self.right.to_string()
        )
    }
}

pub struct Implies<T> {
    antecedent: Box<dyn StlOperator<T>>,
    consequent: Box<dyn StlOperator<T>>,
}

impl<T: Clone> StlOperator<T> for Implies<T> {
    fn robustness(&self, step: Step<T>) -> f64 {
        return -self
            .antecedent
            .robustness(step.clone())
            .min(self.consequent.robustness(step.clone()));
    }
    fn to_string(&self) -> String {
        format!(
            "({}) -> ({})",
            self.antecedent.to_string(),
            self.consequent.to_string()
        )
    }
}

pub enum Atomic {
    LessThan(f64),
    GreaterThan(f64),
    True,
    False,
}

impl<T: Into<f64> + Clone> StlOperator<T> for Atomic {
    fn robustness(&self, step: Step<T>) -> f64 {
        let value = step.value.into();
        match self {
            Atomic::True => f64::INFINITY,
            Atomic::False => f64::NEG_INFINITY,
            Atomic::GreaterThan(c) => value - c,
            Atomic::LessThan(c) => c - value,
        }
    }
    fn to_string(&self) -> String {
        match self {
            Atomic::True => "True".to_string(),
            Atomic::False => "False".to_string(),
            Atomic::GreaterThan(val) => format!("x > {}", val),
            Atomic::LessThan(val) => format!("x < {}", val),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formula_to_string() {
        // Example usage of the STL operators

        let stl_formula: Implies<f64> = Implies {
            antecedent: Box::new(And {
                left: Box::new(Atomic::GreaterThan(5.0)),
                right: Box::new(Or {
                    left: Box::new(Not {
                        operand: Box::new(Atomic::LessThan(3.0)),
                    }),
                    right: Box::new(Eventually {
                        interval: TimeInterval {
                            start: Duration::from_secs(0),
                            end: Duration::from_secs(10),
                        },
                        operand: Box::new(Globally {
                            interval: TimeInterval {
                                start: Duration::from_secs(2),
                                end: Duration::from_secs(8),
                            },
                            operand: Box::new(Until {
                                interval: TimeInterval {
                                    start: Duration::from_secs(1),
                                    end: Duration::from_secs(5),
                                },
                                left: Box::new(Atomic::True),
                                right: Box::new(Atomic::False),
                                cache: RingBuffer::new(), // Replace with appropriate ring buffer implementation
                            }),
                            cache: RingBuffer::new(), // Replace with appropriate ring buffer implementation
                        }),
                        cache: RingBuffer::new(), // Replace with appropriate ring buffer implementation
                    }),
                }),
            }),
            consequent: Box::new(Atomic::LessThan(7.0)),
        };

        // println!("STL Formula: {}", stl_formula.to_string());
        assert_eq!(
            stl_formula.to_string(),
            "((x > 5) ∧ ((¬(x < 3)) v (F[0, 10](G[2, 8]((True) U[1, 5] (False)))))) -> (x < 7)"
        );
    }

    #[test]
    fn atomic_greater_than_robustness() {
        let atomic = Atomic::GreaterThan(10.0);
        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), 5.0);

        let step = Step {
            value: 8.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), -2.0);
    }

    #[test]
    fn atomic_less_than_robustness() {
        let atomic = Atomic::LessThan(10.0);
        let step = Step {
            value: 5.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), 5.0);

        let step = Step {
            value: 12.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), -2.0);
    }

    #[test]
    fn atomic_true_robustness() {
        let atomic = Atomic::True;
        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), f64::INFINITY);
    }

    #[test]
    fn atomic_false_robustness() {
        let atomic = Atomic::False;
        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(step), f64::NEG_INFINITY);
    }

    #[test]
    fn not_operator_robustness() {
        let not = Not {
            operand: Box::new(Atomic::GreaterThan(10.0)),
        };
        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(not.robustness(step), -5.0);
    }

    #[test]
    fn and_operator_robustness() {
        let and = And {
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(20.0)),
        };
        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(and.robustness(step), -5.0);
    }

    #[test]
    fn or_operator_robustness() {
        let or = Or {
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(5.0)),
        };
        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(or.robustness(step), 5.0);
    }

    #[test]
    fn implies_operator_robustness() {
        let implies = Implies {
            antecedent: Box::new(Atomic::GreaterThan(10.0)),
            consequent: Box::new(Atomic::LessThan(20.0)),
        };
        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(implies.robustness(step), -5.0);
    }
}
