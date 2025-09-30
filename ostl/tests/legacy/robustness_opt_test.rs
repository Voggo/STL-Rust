#[cfg(test)]
mod tests {
    use ostl::{
        ring_buffer::RingBuffer,
        stl::operators_optimized::{
            Atomic, And, Eventually, Globally, Implies, Not, Or, Step, TimeInterval, Until,
        },
    };
    use std::time::Duration;

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
        assert_eq!(atomic.robustness(&step), 5.0);

        let step = Step {
            value: 8.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(&step), -2.0);
    }

    #[test]
    fn atomic_less_than_robustness() {
        let atomic = Atomic::LessThan(10.0);
        let step = Step {
            value: 5.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(&step), 5.0);

        let step = Step {
            value: 12.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(&step), -2.0);
    }

    #[test]
    fn atomic_true_robustness() {
        let atomic = Atomic::True;
        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(&step), f64::INFINITY);
    }

    #[test]
    fn atomic_false_robustness() {
        let atomic = Atomic::False;
        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };
        assert_eq!(atomic.robustness(&step), f64::NEG_INFINITY);
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
        assert_eq!(not.robustness(&step), -5.0);
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
        assert_eq!(and.robustness(&step), 5.0);
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
        assert_eq!(or.robustness(&step), 5.0);
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
        assert_eq!(implies.robustness(&step), -5.0);
    }
}