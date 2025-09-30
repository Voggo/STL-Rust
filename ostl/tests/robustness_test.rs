#[cfg(test)]
mod tests {
    use ostl::ring_buffer::{RingBuffer, Step};
    use ostl::stl::{
        core::{StlOperatorTrait, TimeInterval},
        robustness_cached::{And, Atomic, Eventually, Globally, Implies, Not, Or, Until},
        robustness_naive::{StlFormula, StlOperator},
    };
    use std::time::Duration;

    fn run_robustness_test<T: Clone>(
        mut formula_naive: Box<dyn StlOperatorTrait<T>>,
        mut formula_opt: Box<dyn StlOperatorTrait<T>>,
        step: &Step<T>,
        expected: Option<f64>,
    ) {
        let robustness_naive = formula_naive.robustness(step);
        let robustness_opt = formula_opt.robustness(step);

        assert_eq!(
            robustness_naive, expected,
            "Naive implementation failed {:?} != {:?}",
            robustness_naive, expected
        );
        assert_eq!(
            robustness_opt, expected,
            "Optimized implementation failed {:?} != {:?}",
            robustness_opt, expected
        );
        assert_eq!(
            robustness_naive, robustness_opt,
            "Mismatch between naive and optimized implementations {:?} != {:?}",
            robustness_naive, robustness_opt
        );
    }

    #[test]
    fn formula_to_string() {
        // Example usage of the STL operators
        let stl_formula_opt: Implies<f64> = Implies {
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
        let stl_operator_naive = StlOperator::Implies(
            Box::new(StlOperator::And(
                Box::new(StlOperator::GreaterThan(5.0)),
                Box::new(StlOperator::Or(
                    Box::new(StlOperator::Not(Box::new(StlOperator::LessThan(3.0)))),
                    Box::new(StlOperator::Eventually(
                        TimeInterval {
                            start: Duration::from_secs(0),
                            end: Duration::from_secs(10),
                        },
                        Box::new(StlOperator::Globally(
                            TimeInterval {
                                start: Duration::from_secs(2),
                                end: Duration::from_secs(8),
                            },
                            Box::new(StlOperator::Until(
                                TimeInterval {
                                    start: Duration::from_secs(1),
                                    end: Duration::from_secs(5),
                                },
                                Box::new(StlOperator::True),
                                Box::new(StlOperator::False),
                            )),
                        )),
                    )),
                )),
            )),
            Box::new(StlOperator::LessThan(7.0)),
        );

        let stl_formula_naive = StlFormula {
            formula: stl_operator_naive,
            signal: RingBuffer::<f64>::new(),
        };

        // println!("STL Formula: {}", stl_formula.to_string());
        assert_eq!(
            stl_formula_opt.to_string(),
            "((x > 5) ∧ ((¬(x < 3)) v (F[0, 10](G[2, 8]((True) U[1, 5] (False)))))) -> (x < 7)"
        );
        assert_eq!(
            stl_formula_naive.to_string(),
            "((x > 5) ∧ ((¬(x < 3)) v (F[0, 10](G[2, 8]((True) U[1, 5] (False)))))) -> (x < 7)"
        );
        assert_eq!(stl_formula_opt.to_string(), stl_formula_naive.to_string());
    }

    #[test]
    fn atomic_greater_than_robustness() {
        let atomic_opt = Atomic::GreaterThan(10.0);
        let atomic_naive = StlFormula {
            formula: StlOperator::GreaterThan(10.0),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive.clone()),
            Box::new(atomic_opt.clone()),
            &step,
            Some(5.0),
        );

        let step = Step {
            value: 8.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive.clone()),
            Box::new(atomic_opt.clone()),
            &step,
            Some(-2.0),
        );
    }

    #[test]
    fn atomic_less_than_robustness() {
        let atomic_opt = Atomic::LessThan(10.0);
        let atomic_naive = StlFormula {
            formula: StlOperator::LessThan(10.0),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 5.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive.clone()),
            Box::new(atomic_opt.clone()),
            &step,
            Some(5.0),
        );

        let step = Step {
            value: 12.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive.clone()),
            Box::new(atomic_opt.clone()),
            &step,
            Some(-2.0),
        );
    }

    #[test]
    fn atomic_true_robustness() {
        let atomic_opt = Atomic::True;
        let atomic_naive = StlFormula {
            formula: StlOperator::True,
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive),
            Box::new(atomic_opt),
            &step,
            Some(f64::INFINITY),
        );
    }

    #[test]
    fn atomic_false_robustness() {
        let atomic_opt = Atomic::False;
        let atomic_naive = StlFormula {
            formula: StlOperator::False,
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(atomic_naive),
            Box::new(atomic_opt),
            &step,
            Some(f64::NEG_INFINITY),
        );
    }

    #[test]
    fn not_operator_robustness() {
        let not_opt = Not {
            operand: Box::new(Atomic::GreaterThan(10.0)),
        };
        let not_naive = StlFormula {
            formula: StlOperator::Not(Box::new(StlOperator::GreaterThan(10.0))),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(Box::new(not_naive), Box::new(not_opt), &step, Some(-5.0));
    }

    #[test]
    fn and_operator_robustness() {
        let and_opt = And {
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(20.0)),
        };
        let and_naive = StlFormula {
            formula: StlOperator::And(
                Box::new(StlOperator::GreaterThan(10.0)),
                Box::new(StlOperator::LessThan(20.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(Box::new(and_naive), Box::new(and_opt), &step, Some(5.0));
    }

    #[test]
    fn or_operator_robustness() {
        let or_opt = Or {
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(5.0)),
        };
        let or_naive = StlFormula {
            formula: StlOperator::Or(
                Box::new(StlOperator::GreaterThan(10.0)),
                Box::new(StlOperator::LessThan(5.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(Box::new(or_naive), Box::new(or_opt), &step, Some(5.0));
    }

    #[test]
    fn implies_operator_robustness() {
        let implies_opt = Implies {
            antecedent: Box::new(Atomic::GreaterThan(10.0)),
            consequent: Box::new(Atomic::LessThan(20.0)),
        };
        let implies_naive = StlFormula {
            formula: StlOperator::Implies(
                Box::new(StlOperator::GreaterThan(10.0)),
                Box::new(StlOperator::LessThan(20.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(implies_naive),
            Box::new(implies_opt),
            &step,
            Some(5.0),
        );
    }

    #[test]
    fn or_and_law_robustness() {
        //tests defined as p v q  := ¬(¬p ∧ ¬q)
        let mut formula_or = Or {
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(20.0)),
        };

        let mut formula_and = Not {
            operand: Box::new(And {
                left: Box::new(Not {
                    operand: Box::new(Atomic::GreaterThan(10.0)),
                }),
                right: Box::new(Not {
                    operand: Box::new(Atomic::LessThan(20.0)),
                }),
            }),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        assert_eq!(formula_or.robustness(&step), formula_and.robustness(&step));
    }

    #[test]
    fn implies_law_robustness() {
        //tests defined as p -> q  := ¬p v q
        let mut formula_implies = Implies {
            antecedent: Box::new(Atomic::GreaterThan(10.0)),
            consequent: Box::new(Atomic::LessThan(20.0)),
        };

        let mut formula_or = Or {
            left: Box::new(Not {
                operand: Box::new(Atomic::GreaterThan(10.0)),
            }),
            right: Box::new(Atomic::LessThan(20.0)),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        assert_eq!(
            formula_implies.robustness(&step),
            formula_or.robustness(&step)
        );
    }

    #[test]
    fn eventually_law_robustness() {
        //tests defined as F[a,b] p  := True U[a,b] p

        let mut formula_eventually = Eventually {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            operand: Box::new(Atomic::GreaterThan(10.0)),
            cache: RingBuffer::new(),
        };

        let mut formula_until = Until {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            left: Box::new(Atomic::True),
            right: Box::new(Atomic::GreaterThan(10.0)),
            cache: RingBuffer::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        assert_eq!(
            formula_eventually.robustness(&step),
            formula_until.robustness(&step)
        );
    }

    #[test]
    fn globally_law_robustness() {
        //tests defined as G[a,b] p  := ¬F[a,b] ¬p
        let atomic_greater_than = Atomic::GreaterThan(10.0);
        let atomic_greater_than_naive = StlOperator::GreaterThan(10.0);

        let mut formula_globally_opt = Globally {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            operand: Box::new(atomic_greater_than.clone()),
            cache: RingBuffer::new(),
        };

        let formula_globally_naive = StlFormula {
            formula: StlOperator::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(atomic_greater_than_naive.clone()),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let mut formula_eventually_opt = Eventually {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            operand: Box::new(Not {
                operand: Box::new(atomic_greater_than.clone()),
            }),
            cache: RingBuffer::new(),
        };

        let formula_eventually_naive = StlFormula {
            formula: StlOperator::Not(Box::new(StlOperator::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(StlOperator::Not(Box::new(
                    atomic_greater_than_naive.clone(),
                ))),
            ))),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(formula_globally_naive.clone()),
            Box::new(formula_globally_opt.clone()),
            &step,
            Some(5.0),
        );

        run_robustness_test(
            Box::new(formula_eventually_naive.clone()),
            Box::new(formula_eventually_opt.clone()),
            &step,
            Some(-5.0),
        );

        assert_eq!(
            formula_globally_opt.robustness(&step),
            formula_eventually_opt.robustness(&step)
        );

        assert_eq!(
            formula_globally_opt.robustness(&step),
            formula_eventually_opt.robustness(&step)
        );
    }

    #[test]
    fn eventually_operator_robustness() {
        let mut eventually_opt: Eventually<f64, RingBuffer<Option<f64>>> = Eventually {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            operand: Box::new(Atomic::GreaterThan(10.0)),
            cache: RingBuffer::new(),
        };
        let mut eventually_naive = StlFormula {
            formula: StlOperator::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(4),
                },
                Box::new(StlOperator::GreaterThan(10.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let steps = vec![
            Step {
                value: 15.0,
                timestamp: Duration::from_secs(0),
            },
            Step {
                value: 12.0,
                timestamp: Duration::from_secs(2),
            },
            Step {
                value: 8.0,
                timestamp: Duration::from_secs(4),
            },
            Step {
                value: 5.0,
                timestamp: Duration::from_secs(6),
            },
            Step {
                value: 20.0,
                timestamp: Duration::from_secs(8),
            },
        ];

        let expected_rob = vec![None, None, Some(5.0), Some(2.0), Some(10.0)];

        for (step, expected) in steps.into_iter().zip(expected_rob.into_iter()) {
            assert_eq!(
                eventually_naive.robustness(&step),
                expected,
                "Naive implementation failed at step {:?} != {:?}",
                eventually_naive.robustness(&step),
                expected
            );
            assert_eq!(
                eventually_opt.robustness(&step),
                expected,
                "Optimized implementation failed at step {:?} != {:?}",
                eventually_opt.robustness(&step),
                expected
            );
        }
        
        // let step = Step {
        //     value: 15.0,
        //     timestamp: Duration::from_secs(9),
        // };
        // run_robustness_test(
        //     Box::new(eventually_naive),
        //     Box::new(eventually_opt),
        //     &step,
        //     Some(10.0),
        // );

    }

    #[test]
    fn globally_operator_robustness() {
        let globally_opt = Globally {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            operand: Box::new(Atomic::GreaterThan(10.0)),
            cache: RingBuffer::new(),
        };
        let globally_naive = StlFormula {
            formula: StlOperator::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(StlOperator::GreaterThan(10.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(
            Box::new(globally_naive),
            Box::new(globally_opt),
            &step,
            None,
        );
    }

    #[test]
    fn until_operator_robustness() {
        let until_opt = Until {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            left: Box::new(Atomic::GreaterThan(10.0)),
            right: Box::new(Atomic::LessThan(20.0)),
            cache: RingBuffer::new(),
        };
        let until_naive = StlFormula {
            formula: StlOperator::Until(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(10),
                },
                Box::new(StlOperator::GreaterThan(10.0)),
                Box::new(StlOperator::LessThan(20.0)),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(Box::new(until_naive), Box::new(until_opt), &step, None);
    }

    #[test]
    fn nested_temporal_operators_robustness() {
        let nested_opt = Globally {
            interval: TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(20),
            },
            operand: Box::new(Eventually {
                interval: TimeInterval {
                    start: Duration::from_secs(5),
                    end: Duration::from_secs(15),
                },
                operand: Box::new(Until {
                    interval: TimeInterval {
                        start: Duration::from_secs(2),
                        end: Duration::from_secs(10),
                    },
                    left: Box::new(Atomic::GreaterThan(10.0)),
                    right: Box::new(Atomic::LessThan(5.0)),
                    cache: RingBuffer::new(),
                }),
                cache: RingBuffer::new(),
            }),
            cache: RingBuffer::new(),
        };

        let nested_naive = StlFormula {
            formula: StlOperator::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(20),
                },
                Box::new(StlOperator::Eventually(
                    TimeInterval {
                        start: Duration::from_secs(5),
                        end: Duration::from_secs(15),
                    },
                    Box::new(StlOperator::Until(
                        TimeInterval {
                            start: Duration::from_secs(2),
                            end: Duration::from_secs(10),
                        },
                        Box::new(StlOperator::GreaterThan(10.0)),
                        Box::new(StlOperator::LessThan(5.0)),
                    )),
                )),
            ),
            signal: RingBuffer::<f64>::new(),
        };

        let step = Step {
            value: 12.0,
            timestamp: Duration::from_secs(7),
        };

        run_robustness_test(
            Box::new(nested_naive),
            Box::new(nested_opt),
            &step,
            Some(-3.0),
        );
    }
}
