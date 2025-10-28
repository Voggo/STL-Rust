#[cfg(false)] // Disable this test module by default
#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl::{
        core::{RobustnessSemantics, TimeInterval},
        monitor::{FormulaDefinition, MonitoringStrategy, StlMonitor},
    };
    use std::fmt::Debug;
    use std::time::Duration;

    fn run_robustness_test<T, Y>(
        mut monitor_naive: StlMonitor<T, Y>,
        mut monitor_opt: StlMonitor<T, Y>,
        step: &Step<T>,
        expected: Vec<Step<Option<Y>>>,
    )
    where
        T: Clone,
        Y: RobustnessSemantics + Copy + Debug + PartialEq,
    {
        let robustness_naive = monitor_naive.instantaneous_robustness(step);
        let robustness_opt = monitor_opt.instantaneous_robustness(step);

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

    fn run_multi_step_robustness_test<T, Y>(
        mut monitor_naive: StlMonitor<T, Y>,
        mut monitor_opt: StlMonitor<T, Y>,
        steps: &[Step<T>],
        expected: &[Vec<Step<Option<Y>>>],
    )
    where
        T: Clone,
        Y: RobustnessSemantics + Copy + Debug + PartialEq,
    {
        for (step, exp) in steps.iter().zip(expected.iter()) { 
            let robustness_naive = monitor_naive.instantaneous_robustness(step);
            let robustness_opt = monitor_opt.instantaneous_robustness(step);

            assert_eq!(
                robustness_naive, *exp,
                "Naive implementation failed at timestep {:?}: {:?} != {:?}",
                step.timestamp, robustness_naive, exp
            );
            assert_eq!(
                robustness_opt, *exp,
                "Optimized implementation failed at timestep {:?}: {:?} != {:?}",
                step.timestamp, robustness_opt, exp
            );
            assert_eq!(
                robustness_naive, robustness_opt,
                "Mismatch between naive and optimized implementations at timestep {:?}: {:?} != {:?}",
                step.timestamp, robustness_naive, robustness_opt
            );
        }
    }

    fn generate_steps_and_expected(
        values: Vec<f64>,
        timestamps: Vec<u64>,
        expected_robustness_values: Vec<Option<f64>>, // This is the robustness value, not the full Step<Option<Y>>
    ) -> (Vec<Step<f64>>, Vec<Vec<Step<Option<f64>>>>) { // Changed return type
        let steps: Vec<Step<f64>> = values
            .into_iter()
            .zip(timestamps.into_iter())
            .map(|(value, timestamp)| Step {
                value,
                timestamp: Duration::from_secs(timestamp),
            })
            .collect();

        let expected_outputs: Vec<Vec<Step<Option<f64>>>> = expected_robustness_values
            .into_iter()
            .zip(steps.iter()) // Use steps to get timestamps for expected output steps
            .map(|(expected_val, original_step)| {
                // The instantaneous_robustness method returns a Vec<Step<Option<Y>>>
                // For atomic and simple operators, it typically returns a single step.
                vec![Step::new(expected_val, original_step.timestamp)]
            })
            .collect();

        (steps, expected_outputs)
    }

    fn build_monitors<T, Y>(formula: FormulaDefinition) -> (StlMonitor<T, Y>, StlMonitor<T, Y>)
    where
        T: Clone + Copy + 'static + Into<f64>,
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        let monitor_naive = StlMonitor::builder()
            .formula(formula.clone())
            .strategy(MonitoringStrategy::Naive)
            .build()
            .unwrap();

        let monitor_opt = StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap();

        (monitor_naive, monitor_opt)
    }

    #[test]
    fn formula_to_string() {
        let formula = FormulaDefinition::Implies(
            Box::new(FormulaDefinition::And(
                Box::new(FormulaDefinition::GreaterThan(5.0)),
                Box::new(FormulaDefinition::Or(
                    Box::new(FormulaDefinition::Not(Box::new(
                        FormulaDefinition::LessThan(3.0),
                    ))),
                    Box::new(FormulaDefinition::Eventually(
                        TimeInterval {
                            start: Duration::from_secs(0),
                            end: Duration::from_secs(10),
                        },
                        Box::new(FormulaDefinition::Globally(
                            TimeInterval {
                                start: Duration::from_secs(2),
                                end: Duration::from_secs(8),
                            },
                            Box::new(FormulaDefinition::Until(
                                TimeInterval {
                                    start: Duration::from_secs(1),
                                    end: Duration::from_secs(5),
                                },
                                Box::new(FormulaDefinition::True),
                                Box::new(FormulaDefinition::False),
                            )),
                        )),
                    )),
                )),
            )),
            Box::new(FormulaDefinition::LessThan(7.0)),
        );

        let (monitor_naive, monitor_opt) = build_monitors::<f64, f64>(formula);

        assert_eq!(
            monitor_opt.specification_to_string(),
            "((x > 5) ∧ ((¬(x < 3)) v (F[0, 10](G[2, 8]((True) U[1, 5] (False)))))) → (x < 7)"
        );
        assert_eq!(
            monitor_naive.specification_to_string(),
            "((x > 5) ∧ ((¬(x < 3)) v (F[0, 10](G[2, 8]((True) U[1, 5] (False)))))) → (x < 7)"
        );
        assert_eq!(
            monitor_opt.specification_to_string(),
            monitor_naive.specification_to_string()
        );
    }

    #[test]
    fn atomic_greater_than_robustness() {
        let formula = FormulaDefinition::GreaterThan(10.0);

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let values = vec![15.0, 8.0];
        let timestamps = vec![5, 5];
        let expected = vec![Some(5.0), Some(-2.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn atomic_less_than_robustness() {
        let formula = FormulaDefinition::LessThan(10.0);

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let values = vec![5.0, 12.0];
        let timestamps = vec![5, 5];
        let expected = vec![Some(5.0), Some(-2.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn atomic_true_robustness() {
        let formula = FormulaDefinition::True;

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(f64::INFINITY), step.timestamp)]);
    }

    #[test]
    fn atomic_false_robustness() {
        let formula = FormulaDefinition::False;

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(f64::NEG_INFINITY), step.timestamp)]);
    }

    #[test]
    fn not_operator_robustness() {
        let formula = FormulaDefinition::Not(Box::new(FormulaDefinition::GreaterThan(10.0)));
        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(-5.0), step.timestamp)]);
    }

    #[test]
    fn and_operator_robustness() {
        let formula = FormulaDefinition::And(
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn or_operator_robustness() {
        let formula = FormulaDefinition::Or(
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(5.0)),
        );

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn implies_operator_robustness() {
        let formula = FormulaDefinition::Implies(
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn or_and_law_robustness() {
        // tests defined as p v q := ¬(¬p ∧ ¬q)
        let formula_or = FormulaDefinition::Or(
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let formula_and = FormulaDefinition::Not(Box::new(FormulaDefinition::And(
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::GreaterThan(10.0),
            ))),
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::LessThan(20.0),
            ))),
        )));

        let (monitor_or_naive, monitor_or_opt) = build_monitors(formula_or);
        let (monitor_and_naive, monitor_and_opt) = build_monitors(formula_and);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_or_naive, monitor_or_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
        run_robustness_test(monitor_and_naive, monitor_and_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn implies_law_robustness() {
        //tests defined as p -> q  := ¬p v q
        let formula_implies = FormulaDefinition::Implies(
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let formula_or = FormulaDefinition::Or(
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::GreaterThan(10.0),
            ))),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let (monitor_implies_opt, monitor_implies_naive) = build_monitors(formula_implies.clone());
        let (monitor_or_opt, monitor_or_naive) = build_monitors(formula_or);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_implies_naive, monitor_implies_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
        run_robustness_test(monitor_or_naive, monitor_or_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn eventually_law_robustness() {
        // tests defined as F[a,b] p := True U[a,b] p
        let formula_eventually = FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::GreaterThan(10.0)),
        );

        let formula_until = FormulaDefinition::Until(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::True),
            Box::new(FormulaDefinition::GreaterThan(10.0)),
        );

        let (monitor_eventually_opt, monitor_eventually_naive) =
            build_monitors(formula_eventually.clone());
        let (monitor_until_opt, monitor_until_naive) = build_monitors(formula_until);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(7),
        };

        run_robustness_test(
            monitor_eventually_naive,
            monitor_eventually_opt,
            &step,
            vec![Step::new(Some(5.0), step.timestamp)],
        );
        run_robustness_test(monitor_until_naive, monitor_until_opt, &step, vec![Step::new(Some(5.0), step.timestamp)]);
    }

    #[test]
    fn globally_law_robustness() {
        // tests defined as G[a,b] p := ¬F[a,b] ¬p
        let formula_globally = FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::GreaterThan(10.0)),
        );

        let formula_eventually = FormulaDefinition::Not(Box::new(FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(5),
            },
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::GreaterThan(10.0),
            ))),
        )));

        let (monitor_globally_opt, monitor_globally_naive) =
            build_monitors(formula_globally.clone());
        let (monitor_eventually_opt, monitor_eventually_naive) = build_monitors(formula_eventually);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(7),
        };

        run_robustness_test(
            monitor_globally_naive,
            monitor_globally_opt,
            &step,
            vec![Step::new(Some(5.0), step.timestamp)],
        );
        run_robustness_test(
            monitor_eventually_naive,
            monitor_eventually_opt,
            &step,
            vec![Step::new(Some(5.0), step.timestamp)],
        );
    }

    #[test]
    fn eventually_operator_robustness() {
        let formula = FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            Box::new(FormulaDefinition::GreaterThan(10.0)),
        );

        let (monitor_opt, monitor_naive) = build_monitors(formula);

        let values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let timestamps = vec![0, 2, 4, 6, 8];
        let expected = vec![None, None, Some(5.0), Some(2.0), Some(2.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn globally_operator_robustness() {
        let ti = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(4),
        };
        let formula =
            FormulaDefinition::Globally(ti.clone(), Box::new(FormulaDefinition::GreaterThan(10.0)));

        let (monitor_opt, monitor_naive) = build_monitors(formula);

        let values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let timestamps = vec![0, 2, 4, 6, 8];
        let expected = vec![None, None, Some(-2.0), Some(-5.0), Some(-5.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn until_operator_robustness() {
        let formula = FormulaDefinition::Until(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            Box::new(FormulaDefinition::GreaterThan(10.0)),
            Box::new(FormulaDefinition::LessThan(20.0)),
        );

        let (monitor_opt, monitor_naive) = build_monitors(formula);

        let values = vec![15.0, 12.0, 8.0, 5.0, 12.0];
        let timestamps = vec![0, 2, 4, 6, 8];
        let expected = vec![None, None, Some(5.0), Some(2.0), Some(-2.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn nested_temporal_operators_robustness() {
        let formula = FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(10),
            },
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(1),
                    end: Duration::from_secs(3),
                },
                Box::new(FormulaDefinition::GreaterThan(10.0)),
            )),
        );

        let (monitor_opt, monitor_naive) = build_monitors(formula);

        let values = vec![15.0, 12.0, 8.0, 5.0, 12.0, 20.0];
        let timestamps = vec![0, 3, 6, 9, 12, 15];
        let expected = vec![None, None, None, None, Some(2.0), Some(10.0)];
        let (steps, expected_rob) = generate_steps_and_expected(values, timestamps, expected);

        run_multi_step_robustness_test(monitor_naive, monitor_opt, &steps, &expected_rob);
    }

    #[test]
    fn boolean_tests() {
        let formula = FormulaDefinition::GreaterThan(0.0);

        let (monitor_opt, monitor_naive) = build_monitors(formula);

        let step = Step {
            value: 1.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, vec![Step::new(Some(true), step.timestamp)]);
    }

// --------------------------------------------------- //
// old tests from robustness_cached.rs below here. should be updated versions of the same
//  in the monitor_test.rs file but for now kept here for reference
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

