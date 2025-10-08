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
        expected: Option<Y>,
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
        expected: &[Option<Y>],
    )
    where
        T: Clone,
        Y: RobustnessSemantics + Copy + Debug + PartialEq,
    {
        for (step, &exp) in steps.iter().zip(expected.iter()) {
            let robustness_naive = monitor_naive.instantaneous_robustness(step);
            let robustness_opt = monitor_opt.instantaneous_robustness(step);

            assert_eq!(
                robustness_naive, exp,
                "Naive implementation failed at timestep {:?}: {:?} != {:?}",
                step.timestamp, robustness_naive, exp
            );
            assert_eq!(
                robustness_opt, exp,
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
        expected: Vec<Option<f64>>,
    ) -> (Vec<Step<f64>>, Vec<Option<f64>>) {
        let steps = values
            .into_iter()
            .zip(timestamps.into_iter())
            .map(|(value, timestamp)| Step {
                value,
                timestamp: Duration::from_secs(timestamp),
            })
            .collect();
        (steps, expected)
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

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(f64::INFINITY));
    }

    #[test]
    fn atomic_false_robustness() {
        let formula = FormulaDefinition::False;

        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 0.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(f64::NEG_INFINITY));
    }

    #[test]
    fn not_operator_robustness() {
        let formula = FormulaDefinition::Not(Box::new(FormulaDefinition::GreaterThan(10.0)));
        let (monitor_naive, monitor_opt) = build_monitors(formula);

        let step = Step {
            value: 15.0,
            timestamp: Duration::from_secs(5),
        };

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(-5.0));
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

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(5.0));
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

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(5.0));
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

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(5.0));
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

        run_robustness_test(monitor_or_naive, monitor_or_opt, &step, Some(5.0));
        run_robustness_test(monitor_and_naive, monitor_and_opt, &step, Some(5.0));
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

        run_robustness_test(monitor_implies_naive, monitor_implies_opt, &step, Some(5.0));
        run_robustness_test(monitor_or_naive, monitor_or_opt, &step, Some(5.0));
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
            Some(5.0),
        );
        run_robustness_test(monitor_until_naive, monitor_until_opt, &step, Some(5.0));
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
            Some(5.0),
        );
        run_robustness_test(
            monitor_eventually_naive,
            monitor_eventually_opt,
            &step,
            Some(5.0),
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

        run_robustness_test(monitor_naive, monitor_opt, &step, Some(true));
    }
}
