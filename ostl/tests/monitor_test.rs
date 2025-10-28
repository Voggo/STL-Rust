#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl::core::{RobustnessSemantics, TimeInterval};
    use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
    use rstest::{fixture, rstest};
    use std::fmt::Debug;
    use std::time::Duration;
    use std::vec;

    // ---
    // Helper Functions
    // ---

    fn convert_f64_vec_to_bool_vec(
        input: Vec<Vec<Step<Option<f64>>>>,
        zero_is_true: Option<bool>,
    ) -> Vec<Vec<Step<Option<bool>>>> {
        input
            .into_iter()
            .map(|inner_vec| {
                inner_vec
                    .into_iter()
                    .map(|step| {
                        let bool_value = step
                            .value
                            .map(|v| v > 0.0 || (zero_is_true.unwrap_or(false) && v == 0.0));
                        Step::new("output", bool_value, step.timestamp)
                    })
                    .collect()
            })
            .collect()
    }

    // ---
    // Formula Fixtures
    // ---

    #[fixture]
    #[once]
    fn formula_1() -> FormulaDefinition {
        FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(2),
            },
            Box::new(FormulaDefinition::GreaterThan("x",3.0)),
        )
    }

    #[fixture]
    #[once]
    fn formula_2() -> FormulaDefinition {
        // (G[0,2] (x > 0)) U[0,6] (F[0,2] (x > 3))
        FormulaDefinition::Until(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(6),
            },
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",0.0)),
            )),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",3.0)),
            )),
        )
    }

    #[fixture]
    #[once]
    fn formula_3() -> FormulaDefinition { 
        // F[0,2] (x > 5) && G[0, 2] (x > 0)
        FormulaDefinition::And(
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",5.0)),
            )),
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",0.0)),
            )),
        )
    }

    #[fixture]
    #[once]
    fn formula_4() -> FormulaDefinition {
        // F[0, 2] (x > 5) && True
        FormulaDefinition::And(
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",5.0)),
            )),
            Box::new(FormulaDefinition::True),
        )
    }

    #[fixture]
    #[once]
    fn formula_5() -> FormulaDefinition {
        // F[0, 2] (x > 5)
        FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(2),
            },
            Box::new(FormulaDefinition::GreaterThan("x",5.0)),
        )
    }

    #[fixture]
    #[once]
    fn formula_6() -> FormulaDefinition {
        // G[0, 5] (x > 0) -> F[0, 2] (x > 3)
        FormulaDefinition::Implies(
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(5),
                },
                Box::new(FormulaDefinition::GreaterThan("x",0.0)),
            )),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan("x",3.0)),
            )),
        )
    }

    #[fixture]
    #[once]
    fn formula_7() -> FormulaDefinition {
        // !x<5 || F
        FormulaDefinition::Or(
            Box::new(FormulaDefinition::Not(Box::new(
                FormulaDefinition::LessThan("x",5.0),
            ))),
            Box::new(FormulaDefinition::False),
        )
    }

    // ---
    // Signal Fixtures
    // ---

    #[fixture]
    #[once]
    fn signal_1() -> Vec<Step<f64>> {
        vec![
            Step::new("x", 5.0, Duration::from_secs(0)),
            Step::new("x", 4.0, Duration::from_secs(1)),
            Step::new("x", 6.0, Duration::from_secs(2)),
            Step::new("x", 2.0, Duration::from_secs(3)),
            Step::new("x", 5.0, Duration::from_secs(4)),
        ]
    }

    #[fixture]
    #[once]
    fn signal_2() -> Vec<Step<f64>> {
        vec![
            Step::new("x", 1.0, Duration::from_secs(0)),
            Step::new("x", 1.0, Duration::from_secs(1)),
            Step::new("x", 1.0, Duration::from_secs(2)),
            Step::new("x", 2.0, Duration::from_secs(3)),
            Step::new("x", 3.0, Duration::from_secs(4)),
            Step::new("x", 4.0, Duration::from_secs(5)),
            Step::new("x", 0.0, Duration::from_secs(6)),
            Step::new("x", 0.0, Duration::from_secs(7)),
            Step::new("x", 0.0, Duration::from_secs(8)),
            Step::new("x", 1.0, Duration::from_secs(9)),
            Step::new("x", 2.0, Duration::from_secs(10)),
        ]
    }

    #[fixture]
    #[once]
    fn signal_3() -> Vec<Step<f64>> {
        vec![
            Step::new("x", 0.0, Duration::from_secs(0)),
            Step::new("x", 6.0, Duration::from_secs(1)),
            Step::new("x", 1.0, Duration::from_secs(2)),
            Step::new("x", 0.0, Duration::from_secs(3)),
            Step::new("x", 8.0, Duration::from_secs(4)),
            Step::new("x", 1.0, Duration::from_secs(5)),
            Step::new("x", 7.0, Duration::from_secs(6)),
        ]
    }

    // ---
    // Expected Result "Oracles" (Plain functions)
    // ---

    fn exp_f1_s1_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new("output", Some(1.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(-1.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(-1.0), Duration::from_secs(2))],
        ]
    }

    fn exp_f1_s1_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f1_s1_f64_strict(), None)
    }

    fn exp_f1_s1_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new("output", Some(true), Duration::from_secs(0))],
            vec![
                Step::new("output", Some(false), Duration::from_secs(1)),
                Step::new("output", Some(false), Duration::from_secs(2)),
                Step::new("output", Some(false), Duration::from_secs(3)),
            ],
            vec![],
        ]
    }

    fn exp_f2_s2_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![Step::new("output", Some(1.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(2))],
        ]
    }

    fn exp_f2_s2_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f2_s2_f64_strict(), None)
    }

    fn exp_f2_s2_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![
                Step::new("output", Some(true), Duration::from_secs(0)),
                Step::new("output", Some(true), Duration::from_secs(1)),
                Step::new("output", Some(true), Duration::from_secs(2)),
                Step::new("output", Some(true), Duration::from_secs(3)),
            ],
            vec![
                Step::new("output", Some(false), Duration::from_secs(4)),
                Step::new("output", Some(false), Duration::from_secs(5)),
                Step::new("output", Some(false), Duration::from_secs(6)),
            ],
            vec![Step::new("output", Some(false), Duration::from_secs(7))],
            vec![Step::new("output", Some(false), Duration::from_secs(8))],
            vec![],
            vec![],
        ]
    }

    fn exp_f3_s3_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new("output", Some(0.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(2))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(3))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(4))],
        ]
    }

    fn exp_f3_s3_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f3_s3_f64_strict(), None)
    }

    fn exp_f3_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![Step::new("output", Some(false), Duration::from_secs(0))],
            vec![],
            vec![],
            vec![
                Step::new("output", Some(false), Duration::from_secs(1)),
                Step::new("output", Some(false), Duration::from_secs(2)),
                Step::new("output", Some(false), Duration::from_secs(3)),
            ],
            vec![],
            vec![],
            vec![Step::new("output", Some(true), Duration::from_secs(4))],
        ]
    }

    fn exp_f4_s3_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new("output", Some(1.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(3.0), Duration::from_secs(2))],
            vec![Step::new("output", Some(3.0), Duration::from_secs(3))],
            vec![Step::new("output", Some(3.0), Duration::from_secs(4))],
        ]
    }

    fn exp_f4_s3_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f4_s3_f64_strict(), None)
    }

    fn exp_f4_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![
                Step::new("output", Some(true), Duration::from_secs(0)),
                Step::new("output", Some(true), Duration::from_secs(1)),
            ],
            vec![],
            vec![],
            vec![
                Step::new("output", Some(true), Duration::from_secs(2)),
                Step::new("output", Some(true), Duration::from_secs(3)),
                Step::new("output", Some(true), Duration::from_secs(4)),
            ],
            vec![],
            vec![
                Step::new("output", Some(true), Duration::from_secs(5)),
                Step::new("output", Some(true), Duration::from_secs(6)),
            ],
        ]
    }

    fn exp_f6_s2_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![
                // step t=5s
                Step::new("output", Some(false), Duration::from_secs(0)),
            ],
            vec![
                // step t=6s, x=0.0, G[0,5](x>0) is false => short-circuit to true
                Step::new("output", Some(true), Duration::from_secs(1)),
                Step::new("output", Some(true), Duration::from_secs(2)),
                Step::new("output", Some(true), Duration::from_secs(3)),
                Step::new("output", Some(true), Duration::from_secs(4)),
                Step::new("output", Some(true), Duration::from_secs(5)),
                Step::new("output", Some(true), Duration::from_secs(6)),
            ],
            vec![Step::new("output", Some(true), Duration::from_secs(7))],
            vec![Step::new("output", Some(true), Duration::from_secs(8))],
            vec![],
            vec![],
        ]
    }
    fn exp_f6_s2_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![Step::new("output", Some(-1.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(2))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(3))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(4))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(5))],
        ]
    }
    fn exp_f6_s2_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f6_s2_f64_strict(), Some(true))
    }

    fn exp_f7_s3_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![Step::new("output", Some(-5.0), Duration::from_secs(0))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(-4.0), Duration::from_secs(2))],
            vec![Step::new("output", Some(-5.0), Duration::from_secs(3))],
            vec![Step::new("output", Some(3.0), Duration::from_secs(4))],
            vec![Step::new("output", Some(-4.0), Duration::from_secs(5))],
            vec![Step::new("output", Some(2.0), Duration::from_secs(6))],
        ]
    }

    fn exp_f7_s3_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f7_s3_f64_strict(), None)
    }

    fn exp_f7_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        exp_f7_s3_bool_strict()
    }

    /// This helper function contains the actual test logic.
    /// It is called by the `rstest` runners below.
    fn run_monitor_test<Y>(
        formulas: Vec<FormulaDefinition>,
        signal: Vec<Step<f64>>,
        strategy: MonitoringStrategy,
        evaluation_mode: EvaluationMode,
        expected: Vec<Vec<Step<Option<Y>>>>,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        for (i, formula) in formulas.into_iter().enumerate() {
            let mut monitor = StlMonitor::builder()
                .formula(formula.clone())
                .strategy(strategy)
                .evaluation_mode(evaluation_mode)
                .build()
                .unwrap();

            let mut all_results = Vec::new();
            for step in signal.clone() {
                all_results.push(monitor.instantaneous_robustness(&step));
                // println!(
                //     "Step at {:?}, Monitor Output: {:?}",
                //     step.timestamp,
                //     all_results.last().unwrap()
                // );
            }

            assert_eq!(
                all_results,
                expected,
                "Test failed for formula at index {} ({}) with strategy {:?} and mode {:?}",
                i,
                monitor.specification_to_string(),
                strategy,
                evaluation_mode
            );
        }
    }

    #[rstest]
    // --- f64 Strict Cases ---
    // These run with EvaluationMode::Strict and are tested against
    // both Naive and Incremental strategies.
    #[case::f1_s1(vec![formula_1()], signal_1(), exp_f1_s1_f64_strict())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_f64_strict())]
    #[case::f3_s3(vec![formula_3()], signal_3(), exp_f3_s3_f64_strict())]
    #[case::f6_s2(vec![formula_6()], signal_2(), exp_f6_s2_f64_strict())]
    #[case::f4_s3(vec![formula_4(), formula_5()], signal_3(), exp_f4_s3_f64_strict())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_f64_strict())]
    fn test_f64_strict<Y>(
        #[case] formulas: Vec<FormulaDefinition>,
        #[case] signal: Vec<Step<f64>>,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
        #[values(MonitoringStrategy::Naive, MonitoringStrategy::Incremental)]
        strategy: MonitoringStrategy,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        run_monitor_test(formulas, signal, strategy, EvaluationMode::Strict, expected);
    }

    #[rstest]
    // --- bool Strict Cases ---
    // These run with EvaluationMode::Strict and are tested against
    // both Naive and Incremental strategies.
    #[case::f1_s1(vec![formula_1()], signal_1(), exp_f1_s1_bool_strict())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_bool_strict())]
    #[case::f3_s3(vec![formula_3()], signal_3(), exp_f3_s3_bool_strict())]
    #[case::f6_s2(vec![formula_6()], signal_2(), exp_f6_s2_bool_strict())]
    #[case::f4_s3(vec![formula_4(), formula_5()], signal_3(), exp_f4_s3_bool_strict())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_bool_strict())]
    fn test_bool_strict<Y>(
        #[case] formulas: Vec<FormulaDefinition>,
        #[case] signal: Vec<Step<f64>>,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
        #[values(MonitoringStrategy::Naive, MonitoringStrategy::Incremental)]
        strategy: MonitoringStrategy,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        run_monitor_test(formulas, signal, strategy, EvaluationMode::Strict, expected);
    }

    #[rstest]
    // --- bool Eager Cases ---
    // These run with EvaluationMode::Eager and are tested *only*
    // against the Incremental strategy (Naive+Eager is invalid).
    #[case::f1_s1(vec![formula_1()], signal_1(), exp_f1_s1_bool_eager())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_bool_eager())]
    #[case::f3_s3(vec![formula_3()], signal_3(), exp_f3_s3_bool_eager())]
    #[case::f6_s2(vec![formula_6()], signal_2(), exp_f6_s2_bool_eager())]
    #[case::f4_s3(vec![formula_4(), formula_5()], signal_3(), exp_f4_s3_bool_eager())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_bool_eager())]
    fn test_bool_eager<Y>(
        #[case] formulas: Vec<FormulaDefinition>,
        #[case] signal: Vec<Step<f64>>,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
        #[values(MonitoringStrategy::Incremental)] strategy: MonitoringStrategy,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        run_monitor_test(formulas, signal, strategy, EvaluationMode::Eager, expected);
    }

    // ---
    // Test Runner for Build Failures
    // ---

    #[rstest]
    #[should_panic]
    fn test_monitor_build_fails_f64_eager(
        #[values(MonitoringStrategy::Naive, MonitoringStrategy::Incremental)]
        strategy: MonitoringStrategy,
    ) {
        // Eager mode + f64 robustness is an invalid combination
        let _: StlMonitor<f64, f64> = StlMonitor::builder()
            .formula(formula_1())
            .strategy(strategy)
            .evaluation_mode(EvaluationMode::Eager)
            .build()
            .unwrap();
    }

    #[rstest]
    #[should_panic]
    fn test_monitor_build_fails_bool_naive_eager() {
        // Eager mode + Naive strategy is an invalid combination
        let _: StlMonitor<f64, bool> = StlMonitor::builder()
            .formula(formula_1())
            .strategy(MonitoringStrategy::Naive)
            .evaluation_mode(EvaluationMode::Eager)
            .build()
            .unwrap();
    }
}
