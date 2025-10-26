#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl::core::{RobustnessSemantics, TimeInterval};
    use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
    use rstest::{fixture, rstest};
    use std::fmt::Debug;
    use std::time::Duration;

    // ---
    // Helper Functions
    // ---

    fn convert_f64_vec_to_bool_vec(
        input: Vec<Vec<Step<Option<f64>>>>,
    ) -> Vec<Vec<Step<Option<bool>>>> {
        input
            .into_iter()
            .map(|inner_vec| {
                inner_vec
                    .into_iter()
                    .map(|step| {
                        let bool_value = step.value.map(|v| v > 0.0);
                        Step::new(bool_value, step.timestamp)
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
            Box::new(FormulaDefinition::GreaterThan(3.0)),
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
                Box::new(FormulaDefinition::GreaterThan(0.0)),
            )),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan(3.0)),
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
                Box::new(FormulaDefinition::GreaterThan(5.0)),
            )),
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan(0.0)),
            )),
        )
    }

    // ---
    // Signal Fixtures
    // ---

    #[fixture]
    #[once]
    fn signal_1() -> Vec<Step<f64>> {
        vec![
            Step::new(5.0, Duration::from_secs(0)),
            Step::new(4.0, Duration::from_secs(1)),
            Step::new(6.0, Duration::from_secs(2)),
            Step::new(2.0, Duration::from_secs(3)),
            Step::new(5.0, Duration::from_secs(4)),
        ]
    }

    #[fixture]
    #[once]
    fn signal_2() -> Vec<Step<f64>> {
        vec![
            Step::new(1.0, Duration::from_secs(0)),
            Step::new(1.0, Duration::from_secs(1)),
            Step::new(1.0, Duration::from_secs(2)),
            Step::new(2.0, Duration::from_secs(3)),
            Step::new(3.0, Duration::from_secs(4)),
            Step::new(4.0, Duration::from_secs(5)),
            Step::new(0.0, Duration::from_secs(6)),
            Step::new(0.0, Duration::from_secs(7)),
            Step::new(0.0, Duration::from_secs(8)),
            Step::new(1.0, Duration::from_secs(9)),
            Step::new(2.0, Duration::from_secs(10)),
        ]
    }

    #[fixture]
    #[once]
    fn signal_3() -> Vec<Step<f64>> {
        vec![
            Step::new(0.0, Duration::from_secs(0)),
            Step::new(6.0, Duration::from_secs(1)),
            Step::new(1.0, Duration::from_secs(2)),
            Step::new(0.0, Duration::from_secs(3)),
            Step::new(8.0, Duration::from_secs(4)),
            Step::new(1.0, Duration::from_secs(5)),
            Step::new(7.0, Duration::from_secs(6)),
        ]
    }

    // ---
    // Expected Result "Oracles" (Plain functions)
    // ---

    fn exp_f1_s1_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new(Some(1.0), Duration::from_secs(0))],
            vec![Step::new(Some(-1.0), Duration::from_secs(1))],
            vec![Step::new(Some(-1.0), Duration::from_secs(2))],
        ]
    }

    fn exp_f1_s1_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f1_s1_f64_strict())
    }

    fn exp_f1_s1_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new(Some(true), Duration::from_secs(0))],
            vec![
                Step::new(Some(false), Duration::from_secs(1)),
                Step::new(Some(false), Duration::from_secs(2)),
                Step::new(Some(false), Duration::from_secs(3)),
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
            vec![Step::new(Some(1.0), Duration::from_secs(0))],
            vec![Step::new(Some(1.0), Duration::from_secs(1))],
            vec![Step::new(Some(1.0), Duration::from_secs(2))],
        ]
    }

    fn exp_f2_s2_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f2_s2_f64_strict())
    }

    fn exp_f2_s2_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![
                Step::new(Some(true), Duration::from_secs(0)),
                Step::new(Some(true), Duration::from_secs(1)),
                Step::new(Some(true), Duration::from_secs(2)),
                Step::new(Some(true), Duration::from_secs(3)),
            ],
            vec![
                Step::new(Some(false), Duration::from_secs(4)),
                Step::new(Some(false), Duration::from_secs(5)),
                Step::new(Some(false), Duration::from_secs(6)),
            ],
            vec![Step::new(Some(false), Duration::from_secs(7))],
            vec![Step::new(Some(false), Duration::from_secs(8))],
            vec![],
            vec![],
        ]
    }

    fn exp_f3_s3_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new(Some(0.0), Duration::from_secs(0))],
            vec![Step::new(Some(0.0), Duration::from_secs(1))],
            vec![Step::new(Some(0.0), Duration::from_secs(2))],
            vec![Step::new(Some(0.0), Duration::from_secs(3))],
            vec![Step::new(Some(1.0), Duration::from_secs(4))],
        ]
    }

    fn exp_f3_s3_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f3_s3_f64_strict())
    }

    fn exp_f3_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![Step::new(Some(false), Duration::from_secs(0))],
            vec![],
            vec![],
            vec![
                Step::new(Some(false), Duration::from_secs(1)),
                Step::new(Some(false), Duration::from_secs(2)),
                Step::new(Some(false), Duration::from_secs(3)),
            ],
            vec![],
            vec![],
            vec![Step::new(Some(true), Duration::from_secs(4))],
        ]
    }

    // ---
    // Single, Unified Test Runner
    // ---

    #[rstest]
    // --- Cases for Formula 1, Signal 1 ---
    #[case::f1_s1_naive_f64_strict(
        formula_1(),
        signal_1(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f1_s1_f64_strict()
    )]
    #[case::f1_s1_inc_f64_strict(
        formula_1(),
        signal_1(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Strict,
        exp_f1_s1_f64_strict()
    )]
    #[case::f1_s1_naive_bool_strict(
        formula_1(),
        signal_1(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f1_s1_bool_strict()
    )]
    #[case::f1_s1_inc_bool_eager(
        formula_1(),
        signal_1(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Eager,
        exp_f1_s1_bool_eager()
    )]
    // --- Cases for Formula 2, Signal 2 ---
    #[case::f2_s2_inc_bool_eager(
        formula_2(),
        signal_2(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Eager,
        exp_f2_s2_bool_eager()
    )]
    #[case::f2_s2_naive_bool_strict(
        formula_2(),
        signal_2(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f2_s2_bool_strict()
    )]
    #[case::f2_s2_naive_f64_strict(
        formula_2(),
        signal_2(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f2_s2_f64_strict()
    )]
    #[case::f2_s2_inc_f64_strict(
        formula_2(),
        signal_2(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Strict,
        exp_f2_s2_f64_strict()
    )]
    // --- Cases for Formula 3, Signal 3 ---
    #[case::f3_s3_inc_bool_eager(
        formula_3(),
        signal_3(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Eager,
        exp_f3_s3_bool_eager()
    )]
    #[case::f3_s3_naive_bool_strict(
        formula_3(),
        signal_3(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f3_s3_bool_strict()
    )]
    #[case::f3_s3_naive_f64_strict(
        formula_3(),
        signal_3(),
        MonitoringStrategy::Naive,
        EvaluationMode::Strict,
        exp_f3_s3_f64_strict()
    )]
    #[case::f3_s3_inc_f64_strict(
        formula_3(),
        signal_3(),
        MonitoringStrategy::Incremental,
        EvaluationMode::Strict,
        exp_f3_s3_f64_strict()
    )]
    fn test_monitor_matrix<Y>(
        #[case] formula: FormulaDefinition,
        #[case] signal: Vec<Step<f64>>,
        #[case] strategy: MonitoringStrategy,
        #[case] evaluation_mode: EvaluationMode,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        let mut monitor = StlMonitor::builder()
            .formula(formula)
            .strategy(strategy)
            .evaluation_mode(evaluation_mode)
            .build()
            .unwrap(); // We only test success cases here

        let mut all_results = Vec::new();
        for step in signal {
            all_results.push(monitor.instantaneous_robustness(&step));
        }

        assert_eq!(all_results, expected);
    }

    // ---
    // Test Runner for Build Failures
    // ---

    // #[rstest]
    // #[case::inc_strict_bool(MonitoringStrategy::Incremental, EvaluationMode::Strict, TypeId::of::<bool>())]
    // #[case::naive_eager_f64(MonitoringStrategy::Naive, EvaluationMode::Eager, TypeId::of::<f64>())]
    // #[case::inc_eager_f64(MonitoringStrategy::Incremental, EvaluationMode::Eager, TypeId::of::<f64>())]
    // fn test_monitor_build_fails(
    //     #[case] strategy: MonitoringStrategy,
    //     #[case] evaluation_mode: EvaluationMode,
    //     #[case] type_id: std::any::TypeId,
    // ) {
    //     let formula = formula_1(); // Use any valid formula

    //     let result = if type_id == std::any::TypeId::of::<bool>() {
    //         StlMonitor::<f64, bool>::builder()
    //             .formula(formula)
    //             .strategy(strategy)
    //             .evaluation_mode(evaluation_mode)
    //             .build()
    //     } else {
    //         // This branch is a bit of a hack to handle the generic,
    //         // but it's necessary for testing.
    //         StlMonitor::<f64, f64>::builder()
    //             .formula(formula)
    //             .strategy(strategy)
    //             .evaluation_mode(evaluation_mode)
    //             .build()
    //             .map(|_| ()) // Discard the Ok value to match types
    //             .map_err(|e| e)
    //     };

    //     assert!(result.is_err());
    // }
}
