#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl::core::{RobustnessSemantics, TimeInterval};
    use ostl::stl::monitor::{FormulaDefinition, MonitoringStrategy, StlMonitor, EvaluationMode};
    use rstest::{fixture, rstest};
    use std::fmt::Debug;
    use std::time::Duration;

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
            // Left operand
            Box::new(FormulaDefinition::Globally(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(2),
                },
                Box::new(FormulaDefinition::GreaterThan(0.0)),
            )),
            // Right operand
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
    fn signal_3 () -> Vec<Step<f64>> {
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

    #[fixture]
    #[once]
    fn exp_f3_s3_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f3_s3_f64_strict())
    }

    fn exp_f3_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![
                Step::new(Some(false), Duration::from_secs(0)),
            ],
            vec![],
            vec![],
            vec![
                Step::new(Some(false), Duration::from_secs(1)),
                Step::new(Some(false), Duration::from_secs(2)),
                Step::new(Some(false), Duration::from_secs(3)),
            ],
            vec![],
            vec![],
            vec![
                Step::new(Some(false), Duration::from_secs(2)),
                Step::new(Some(false), Duration::from_secs(3)),
                Step::new(Some(true), Duration::from_secs(4)),
            ],
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
            vec![Step::new(Some(1.0), Duration::from_secs(0))], // t=0 completes at t=8
            vec![Step::new(Some(1.0), Duration::from_secs(1))], // t=1 completes at t=9
            vec![Step::new(Some(1.0), Duration::from_secs(2))], // t=2 completes at t=10
        ]
    }

    #[fixture]
    #[once]
    fn exp_f2_s2_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f2_s2_f64_strict())
    }

    #[fixture]
    #[once]
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



    #[fixture]
    #[once]
    fn exp_f1_s1_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],
            vec![],
            vec![Step::new(Some(1.0), Duration::from_secs(0))],
            vec![Step::new(Some(-1.0), Duration::from_secs(1))],
            vec![Step::new(Some(-1.0), Duration::from_secs(2))],
        ]
    }

    #[fixture]
    #[once]
    fn exp_f1_s1_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f1_s1_f64_strict())
    }

    #[fixture]
    #[once]
    fn exp_f1_s1_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![
                Step::new(Some(true), Duration::from_secs(0)), // t=0 completes
            ],
            vec![
                Step::new(Some(false), Duration::from_secs(1)), // t=1 completes
                Step::new(Some(false), Duration::from_secs(2)), // t=2 short-circuits to false
                Step::new(Some(false), Duration::from_secs(3)), // t=3 short-circuits to false
            ],
            vec![],
        ]
    }

    #[rstest]
    #[case::naive_f64_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f1_s1_f64_strict())]
    #[case::incremental_f64_strict(MonitoringStrategy::Incremental, EvaluationMode::Strict, exp_f1_s1_f64_strict())]
    #[case::naive_bool_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f1_s1_bool_strict())]
    #[case::incremental_bool_eager(MonitoringStrategy::Incremental, EvaluationMode::Eager, exp_f1_s1_bool_eager())]
    fn test_formula_1<Y>(
        #[case] strategy: MonitoringStrategy,
        #[case] evaluation_mode: EvaluationMode,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        let formula = formula_1();
        let signal = signal_1();

        let mut monitor = StlMonitor::builder()
            .formula(formula)
            .strategy(strategy)
            .evaluation_mode(evaluation_mode)
            .build()
            .unwrap();

        let mut all_results = Vec::new();
        for step in signal {
            all_results.push(monitor.instantaneous_robustness(&step));
        }

        assert_eq!(all_results, expected);
    }

    #[rstest]
    #[case::incremental_bool_eager(MonitoringStrategy::Incremental, EvaluationMode::Eager, exp_f2_s2_bool_eager())]
    #[case::naive_bool_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f2_s2_bool_strict())]
    #[case::naive_f64_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f2_s2_f64_strict())]
    #[case::incremental_f64_strict(MonitoringStrategy::Incremental, EvaluationMode::Strict, exp_f2_s2_f64_strict())]
    fn test_formula_2<Y>(
        #[case] strategy: MonitoringStrategy,
        #[case] evaluation_mode: EvaluationMode,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        let formula = formula_2();
        let signal = signal_2();

        let mut monitor = StlMonitor::builder()
            .formula(formula)
            .strategy(strategy)
            .evaluation_mode(evaluation_mode)
            .build()
            .unwrap();

        let mut all_results = Vec::new();
        for step in signal {
            all_results.push(monitor.instantaneous_robustness(&step));
        }

        assert_eq!(all_results, expected);
    }

    #[rstest]
    #[case::incremental_bool_eager(MonitoringStrategy::Incremental, EvaluationMode::Eager, exp_f3_s3_bool_eager())]
    #[case::naive_bool_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f3_s3_bool_strict())]
    #[case::naive_f64_strict(MonitoringStrategy::Naive, EvaluationMode::Strict, exp_f3_s3_f64_strict())]
    #[case::incremental_f64_strict(MonitoringStrategy::Incremental, EvaluationMode::Strict, exp_f3_s3_f64_strict())]
    fn test_formula_3<Y>(
        #[case] strategy: MonitoringStrategy,
        #[case] evaluation_mode: EvaluationMode,
        #[case] expected: Vec<Vec<Step<Option<Y>>>>,
    ) where
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq,
    {
        let formula = formula_3();
        let signal = signal_3();

        let mut monitor = StlMonitor::builder()
            .formula(formula)
            .strategy(strategy)
            .evaluation_mode(evaluation_mode)
            .build()
            .unwrap();

        let mut all_results = Vec::new();
        for step in signal {
            all_results.push(monitor.instantaneous_robustness(&step));
        }

        assert_eq!(all_results, expected);
    }
}
