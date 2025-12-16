#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl;
    use ostl::stl::core::{RobustnessInterval, RobustnessSemantics};
    use ostl::stl::monitor::{EvaluationMode, MonitoringStrategy, StlMonitor};
    use ostl::stl::formula_definition::FormulaDefinition;
    use pretty_assertions::assert_eq;
    use rstest::{fixture, rstest};
    use std::collections::HashMap;
    use std::f64::consts::PI;
    use std::fmt::Debug;
    use std::time::Duration;
    use std::vec;

    fn convert_f64_vec_to_bool_vec(
        input: Vec<Vec<Step<Option<f64>>>>,
    ) -> Vec<Vec<Step<Option<bool>>>> {
        input
            .into_iter()
            .map(|inner_vec| {
                inner_vec
                    .into_iter()
                    .map(|step| {
                        let bool_value = step
                            .value
                            .map(|v| v > 0.0 || (v == 0.0 && v.is_sign_negative()));
                        Step::new("output", bool_value, step.timestamp)
                    })
                    .collect()
            })
            .collect()
    }

    // Helper to create a vector of steps
    fn create_steps(name: &'static str, values: Vec<f64>, timestamps: Vec<u64>) -> Vec<Step<f64>> {
        values
            .into_iter()
            .zip(timestamps)
            .map(|(val, ts)| Step::new(name, val, Duration::from_secs(ts)))
            .collect()
    }

    fn combine_and_sort_steps(step_vectors: Vec<Vec<Step<f64>>>) -> Vec<Step<f64>> {
        let mut combined_steps = step_vectors.into_iter().flatten().collect::<Vec<_>>();
        combined_steps.sort_by_key(|step| step.timestamp);
        combined_steps
    }

    // ---
    // Formula Fixtures
    // ---

    #[fixture]
    #[once]
    fn formula_1() -> FormulaDefinition {
        stl! {
            G[0,2] (x > 3)
        }
    }

    #[fixture]
    #[once]
    fn formula_1_alt() -> FormulaDefinition {
        // G[a,b](phi) is equivalent to !(F[a,b]( ! (phi)))
        stl! {
            !(F[0,2] (x <= 3))
        }
    }

    #[fixture]
    #[once]
    fn formula_1_alt_2() -> FormulaDefinition {
        // F[a,b](phi) is equivalent to (true) U[a,b] (phi), so G[a,b](phi) is also equivalent to !( (true) U[a,b] ( ! (phi)))
        stl! {
            !( (true) U[0,2] (x <= 3))
        }
    }

    #[fixture]
    #[once]
    fn formula_2() -> FormulaDefinition {
        stl! {(G[0,2] (x > 0)) U[0,6] (F[0,2] (x > 3))}
    }

    #[fixture]
    #[once]
    fn formula_3() -> FormulaDefinition {
        stl! {(F[0,2] (x > 5)) && (G[0, 2] (x > 0))}
    }

    #[fixture]
    #[once]
    fn formula_3_alt() -> FormulaDefinition {
        // G[a,b](phi) is equivalent to !(F[a,b]( ! (phi)))
        stl! {((true) U[0,2] (x > 5)) && (!(F[0,2] (x <= 0)))}
    }

    #[fixture]
    #[once]
    fn formula_4() -> FormulaDefinition {
        stl! {(F[0, 2] (x > 5)) && (true)}
    }

    #[fixture]
    #[once]
    fn formula_5() -> FormulaDefinition {
        stl! {F[0, 2] (x > 5)}
    }

    #[fixture]
    #[once]
    fn formula_5_alt() -> FormulaDefinition {
        // F[a,b](phi) is equivalent to (true) U[a,b] (phi)
        stl! {(true) U[0, 2] (x > 5)}
    }

    #[fixture]
    #[once]
    fn formula_6() -> FormulaDefinition {
        stl! {(G[0, 5] (x > 0)) -> (F[0, 2] (x > 3))}
    }

    #[fixture]
    #[once]
    fn formula_6_alt() -> FormulaDefinition {
        // using rules for globally (in terms of eventually) and eventually (in terms of until)
        stl! {
            (!(F[0,5](x <= 0))) -> ((true) U[0,2](x > 3))
        }
    }

    #[fixture]
    #[once]
    fn formula_7() -> FormulaDefinition {
        stl! {(!(x < 5)) || (false)}
    }

    #[fixture]
    #[once]
    fn formula_8() -> FormulaDefinition {
        stl! {(G[0,2](x>0)) && (y<5)}
    }

    #[fixture]
    #[once]
    fn formula_8_alt() -> FormulaDefinition {
        stl! {(!(F[0,2](x<=0))) && (y<5)}
    }

    #[fixture]
    #[once]
    fn formula_9() -> FormulaDefinition {
        stl! {F[0, 10](G[0, 10](F[0, 10](G[0, 10](x > 0))))}
    }
    // ---
    // Signal Fixtures
    // ---

    #[fixture]
    #[once]
    fn signal_1() -> Vec<Step<f64>> {
        create_steps("x", vec![5.0, 4.0, 6.0, 2.0, 5.0], vec![0, 1, 2, 3, 4])
    }

    #[fixture]
    #[once]
    fn signal_2() -> Vec<Step<f64>> {
        create_steps(
            "x",
            vec![1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 1.0, 2.0],
            vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )
    }

    #[fixture]
    #[once]
    fn signal_3() -> Vec<Step<f64>> {
        create_steps(
            "x",
            vec![0.0, 6.0, 1.0, 0.0, 8.0, 1.0, 7.0],
            vec![0, 1, 2, 3, 4, 5, 6],
        )
    }

    #[fixture]
    #[once]
    fn signal_4() -> Vec<Step<f64>> {
        let x_steps = create_steps(
            "x",
            vec![0.0, 6.0, 1.0, 3.0, 8.0, 1.0, 7.0],
            vec![0, 1, 2, 3, 4, 5, 6],
        );
        let y_steps = create_steps(
            "y",
            vec![4.0, 3.0, 6.0, 7.0, 2.0, 1.0, 0.0],
            vec![0, 1, 2, 3, 4, 5, 6],
        );

        // Combine and sort the steps chronologically
        combine_and_sort_steps(vec![x_steps, y_steps])
    }

    #[fixture]
    #[once]
    fn monotonic_increasing() -> Vec<Step<f64>> {
        const N: usize = 51;
        (0..N)
            .map(|i| {
                let timestamp = Duration::from_secs(i as u64);
                let t = i as f64 / (N as f64 - 1.0);
                let value = -10.0 + 20.0 * t; // from -10 to 10
                Step::new("x", value, timestamp)
            })
            .collect()
    }
    #[fixture]
    #[once]
    fn monotonic_decreasing() -> Vec<Step<f64>> {
        const N: usize = 51;
        (0..N)
            .map(|i| {
                let timestamp = Duration::from_secs(i as u64);
                let t = i as f64 / (N as f64 - 1.0);
                let value = 10.0 - 20.0 * t; // from 10 to -10
                Step::new("x", value, timestamp)
            })
            .collect()
    }
    #[fixture]
    #[once]
    fn sinusoid() -> Vec<Step<f64>> {
        const N: usize = 51;
        (0..N)
            .map(|i| {
                let timestamp = Duration::from_secs(i as u64);
                let t = i as f64 / (N as f64 - 1.0);
                let value = 10.0 * (2.0 * PI * t).sin();
                Step::new("x", value, timestamp)
            })
            .collect()
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
        convert_f64_vec_to_bool_vec(exp_f1_s1_f64_strict())
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
                Step::new("output", Some(true), Duration::from_secs(0)),
                Step::new("output", Some(true), Duration::from_secs(1)),
                Step::new("output", Some(true), Duration::from_secs(2)),
                Step::new("output", Some(true), Duration::from_secs(3)),
            ],
            vec![
                Step::new("output", Some(false), Duration::from_secs(4)),
                Step::new("output", Some(false), Duration::from_secs(5)),
            ],
            vec![],
            vec![Step::new("output", Some(false), Duration::from_secs(6))],
            vec![Step::new("output", Some(false), Duration::from_secs(7))],
            vec![Step::new("output", Some(false), Duration::from_secs(8))],
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
        convert_f64_vec_to_bool_vec(exp_f3_s3_f64_strict())
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
        convert_f64_vec_to_bool_vec(exp_f4_s3_f64_strict())
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
            vec![Step::new("output", Some(-0.0), Duration::from_secs(1))],
            vec![Step::new("output", Some(0.0), Duration::from_secs(2))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(3))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(4))],
            vec![Step::new("output", Some(1.0), Duration::from_secs(5))],
        ]
    }
    fn exp_f6_s2_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![Step::new("output", Some(false), Duration::from_secs(0))],
            vec![Step::new("output", Some(true), Duration::from_secs(1))],
            vec![Step::new("output", Some(true), Duration::from_secs(2))],
            vec![Step::new("output", Some(true), Duration::from_secs(3))],
            vec![Step::new("output", Some(true), Duration::from_secs(4))],
            vec![Step::new("output", Some(true), Duration::from_secs(5))],
        ]
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
        convert_f64_vec_to_bool_vec(exp_f7_s3_f64_strict())
    }

    fn exp_f7_s3_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        exp_f7_s3_bool_strict()
    }

    fn exp_f8_s4_f64_strict() -> Vec<Vec<Step<Option<f64>>>> {
        vec![
            vec![],                                                        // x@t=0
            vec![],                                                        // y@t=0
            vec![],                                                        // x@t=1
            vec![],                                                        // y@t=1
            vec![Step::new("output", Some(0.0), Duration::from_secs(0))],  // x@t=2
            vec![],                                                        // y@t=2
            vec![Step::new("output", Some(1.0), Duration::from_secs(1))],  // x@t=3
            vec![],                                                        // y@t=3
            vec![Step::new("output", Some(-1.0), Duration::from_secs(2))], // x@t=4
            vec![],                                                        // y@t=5
            vec![Step::new("output", Some(-2.0), Duration::from_secs(3))], // x@t=5
            vec![],                                                        // y@t=4
            vec![Step::new("output", Some(1.0), Duration::from_secs(4))],  // x@t=6
            vec![],                                                        // y@t=6
        ]
    }

    fn exp_f8_s4_bool_strict() -> Vec<Vec<Step<Option<bool>>>> {
        convert_f64_vec_to_bool_vec(exp_f8_s4_f64_strict())
    }

    fn exp_f8_s4_bool_eager() -> Vec<Vec<Step<Option<bool>>>> {
        vec![
            vec![Step::new("output", Some(false), Duration::from_secs(0))], // x@t=0
            vec![],                                                         // y@t=0
            vec![],                                                         // x@t=1
            vec![],                                                         // y@t=1
            vec![],                                                         // x@t=2
            vec![],                                                         // y@t=2
            vec![
                Step::new("output", Some(true), Duration::from_secs(1)), // x@t=3
                Step::new("output", Some(false), Duration::from_secs(2)),
            ],
            vec![Step::new("output", Some(false), Duration::from_secs(3))], // y@t=3
            vec![],                                                         // x@t=4
            vec![],                                                         // y@t=4
            vec![],                                                         // x@t=5
            vec![],                                                         // y@t=5
            vec![Step::new("output", Some(true), Duration::from_secs(4))],  // x@t=6
            vec![],                                                         // y@t=6
        ]
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
                all_results.push(monitor.update(&step));
                println!(
                    "Step at {:?}, Monitor Output: {:?}",
                    step.timestamp,
                    all_results.last().unwrap()
                );
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
    #[case::f1_s1(vec![formula_1(), formula_1_alt(), formula_1_alt_2()], signal_1(), exp_f1_s1_f64_strict())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_f64_strict())]
    #[case::f3_s3(vec![formula_3(), formula_3_alt()], signal_3(), exp_f3_s3_f64_strict())]
    #[case::f6_s2(vec![formula_6(), formula_6_alt()], signal_2(), exp_f6_s2_f64_strict())]
    #[case::f4_s3(vec![formula_4(), formula_5(), formula_5_alt()], signal_3(), exp_f4_s3_f64_strict())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_f64_strict())]
    #[case::f8_s4(vec![formula_8()], signal_4(), exp_f8_s4_f64_strict())]
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
    #[case::f1_s1(vec![formula_1(), formula_1_alt(), formula_1_alt_2()], signal_1(), exp_f1_s1_bool_strict())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_bool_strict())]
    #[case::f3_s3(vec![formula_3(), formula_3_alt()], signal_3(), exp_f3_s3_bool_strict())]
    #[case::f6_s2(vec![formula_6(), formula_6_alt()], signal_2(), exp_f6_s2_bool_strict())]
    #[case::f4_s3(vec![formula_4(), formula_5(), formula_5_alt()], signal_3(), exp_f4_s3_bool_strict())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_bool_strict())]
    #[case::f8_s4(vec![formula_8()], signal_4(), exp_f8_s4_bool_strict())]
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
    #[case::f1_s1(vec![formula_1(), formula_1_alt(), formula_1_alt_2()], signal_1(), exp_f1_s1_bool_eager())]
    #[case::f2_s2(vec![formula_2()], signal_2(), exp_f2_s2_bool_eager())]
    #[case::f3_s3(vec![formula_3(), formula_3_alt()], signal_3(), exp_f3_s3_bool_eager())]
    #[case::f6_s2(vec![formula_6(), formula_6_alt()], signal_2(), exp_f6_s2_bool_eager())]
    #[case::f4_s3(vec![formula_4(), formula_5(), formula_5_alt()], signal_3(), exp_f4_s3_bool_eager())]
    #[case::f7_s3(vec![formula_7()], signal_3(), exp_f7_s3_bool_eager())]
    #[case::f8_s4(vec![formula_8()], signal_4(), exp_f8_s4_bool_eager())] // TODO: Add eager expected for f8_s4
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

    #[rstest]
    fn test_f64_interval_robustness() {
        // Test that StlMonitor can be built with f64 interval robustness
        let mut monitor: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
            .formula(
                stl! {
                    (G[0,2] (x < 2)) and ((x > 0))
                }
            )
            .strategy(MonitoringStrategy::Incremental)
            .evaluation_mode(EvaluationMode::Eager)
            .build()
            .unwrap();

        println!("Testing formula: {} \n", monitor.specification_to_string());
        // pass step to monitor to ensure it works
        let step = vec![
            Step::new("x", 1.0, Duration::from_secs(0)),
            Step::new("x", -1.0, Duration::from_secs(1)),
            Step::new("x", -1.0, Duration::from_secs(2)),
            // Step::new("x", -2.0, Duration::from_secs(6)),
            // Step::new("x", 2.0, Duration::from_secs(7)),
            // Step::new("x", 5.0, Duration::from_secs(8)),
        ];

        for s in step {
            let output = monitor.update(&s);
            println!(
                "Monitor output at {:?} with input: {:?}: \n {:?} \n",
                &s.timestamp, s.value, output
            );
        }
    }

    fn run_final_rosi_verdicts_check(
        formulas: Vec<FormulaDefinition>,
        signal: Vec<Step<f64>>,
    ) {
        let all_rosi_outputs: Vec<Vec<Step<Option<RobustnessInterval>>>> = formulas
            .into_iter()
            .map(|formula| {
                // Build an incremental, eager monitor that emits RobustnessInterval outputs
                let mut monitor: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
                    .formula(formula.clone())
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Eager)
                    .build()
                    .unwrap();

                // build a strict monitor for comparison
                let mut strict_monitor: StlMonitor<f64, f64> = StlMonitor::builder()
                    .formula(formula)
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Strict)
                    .build()
                    .unwrap();

                println!("Testing formula: {} \n", monitor.specification_to_string());

                signal
                    .iter()
                    .flat_map(|s| {
                        let rosi_outputs = monitor.update(&s);
                        let strict_outputs = strict_monitor.update(&s);

                        // Validate strict vs RoSI for the first (final) strict step, if present
                        if let Some(strict_step) = strict_outputs.first() {
                            if let Some(strict_val) = strict_step.value {
                                let rosi_step = rosi_outputs
                                    .iter()
                                    .find(|step| step.timestamp == strict_step.timestamp)
                                    .expect(&format!(
                                        "No RoSI step found for timestamp {:?}",
                                        strict_step.timestamp
                                    ));
                                if let Some(rosi_iv) = rosi_step.value {
                                    assert_eq!(
                                        strict_val, rosi_iv.0,
                                        "Final strict value {:?} not equal to RoSI lower bound {:?} for step at timestamp {:?}",
                                        strict_val, rosi_iv.0, s.timestamp
                                    );
                                    assert_eq!(
                                        strict_val, rosi_iv.1,
                                        "Final strict value {:?} not equal to RoSI upper bound {:?} for step at timestamp {:?}",
                                        strict_val, rosi_iv.1, s.timestamp
                                    );
                                }
                            }
                        }

                        rosi_outputs.into_iter()
                    })
                    .collect()
            })
            .collect();

        // equality check: compare every other vector against the first (equal is transitive)
        if let Some(first_out) = all_rosi_outputs.first() {
            for (i, rosi_vec) in all_rosi_outputs.iter().enumerate().skip(1) {
                assert_eq!(
                    first_out, rosi_vec,
                    "RoSI outputs differ between formulas at index 0 and {}",
                    i
                );
            }
        }
    }

    fn run_rosi_interval_bounds_check(
        formulas: Vec<FormulaDefinition>,
        signal: Vec<Step<f64>>,
    ) {
        let all_rosi_outputs: Vec<Vec<Step<Option<RobustnessInterval>>>> = formulas
            .into_iter()
            .map(|formula| {
                let mut monitor: StlMonitor<f64, RobustnessInterval> = StlMonitor::builder()
                    .formula(formula)
                    .strategy(MonitoringStrategy::Incremental)
                    .evaluation_mode(EvaluationMode::Eager)
                    .build()
                    .unwrap();

                // Track the last seen interval for each timestamp. As the monitor refines
                // partial/unknown intervals, the lower bound should never decrease and the
                // upper bound should never increase for a given timestamp.
                let mut last_intervals: HashMap<std::time::Duration, RobustnessInterval> = HashMap::new();

                let mut rosi_output: Vec<Step<Option<RobustnessInterval>>> = Vec::new();

                for s in signal.clone() {
                    let outputs = monitor.update(&s);

                    for out_step in outputs.iter() {
                        if let Some(iv) = out_step.value {
                            if let Some(prev) = last_intervals.get(&out_step.timestamp) {
                                // New lower bound must be >= previous lower bound
                                assert!(
                                    iv.0 >= prev.0,
                                    "Lower bound for timestamp {:?} decreased: previous={:?}, new={:?}",
                                    out_step.timestamp,
                                    prev,
                                    iv
                                );

                                // New upper bound must be <= previous upper bound
                                assert!(
                                    iv.1 <= prev.1,
                                    "Upper bound for timestamp {:?} increased: previous={:?}, new={:?}",
                                    out_step.timestamp,
                                    prev,
                                    iv
                                );
                            }
                            last_intervals.insert(out_step.timestamp, iv);
                        }
                    }

                    rosi_output.extend(outputs.clone());
                }

                rosi_output
            })
            .collect();

        assert!(!all_rosi_outputs.is_empty(), "No RoSI outputs collected");
        let first_out = &all_rosi_outputs[0];
        for (i, rosi_vec) in all_rosi_outputs.iter().enumerate().skip(1) {
            assert_eq!(
                first_out, rosi_vec,
                "RoSI outputs differ between formulas at index 0 and {}",
                i
            );
        }
    }

    #[rstest]
    #[case(vec![formula_1(), formula_1_alt(), formula_1_alt_2()])]
    #[case(vec![formula_2()])]
    #[case(vec![formula_3(), formula_3_alt()])]
    #[case(vec![formula_4()])]
    #[case(vec![formula_5(), formula_5_alt()])]
    #[case(vec![formula_6(), formula_6_alt()])]
    #[case(vec![formula_7()])]
    #[case(vec![formula_8()])]
    #[case(vec![formula_9()])]
    fn test_final_rosi_verdicts(
        #[case] formulas: Vec<FormulaDefinition>,
        #[values(monotonic_increasing(), monotonic_decreasing(), sinusoid())] signal: Vec<
            Step<f64>,
        >,
    ) {
        run_final_rosi_verdicts_check(formulas, signal);
    }

    #[rstest]
    #[case(vec![formula_1(), formula_1_alt(), formula_1_alt_2()])]
    #[case(vec![formula_2()])]
    #[case(vec![formula_3(), formula_3_alt()])]
    #[case(vec![formula_4()])]
    #[case(vec![formula_5(), formula_5_alt()])]
    #[case(vec![formula_6(), formula_6_alt()])]
    #[case(vec![formula_7()])]
    #[case(vec![formula_8()])]
    #[case(vec![formula_9()])]
    fn test_rosi_interval_bounds(
        #[case] formulas: Vec<FormulaDefinition>,
        #[values(monotonic_increasing(), monotonic_decreasing(), sinusoid())] signal: Vec<
            Step<f64>,
        >,
    ) {
        run_rosi_interval_bounds_check(formulas, signal);
    }

    #[rstest]
    fn test_library_formulas_rosi(
        #[values(monotonic_increasing(), monotonic_decreasing(), sinusoid())] signal: Vec<Step<f64>>,
    ) {
        let lib_formulas = ostl::stl::formulas::get_formulas(&[]);
        for (id, formula) in lib_formulas {
            println!("Testing library formula id: {}", id);
            run_final_rosi_verdicts_check(vec![formula.clone()], signal.clone());
            run_rosi_interval_bounds_check(vec![formula], signal.clone());
        }
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
