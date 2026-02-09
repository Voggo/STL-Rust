#[cfg(test)]
mod common;
mod fixtures;

use fixtures::formulas::*;
use fixtures::signals::*;
use ostl::ring_buffer::Step;
use ostl::stl::core::RobustnessInterval;
use ostl::stl::formula_definition::FormulaDefinition;
use ostl::stl::monitor::{Algorithm, Robustness, Rosi, StlMonitor};
use pretty_assertions::assert_eq;
use rstest::rstest;
use std::collections::HashMap;
use std::vec;

fn run_final_rosi_verdicts_check(formulas: Vec<FormulaDefinition>, signal: Vec<Step<f64>>) {
    let all_rosi_outputs: Vec<Vec<Step<RobustnessInterval>>> = formulas
            .into_iter()
            .map(|formula| {
                // Build an incremental monitor that emits RobustnessInterval outputs
                let mut monitor = StlMonitor::builder()
                    .formula(formula.clone())
                    .semantics(Rosi)
                    .algorithm(Algorithm::Incremental)
                    .build()
                    .unwrap();

                // build a robustness monitor for comparison
                let mut strict_monitor = StlMonitor::builder()
                    .formula(formula.clone())
                    .semantics(Robustness)
                    .algorithm(Algorithm::Incremental)
                    .build()
                    .unwrap();

                println!("Testing formula:\n{} \n", formula.to_tree_string(2));

                signal
                    .iter()
                    .flat_map(|s| {
                        let rosi_output = monitor.update(s);
                        let strict_output = strict_monitor.update(s);
                        let rosi_outputs = rosi_output.all_outputs();
                        let strict_outputs = strict_output.all_outputs();

                        // Validate strict vs RoSI for the first (final) strict step, if present
                        if let Some(strict_step) = strict_outputs.first() {
                            let strict_val = strict_step.value;
                            let rosi_step = rosi_outputs
                                .iter()
                                .find(|step| step.timestamp == strict_step.timestamp)
                                .unwrap_or_else(|| panic!("No RoSI step found for timestamp {:?} at input step {:?}",
                                    strict_step.timestamp, s.timestamp));
                            let rosi_iv = rosi_step.value;
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

fn run_rosi_interval_bounds_check(formulas: Vec<FormulaDefinition>, signal: Vec<Step<f64>>) {
    let all_rosi_outputs: Vec<Vec<Step<RobustnessInterval>>> = formulas
        .into_iter()
        .map(|formula| {
            let mut monitor = StlMonitor::builder()
                .formula(formula)
                .semantics(Rosi)
                .algorithm(Algorithm::Incremental)
                .build()
                .unwrap();

            // Track the last seen interval for each timestamp. As the monitor refines
            // partial/unknown intervals, the lower bound should never decrease and the
            // upper bound should never increase for a given timestamp.
            let mut last_intervals: HashMap<std::time::Duration, RobustnessInterval> =
                HashMap::new();

            let mut rosi_output: Vec<Step<RobustnessInterval>> = Vec::new();

            for s in signal.clone() {
                let output = monitor.update(&s);

                for out_step in output.outputs_iter() {
                    let iv = out_step.value;
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

                rosi_output.extend(output.all_outputs());
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
    #[values(monotonic_increasing(), monotonic_decreasing(), sinusoid())] signal: Vec<Step<f64>>,
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
    #[values(monotonic_increasing(), monotonic_decreasing(), sinusoid())] signal: Vec<Step<f64>>,
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
