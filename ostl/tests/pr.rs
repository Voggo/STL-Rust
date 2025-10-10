#[cfg(test)]
mod tests {
    use ostl::ring_buffer::Step;
    use ostl::stl::{
        core::{RobustnessSemantics, TimeInterval},
        monitor::{FormulaDefinition, MonitoringStrategy, StlMonitor},
    };
    use proptest::prelude::*;
    use std::fmt::{Debug, Display};
    use std::time::Duration;

    fn build_monitor<T, Y>(formula: FormulaDefinition) -> StlMonitor<T, Y>
    where
        T: Clone + Copy + 'static + Into<f64>,
        Y: RobustnessSemantics + 'static + Copy + Debug + PartialEq + Default + Display,
    {
        StlMonitor::builder()
            .formula(formula)
            .strategy(MonitoringStrategy::Incremental)
            .build()
            .unwrap()
    }

    fn run_multi_step_robustness_test<T, Y>(mut monitor_opt: StlMonitor<T, Y>, steps: &[Step<T>])
    where
        T: Clone,
        Y: RobustnessSemantics + Copy + Debug + PartialEq,
    {
        println!("Running multi-step robustness test with spec: {}", monitor_opt.specification_to_string());
        for step in steps.iter() {
            let _robustness_opt = monitor_opt.instantaneous_robustness(step);
        }
    }

    /// arb_step generates a single Step<f64> with random value and timestamp.
    fn arb_step() -> impl Strategy<Value = Step<f64>> {
        (0.0..100.0, 0..100u64).prop_map(|(value, timestamp)| Step {
            value,
            timestamp: Duration::from_secs(timestamp),
        })
    }

    /// arb_steps generates a vector of Steps with unique, sorted timestamps (max 100 steps).
    fn arb_steps(max_steps: usize) -> impl Strategy<Value = Vec<Step<f64>>> {
        prop::collection::vec(arb_step(), 0..=max_steps).prop_map(|mut steps| {
            steps.sort_by_key(|s| s.timestamp);
            steps.dedup_by_key(|s| s.timestamp);
            steps
        })
    }

    /// arb_atomic generates atomic formula definitions.
    fn arb_atomic() -> impl Strategy<Value = FormulaDefinition> {
        prop_oneof![
            (-50.0..50.0).prop_map(FormulaDefinition::GreaterThan),
            (-50.0..50.0).prop_map(FormulaDefinition::LessThan),
            Just(FormulaDefinition::True),
            Just(FormulaDefinition::False),
        ]
    }

    /// arb_temporal_operators generates temporal formula definitions.
    fn arb_temporal_operators() -> impl Strategy<Value = FormulaDefinition> {
        (0u64..20, 21u64..50).prop_flat_map(|(start, end)| {
            let interval = TimeInterval {
                start: Duration::from_secs(start),
                end: Duration::from_secs(end),
            };
            let subformula = Box::new(FormulaDefinition::GreaterThan(50.0));

            prop_oneof![
                Just(FormulaDefinition::Eventually(interval, subformula.clone())),
                Just(FormulaDefinition::Globally(interval, subformula.clone())),
                Just(FormulaDefinition::Until(
                    interval,
                    subformula.clone(),
                    subformula
                )),
            ]
        })
    }

    /// arb_formula_recursive generates complex formula definitions recursively up to a certain depth.
    fn arb_formula_recursive(depth: u32) -> BoxedStrategy<FormulaDefinition> {
        if depth == 0 {
            return arb_atomic().boxed();
        }

        let leaf = arb_formula_recursive(depth - 1);

        prop_oneof![
            arb_atomic(),
            arb_temporal_operators(),
            (leaf.clone(), leaf.clone())
                .prop_map(|(l, r)| FormulaDefinition::And(Box::new(l), Box::new(r))),
            (leaf.clone(), leaf.clone())
                .prop_map(|(l, r)| FormulaDefinition::Or(Box::new(l), Box::new(r))),
            leaf.clone()
                .prop_map(|op| FormulaDefinition::Not(Box::new(op))),
            (leaf.clone(), leaf)
                .prop_map(|(l, r)| FormulaDefinition::Implies(Box::new(l), Box::new(r))),
        ]
        .boxed()
    }

    proptest! {
        #[test]
        fn pt_nested_operators(steps in arb_steps(100), formula in arb_formula_recursive(5)) {
            let monitor_opt = build_monitor::<f64, f64>(formula);
            run_multi_step_robustness_test(monitor_opt, &steps);
        }
    }
}
