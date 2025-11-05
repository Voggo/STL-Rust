use ostl::ring_buffer::Step;

use ostl::stl::core::{RobustnessInterval, TimeInterval};

use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};

use rand::prelude::*;

use rand_distr::{Distribution, Normal};

use std::thread::sleep;

use std::time::Duration;

struct SignalStepGenerator {
    signal_name: &'static str,

    current_time: Duration,

    rng: ThreadRng,

    normal_dist: Normal<f64>,

    jitter: Normal<f64>,
}

impl SignalStepGenerator {
    fn new(signal_name: &'static str, mean: f64, std_dev: f64, jitter: f64) -> Self {
        SignalStepGenerator {
            signal_name: signal_name,

            current_time: Duration::from_secs(0),

            rng: rand::rng(),

            normal_dist: Normal::new(mean, std_dev).unwrap(),

            jitter: Normal::new(0.0, jitter).unwrap(),
        }
    }

    fn next_step(&mut self) -> Step<f64> {
        let value = self.normal_dist.sample(&mut self.rng);

        let step = Step::new(self.signal_name, value, self.current_time);

        self.current_time += Duration::from_secs_f64(1. + self.jitter.sample(&mut self.rng));

        step
    }
}

fn get_formula() -> FormulaDefinition {
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
            Box::new(FormulaDefinition::GreaterThan("x", 0.0)),
        )),
        Box::new(FormulaDefinition::Eventually(
            TimeInterval {
                start: Duration::from_secs(0),

                end: Duration::from_secs(2),
            },
            Box::new(FormulaDefinition::GreaterThan("y", 3.0)),
        )),
    )
}

fn main() {
    let formula = get_formula();

    let mut monitor: StlMonitor<f64, f64> = StlMonitor::builder()
        .formula(formula)
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Strict)
        .build()
        .unwrap();

    println!("Monitoring formula: {}", monitor.specification_to_string());

    let mut signal_generator_x = SignalStepGenerator::new("x", 2.0, 1.0, 0.1);

    let mut signal_generator_y = SignalStepGenerator::new("y", 4.0, 4.0, 0.1);

    loop {
        let step_x = signal_generator_x.next_step();

        let step_y = signal_generator_y.next_step();

        let result_x = monitor.update(&step_x);

        println!(
            "At time {:?}, signal value: {:.2}, robustness: {:?}",
            step_x.timestamp, step_x.value, result_x
        );

        let result_y = monitor.update(&step_y);

        println!(
            "At time {:?}, signal value: {:.2}, robustness: {:?}",
            step_y.timestamp, step_y.value, result_y
        );

        sleep(Duration::from_millis(1000));
    }
}
