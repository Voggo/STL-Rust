use ostl::ring_buffer::Step;
use ostl::stl;
use ostl::stl::core::RobustnessInterval;
use ostl::stl::monitor::{EvaluationMode, MonitoringStrategy, StlMonitor};
use std::time::Duration;

fn main() {
    // let f = stl!(
    //     eventually [0, 2] (x > 2) or eventually [0, 3] (y > 0)
    // );
    let f = stl!(
        x > 0 and x <10
    );
    // let f = stl!(
    //     (F[0, 10]((F[0, 10](x > 0)) && (F[0, 10](x > 0)))) && (F[0, 10]((F[0, 10](x > 0)) && (F[0, 10](x > 0))))
    // );

    println!("Formula Structure:\n{}", f.to_tree_string(2));

    let mut monitor = StlMonitor::<f64, RobustnessInterval>::builder()
        .formula(f)
        .evaluation_mode(EvaluationMode::Eager)
        .strategy(MonitoringStrategy::Incremental)
        .build()
        .unwrap();
    let x = vec![
        Step::new("x", -1.0, Duration::from_secs(0)),
        Step::new("y", -2.0, Duration::from_secs(0)),
        Step::new("y", -3.0, Duration::from_secs(1)),
        Step::new("x", -4.0, Duration::from_secs(2)),
        Step::new("x", -5.0, Duration::from_secs(4)),
        Step::new("x", -6.0, Duration::from_secs(6)),
        Step::new("x", -7.0, Duration::from_secs(8)),
        Step::new("y", -8.0, Duration::from_secs(9)),
        Step::new("x", -9.0, Duration::from_secs(10)),
        Step::new("y", 1.0, Duration::from_secs(10)),
    ];

    // let x = sinusoid();

    for step in x {
        // println!("INPUT STEP AT TIME: {:?} ({:?})", step.timestamp, step.signal);
        let monitor_output = monitor.update(&step);
        println!("{:?}", monitor_output.all_outputs());
        println!("-------------------------");
    }
}
