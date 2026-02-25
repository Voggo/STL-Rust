use ostl::ring_buffer::Step;
use ostl::stl;
use ostl::stl::core::Variables;
use ostl::stl::monitor::{Algorithm, StlMonitor, semantic_markers};
use ostl::synchronizer::SynchronizationStrategy;
// use std::collections::HashMap;
use std::time::Duration;

fn main() {
    let phi1 = stl!(G[0, 5] temp < $MAX_TEMP);
    let phi2 = stl!(pressure > 10.0 -> F[0, 2] valve_open == 1.0);
    let phi = stl!(phi1 and phi2);

    println!("Formula Structure:\n{}", phi.to_tree_string(2));

    let vars = Variables::new();
    vars.set("MAX_TEMP", 120.0);

    let mut monitor = StlMonitor::builder()
        .formula(phi)
        .semantics(semantic_markers::Rosi)
        .algorithm(Algorithm::Incremental)
        .synchronization_strategy(SynchronizationStrategy::ZeroOrderHold)
        .variables(vars.clone())
        .build()
        .expect("Failed to build STL monitor");

    println!("{}", monitor);

    monitor.update(&Step::new("temp", 125.5, Duration::from_secs(0)));
    monitor.update(&Step::new("pressure", 15.0, Duration::from_secs(0)));
    let res = monitor.update(&Step::new("valve_open", 1.0, Duration::from_secs(0)));

    // let batch = HashMap::from([
    //     (
    //         "temp",
    //         vec![
    //             Step::new("temp", 125.5, Duration::from_secs(0)),
    //             Step::new("temp", 115.0, Duration::from_secs(1)),
    //             ...
    //         ],
    //     ),
    //     (
    //         "pressure",
    //         vec![
    //             Step::new("pressure", 15.0, Duration::from_secs(0)),
    //             Step::new("pressure", 8.0, Duration::from_secs(1)),
    //             ...
    //         ],
    //     ),
    //     (
    //         "valve_open",
    //         vec![
    //             Step::new("valve_open", 1.0, Duration::from_secs(0)),
    //             Step::new("valve_open", 0.0, Duration::from_secs(1)),
    //             ...
    //         ],
    //     ),
    // ]);
    // let res = monitor.update_batch(&batch);

    println!("Monitor Output (Display) :\n{}", res);
    println!("Monitor Output (Debug) :\n{:?}", res);
    // let res = monitor.update(&Step::new("x", 2.0, Duration::from_secs(1)));
    // println!("Monitor Output:\n{}", res);
    // let res = monitor.update(&Step::new("x", -1.0, Duration::from_secs(2)));
    // println!("Monitor Output:\n{}", res);
    // vars.set("B", -2.0);
    // let res = monitor.update(&Step::new("x", -3.0, Duration::from_secs(3)));
    // println!("Monitor Output:\n{}", res);
    // let res = monitor.update(&Step::new("x", 3.0, Duration::from_secs(4)));
    // println!("Monitor Output:\n{}", res);

    // let x = vec![
    //     Step::new("x", -1.0, Duration::from_secs(0)),
    //     Step::new("y", -2.0, Duration::from_secs(0)),
    //     Step::new("y", -3.0, Duration::from_secs(1)),
    //     Step::new("x", -4.0, Duration::from_secs(2)),
    //     Step::new("x", -5.0, Duration::from_secs(4)),
    //     Step::new("x", -6.0, Duration::from_secs(6)),
    //     Step::new("x", -7.0, Duration::from_secs(8)),
    //     Step::new("y", -8.0, Duration::from_secs(9)),
    //     Step::new("x", -9.0, Duration::from_secs(10)),
    //     Step::new("y", 1.0, Duration::from_secs(10)),
    // ];

    // for step in x {
    //     println!(
    //         "INPUT STEP AT TIME: {:?} ({:?})",
    //         step.timestamp, step.signal
    //     );
    //     let monitor_output = monitor.update(&step);
    //     println!("{} \n", monitor_output);
    //     println!("-------------------------");
    // }
}
