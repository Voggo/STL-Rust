use ostl::ring_buffer::Step;
use ostl::stl;
use ostl::stl::core::Variables;
use ostl::stl::monitor::{Algorithm, StlMonitor, semantic_markers};
// use ostl::synchronizer::SynchronizationStrategy;
use std::time::Duration;

fn main() {
    let vars = Variables::new();
    vars.set("B", 0.0);

    let f = stl!(F[0,5](x > $B));

    println!("Formula Structure:\n{}", f.to_tree_string(2));

    let mut monitor = StlMonitor::builder()
        .formula(f)
        .algorithm(Algorithm::Incremental)
        .semantics(semantic_markers::EagerSatisfaction)
        .variables(vars.clone())
        .build()
        .unwrap();

    let res = monitor.update(&Step::new("x", 1.0, Duration::from_secs(0)));
    println!("Monitor Output:\n{}", res);
    let res = monitor.update(&Step::new("x", 2.0, Duration::from_secs(1)));
    println!("Monitor Output:\n{}", res);
    let res = monitor.update(&Step::new("x", -1.0, Duration::from_secs(2)));
    println!("Monitor Output:\n{}", res);
    vars.set("B", -2.0);
    let res = monitor.update(&Step::new("x", -3.0, Duration::from_secs(3)));
    println!("Monitor Output:\n{}", res);
    let res = monitor.update(&Step::new("x", 3.0, Duration::from_secs(4)));
    println!("Monitor Output:\n{}", res);

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
