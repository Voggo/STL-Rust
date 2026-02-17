use ostl::ring_buffer::Step;
use ostl::stl::monitor::{DelayedQuantitative, StlMonitor};
use ostl::stl::parse_stl;
use std::time::Duration;

fn main() {
    // Parse a simple formula
    let formula = parse_stl("x > 5.0").unwrap();

    // Build the monitor with robustness semantics
    let mut monitor = StlMonitor::builder()
        .formula(formula)
        .semantics(DelayedQuantitative)
        .build()
        .unwrap();

    println!("=== Testing Timestamp Validation ===\n");

    // Valid step at t=1
    println!("1. Sending step at t=1s with value 10.0");
    let step1 = Step::new("x", 10.0, Duration::from_secs(1));
    let output1 = monitor.update(&step1);
    println!("   Outputs: {}", output1.total_outputs());

    // Valid step at t=2 (strictly increasing)
    println!("\n2. Sending step at t=2s with value 8.0 (valid - strictly increasing)");
    let step2 = Step::new("x", 8.0, Duration::from_secs(2));
    let output2 = monitor.update(&step2);
    println!("   Outputs: {}", output2.total_outputs());

    // Invalid step at t=2 (equal to previous)
    println!("\n3. Sending step at t=2s with value 12.0 (INVALID - equal timestamp)");
    let step3 = Step::new("x", 12.0, Duration::from_secs(2));
    let output3 = monitor.update(&step3);
    println!("   Outputs: {} (step was ignored)", output3.total_outputs());

    // Invalid step at t=1 (decreasing)
    println!("\n4. Sending step at t=1s with value 15.0 (INVALID - decreasing timestamp)");
    let step4 = Step::new("x", 15.0, Duration::from_secs(1));
    let output4 = monitor.update(&step4);
    println!("   Outputs: {} (step was ignored)", output4.total_outputs());

    // Valid step at t=5 (strictly increasing again)
    println!("\n5. Sending step at t=5s with value 20.0 (valid - strictly increasing)");
    let step5 = Step::new("x", 20.0, Duration::from_secs(5));
    let output5 = monitor.update(&step5);
    println!("   Outputs: {}", output5.total_outputs());

    println!("\n=== Test Complete ===");
    println!(
        "\nNote: Warning messages are printed to stderr when invalid timestamps are detected."
    );
}
