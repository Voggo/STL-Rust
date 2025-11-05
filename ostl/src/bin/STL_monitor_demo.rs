use ostl::ring_buffer::Step;

use ostl::stl::core::{RobustnessInterval, TimeInterval};

use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};

use rand::rngs::ThreadRng;
use rand::rng;
use rand_distr::{Distribution, Normal};

use std::thread::sleep;

use std::time::Duration;

// MQTT client will be imported in main where used

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

            rng: rng(),

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

    // FormulaDefinition::Until(
    //     TimeInterval {
    //         start: Duration::from_secs(0),

    //         end: Duration::from_secs(6),
    //     },
    //     Box::new(FormulaDefinition::Globally(
    //         TimeInterval {
    //             start: Duration::from_secs(0),

    //             end: Duration::from_secs(2),
    //         },
    //         Box::new(FormulaDefinition::GreaterThan("x", 0.0)),
    //     )),
    //     Box::new(FormulaDefinition::Eventually(
    //         TimeInterval {
    //             start: Duration::from_secs(0),

    //             end: Duration::from_secs(2),
    //         },
    //         Box::new(FormulaDefinition::GreaterThan("y", 3.0)),
    //     )),
    // )
    // G[0,100] (x>0 && x < 6)
    FormulaDefinition::Globally(
        TimeInterval {
            start: Duration::from_secs(0),

            end: Duration::from_secs(100),
        },
        Box::new(FormulaDefinition::And(
            Box::new(FormulaDefinition::GreaterThan("x", 0.0)),
            Box::new(FormulaDefinition::LessThan("x", 6.0)),
        )),
    )
}

fn main() {
    let formula = get_formula();

    // Build the monitor
    let mut monitor: StlMonitor<f64, bool> = StlMonitor::builder()
        .formula(formula)
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Eager)
        .build()
        .unwrap();

    println!("Monitoring formula: {}", monitor.specification_to_string());

    // MQTT setup
    use rumqttc::{Client, MqttOptions, QoS};

    let mut mqttoptions = MqttOptions::new("stl_monitor_demo", "localhost", 1883);
    mqttoptions.set_keep_alive(Duration::from_secs(5));
    let (mut client, mut connection) = Client::new(mqttoptions, 10);

    // Spawn a background thread to poll the connection so outgoing packets are processed
    std::thread::spawn(move || {
        for event in connection.iter() {
            // We intentionally ignore most events; printing can be uncommented for debugging
            // e.g., println!("MQTT event: {:?}", event);
            let _ = event;
        }
    });

    // Helper message structs for serialization
    #[derive(serde::Serialize)]
    struct SignalMessage {
        signal: &'static str,
        value: f64,
        timestamp_secs: f64,
    }

    #[derive(serde::Serialize)]
    struct RobustnessMessage {
        outputs: Vec<OutputEntry>,
        timestamp_secs: f64,
    }

    #[derive(serde::Serialize)]
    struct OutputEntry {
        signal: String,
        value: Option<f64>,
        timestamp_secs: f64,
    }

    let mut signal_generator_x = SignalStepGenerator::new("x", 0.0, 0.1, 0.1);

    loop {
        let step_x = signal_generator_x.next_step();

        // Publish the generated step for x
        let msg_x = SignalMessage {
            signal: step_x.signal,
            value: step_x.value,
            timestamp_secs: step_x.timestamp.as_secs_f64(),
        };
        let payload_x = serde_json::to_vec(&msg_x).expect("serialize step_x");
        client
            .publish("stl/signals/x", QoS::AtLeastOnce, false, payload_x)
            .expect("publish x");

        // Update monitor with x only and publish robustness outputs
        let result_x = monitor.update(&step_x);

        // Combine outputs and publish (only from result_x)
        let mut outputs = Vec::new();
        for out in result_x.iter() {
            let entry = OutputEntry {
                signal: out.signal.to_string(),
                value: out.value.map(|v| if v { 1.0 } else { 0.0 }),
                timestamp_secs: out.timestamp.as_secs_f64(),
            };
            outputs.push(entry);
        }

        if !outputs.is_empty() {
            // Use the latest timestamp among output entries as the message timestamp
            let latest_ts = outputs
                .iter()
                .map(|o| o.timestamp_secs)
                .fold(0.0_f64, |a, b| a.max(b));
            let robustness_msg = RobustnessMessage {
                outputs,
                timestamp_secs: latest_ts,
            };
            let payload = serde_json::to_vec(&robustness_msg).expect("serialize robustness");
            client
                .publish("stl/robustness", QoS::AtLeastOnce, false, payload)
                .expect("publish robustness");
        }

        println!(
            "At time {:?}, x: {:.2}, robustness: {:?}",
            step_x.timestamp, step_x.value, result_x
        );

        sleep(Duration::from_millis(500));
    }
}
