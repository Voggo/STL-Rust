use ostl::ring_buffer::Step;

use ostl::stl::core::{TimeInterval};

use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};

use rand::rng;
use rand::rngs::ThreadRng;
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
    // anomaly configuration
    anomaly_at: Option<Duration>,
    anomaly_applied: bool,
    original_mean: f64,
    original_std: f64,
    anomaly_mean_shift: f64,
    anomaly_std_multiplier: f64,
}

impl SignalStepGenerator {
    fn new(
        signal_name: &'static str,
        mean: f64,
        std_dev: f64,
        jitter: f64,
        anomaly_at: Option<Duration>,
        anomaly_mean_shift: f64,
        anomaly_std_multiplier: f64,
    ) -> Self {
        SignalStepGenerator {
            signal_name: signal_name,

            current_time: Duration::from_secs(0),

            rng: rng(),

            normal_dist: Normal::new(mean, std_dev).unwrap(),

            jitter: Normal::new(0.0, jitter).unwrap(),

            anomaly_at,
            anomaly_applied: false,
            original_mean: mean,
            original_std: std_dev,
            anomaly_mean_shift,
            anomaly_std_multiplier,
        }
    }

    fn next_step(&mut self) -> Step<f64> {
        // If an anomaly is scheduled and time has been reached, update distribution once
        if !self.anomaly_applied {
            if let Some(at) = self.anomaly_at {
                if self.current_time >= at {
                    // compute new parameters
                    let new_mean = self.original_mean + self.anomaly_mean_shift;
                    let new_std = (self.original_std * self.anomaly_std_multiplier).max(1e-12);
                    self.normal_dist = Normal::new(new_mean, new_std).unwrap();
                    self.anomaly_applied = true;
                    println!(
                        "Anomaly applied at {:?}: mean -> {:.3} (shift {:.3}), std -> {:.3} (mult {:.3})",
                        self.current_time,
                        new_mean,
                        self.anomaly_mean_shift,
                        new_std,
                        self.anomaly_std_multiplier
                    );
                }
            }
        }

        let value = self.normal_dist.sample(&mut self.rng);

        let step = Step::new(self.signal_name, value, self.current_time);

        self.current_time += Duration::from_secs_f64(1. + self.jitter.sample(&mut self.rng));

        step
    }
}

fn get_formula() -> FormulaDefinition {
    // G[0,20] ( (x < 5 && x > -5) && (x > 1 || x < -1) ) -> F[0,5] ( G[0,2] (x < 1 && x > -1) )
    FormulaDefinition::Globally(
        TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(20),
        },
        Box::new(FormulaDefinition::Implies(
            Box::new(FormulaDefinition::And(
                Box::new(FormulaDefinition::And(
                    Box::new(FormulaDefinition::LessThan("x", 5.0)),
                    Box::new(FormulaDefinition::GreaterThan("x", -5.0)),
                )),
                Box::new(FormulaDefinition::Or(
                    Box::new(FormulaDefinition::GreaterThan("x", 1.0)),
                    Box::new(FormulaDefinition::LessThan("x", -1.0)),
                )),
            )),
            Box::new(FormulaDefinition::Eventually(
                TimeInterval {
                    start: Duration::from_secs(0),
                    end: Duration::from_secs(5),
                },
                Box::new(FormulaDefinition::Globally(
                    TimeInterval {
                        start: Duration::from_secs(0),
                        end: Duration::from_secs(2),
                    },
                    Box::new(FormulaDefinition::And(
                        Box::new(FormulaDefinition::LessThan("x", 1.0)),
                        Box::new(FormulaDefinition::GreaterThan("x", -1.0)),
                    )),
                )),
            )),
        )),
    )
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
    // FormulaDefinition::Globally(
    //     TimeInterval {
    //         start: Duration::from_secs(0),

    //         end: Duration::from_secs(20),
    //     },
    //     Box::new(FormulaDefinition::And(
    //         Box::new(FormulaDefinition::GreaterThan("x", 0.0)),
    //         Box::new(FormulaDefinition::LessThan("x", 6.0)),
    //     )),
    // )
}

fn get_eventually_formula() -> FormulaDefinition {
    // F[0,20] (G[0,4] (x < 0.5 && x > -0.5))
    FormulaDefinition::Eventually(
        TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(20),
        },
        Box::new(FormulaDefinition::Globally(
            TimeInterval {
                start: Duration::from_secs(0),
                end: Duration::from_secs(4),
            },
            Box::new(FormulaDefinition::And(
                Box::new(FormulaDefinition::LessThan("x", 0.5)),
                Box::new(FormulaDefinition::GreaterThan("x", -0.5)),
            )),
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

    // Build a second monitor that checks an "eventually" property on the same signal
    let mut monitor_eventually: StlMonitor<f64, bool> = StlMonitor::builder()
        .formula(get_eventually_formula())
        .strategy(MonitoringStrategy::Incremental)
        .evaluation_mode(EvaluationMode::Eager)
        .build()
        .unwrap();

    println!("Monitoring formula: {}", monitor.specification_to_string());

    // Basic CLI parsing for anomaly injection (no external crates required)
    let mut anomaly_time: Option<Duration> = None;
    let mut anomaly_mean_shift: f64 = 0.0;
    let mut anomaly_std_multiplier: f64 = 1.0;
    // MQTT broker port (default 1883)
    let mut broker_port: u16 = 1883;

    let argv: Vec<String> = std::env::args().collect();
    let mut idx = 1usize;
    while idx < argv.len() {
        match argv[idx].as_str() {
            "--anomaly-time" => {
                if idx + 1 < argv.len() {
                    if let Ok(secs) = argv[idx + 1].parse::<f64>() {
                        anomaly_time = Some(Duration::from_secs_f64(secs));
                    }
                    idx += 1;
                }
            }
            "--anomaly-mean-shift" => {
                if idx + 1 < argv.len() {
                    if let Ok(v) = argv[idx + 1].parse::<f64>() {
                        anomaly_mean_shift = v;
                    }
                    idx += 1;
                }
            }
            "--anomaly-std-multiplier" => {
                if idx + 1 < argv.len() {
                    if let Ok(v) = argv[idx + 1].parse::<f64>() {
                        anomaly_std_multiplier = v;
                    }
                    idx += 1;
                }
            }
            "--port" => {
                if idx + 1 < argv.len() {
                    if let Ok(p) = argv[idx + 1].parse::<u16>() {
                        broker_port = p;
                    }
                    idx += 1;
                }
            }
            _ => {}
        }
        idx += 1;
    }

    if anomaly_time.is_some() {
        println!(
            "Configured anomaly at {:?}: mean_shift={}, std_multiplier={}",
            anomaly_time, anomaly_mean_shift, anomaly_std_multiplier
        );
    }

    // MQTT setup
    use rumqttc::{Client, MqttOptions, QoS};

    // broker port configured via CLI (default 1883)
    let mut mqttoptions = MqttOptions::new("stl_monitor_demo", "localhost", broker_port);
    mqttoptions.set_keep_alive(Duration::from_secs(5));
    let (client, mut connection) = Client::new(mqttoptions, 10);

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

    

    let mut signal_generator_x = SignalStepGenerator::new(
        "x",
        0.0,
        0.25,
        0.1,
        anomaly_time,
        anomaly_mean_shift,
        anomaly_std_multiplier,
    );

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

        // Update monitors with x and collect robustness outputs from both
        let result_x = monitor.update(&step_x);
        let result_evt = monitor_eventually.update(&step_x);

        // Combine outputs from both monitors and publish
        let mut outputs = Vec::new();
        for out in result_x.iter() {
            let entry = OutputEntry {
                // prefix signal name so the plot can distinguish monitors
                signal: format!("spec1_/{}/{}", monitor.specification_to_string(), out.signal),
                value: out.value.map(|v| if v { 1.0 } else { 0.0 }),
                timestamp_secs: out.timestamp.as_secs_f64(),
            };
            outputs.push(entry);
        }
        for out in result_evt.iter() {
            let entry = OutputEntry {
                signal: format!("spec2_/{}/{}", monitor_eventually.specification_to_string(), out.signal),
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
            "At time {:?}, x: {:.2}, robustness_spec1: {:?}, robustness_spec2: {:?}",
            step_x.timestamp, step_x.value, result_x, result_evt
        );

        sleep(Duration::from_millis(750));
    }
}
