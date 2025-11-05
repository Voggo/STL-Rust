use ostl::ring_buffer::Step;
use ostl::stl::core::TimeInterval;
use ostl::stl::monitor::{EvaluationMode, FormulaDefinition, MonitoringStrategy, StlMonitor};
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use std::collections::VecDeque;
use std::error::Error;
use std::time::Duration;

// --- Plotters and Piston Imports ---
use piston_window::{EventLoop, PistonWindow, WindowSettings};
use plotters::prelude::*;
use plotters_piston::draw_piston_window;

// --- Configuration Constants ---
/// How many data points to show on the chart at once
const N_DATA_POINTS: usize = 100;
/// Target frames per second (updates per second)
const FPS: u64 = 10;

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
        // Use a fixed time step for real-time simulation, jitter is less relevant here
        self.current_time += Duration::from_secs_f64(1.0 / FPS as f64);
        step
    }
}

fn get_formula() -> FormulaDefinition {
    // (G[0,2] (x > 0)) U[0,6] (F[0,2] (y > 3))
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

/// Helper function to find min/max in a VecDeque, or a default range
fn get_data_range(data: &VecDeque<f64>) -> (f64, f64) {
    let mut min_val = f64::INFINITY;
    let mut max_val = f64::NEG_INFINITY;
    for &v in data.iter() {
        min_val = min_val.min(v);
        max_val = max_val.max(v);
    }
    if min_val.is_infinite() {
        (-1.0, 1.0) // Default if empty
    } else {
        (min_val - 1.0, max_val + 1.0) // Add some padding
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut window: PistonWindow =
        WindowSettings::new("Real-time STL Robustness", [1024, 768])
            .samples(4)
            .build()
            .unwrap();
    window.set_max_fps(FPS);

    // --- Initialize Monitor and Signals ---
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

    // --- Data buffers for plotting ---
    let mut x_signal_data: VecDeque<f64> = VecDeque::from(vec![0.0; N_DATA_POINTS]);
    let mut y_signal_data: VecDeque<f64> = VecDeque::from(vec![0.0; N_DATA_POINTS]);
    let mut robustness_plot_data: VecDeque<f64> = VecDeque::from(vec![0.0; N_DATA_POINTS]);

    while let Some(_) = draw_piston_window(&mut window, |b| {
        // --- Simulation Step ---
        let step_x = signal_generator_x.next_step();
        let step_y = signal_generator_y.next_step();

        let _ = monitor.update(&step_x);
        let result_y = monitor.update(&step_y);

        // Update data buffers
        x_signal_data.pop_front();
        x_signal_data.push_back(step_x.value);

        y_signal_data.pop_front();
        y_signal_data.push_back(step_y.value);

        robustness_plot_data.pop_front();
        let previous_value = *robustness_plot_data.back().unwrap_or(&0.0);
        let robustness_value = result_y
            .last()
            .and_then(|step| step.value)
            .unwrap_or(previous_value);
        robustness_plot_data.push_back(robustness_value);

        // --- Drawing Logic ---
        let root = b.into_drawing_area();
        root.fill(&WHITE)?;

        let (top, bottom) = root.split_vertically(384);

        // --- Top Chart: Signals ---
        let (sig_min, sig_max) =
            get_data_range(&x_signal_data.iter().chain(&y_signal_data).cloned().collect());
        let mut chart_sig = ChartBuilder::on(&top)
            .caption("Input Signals", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0u32..N_DATA_POINTS as u32, sig_min..sig_max)?;

        chart_sig
            .configure_mesh()
            .x_desc("Time (steps)")
            .y_desc("Signal Value")
            .draw()?;

        chart_sig.draw_series(LineSeries::new(
            (0..).zip(x_signal_data.iter()).map(|(a, b)| (a, *b)),
            &RED,
        ))?;
        chart_sig.draw_series(LineSeries::new(
            (0..).zip(y_signal_data.iter()).map(|(a, b)| (a, *b)),
            &BLUE,
        ))?;

        // --- Bottom Chart: Robustness ---
        let (rob_min, rob_max) = get_data_range(&robustness_plot_data);
        let mut chart_rob = ChartBuilder::on(&bottom)
            .caption("Robustness", ("sans-serif", 30).into_font())
            .margin(10)
            .x_label_area_size(30)
            .y_label_area_size(40)
            .build_cartesian_2d(0u32..N_DATA_POINTS as u32, rob_min.min(-1.0)..rob_max.max(1.0))?;

        chart_rob
            .configure_mesh()
            .x_desc("Time (steps)")
            .y_desc("Robustness")
            .draw()?;

        chart_rob.draw_series(LineSeries::new(
            (0..).zip(robustness_plot_data.iter()).map(|(a, b)| (a, *b)),
            &GREEN.mix(0.8),
        ))?;

        // Draw the y=0 line
        chart_rob.draw_series(LineSeries::new(
            vec![(0, 0.0), (N_DATA_POINTS as u32, 0.0)],
            BLACK.stroke_width(2),
        ))?;

        root.present()?;
        Ok(())
    }) {}

    Ok(())
}