//! # OSTL - Online Signal Temporal Logic
//!
//! `ostl` provides runtime and macro-based tooling to parse, build, and execute
//! Signal Temporal Logic (STL) monitors over streaming data.
//!
//! It includes:
//! - a high-level monitor builder API,
//! - incremental evaluation backend,
//! - multiple semantics (qualitative, quantitative, RoSI), and
//! - optional multi-signal synchronization/interpolation.
//!
//! ## Simple usage
//!
//! ```no_run
//! use ostl::ring_buffer::Step;
//! use ostl::stl::monitor::{Algorithm, DelayedQuantitative, StlMonitor};
//! use std::time::Duration;
//!
//! // Build a monitor from the macro DSL.
//! let formula = ostl::stl!(G[0, 2](x > 5.0));
//! let mut monitor = StlMonitor::builder()
//!     .formula(formula)
//!     .algorithm(Algorithm::Incremental)
//!     .semantics(DelayedQuantitative)
//!     .build()
//!     .unwrap();
//!
//! // Stream updates
//! let out1 = monitor.update(&Step::new("x", 7.0, Duration::from_secs(0))).finalized();
//! let out2 = monitor.update(&Step::new("x", 6.0, Duration::from_secs(1))).finalized();
//! let out3 = monitor.update(&Step::new("x", 4.0, Duration::from_secs(2))).finalized();
//!
//!
//!
//! ```
//!

// Enable use of ::ostl:: paths within this crate for the proc-macro
extern crate self as ostl;

pub mod ring_buffer;
pub mod stl;
pub mod synchronizer;

// Re-export the stl macro at crate root for convenience
pub use ostl_macros::stl;
