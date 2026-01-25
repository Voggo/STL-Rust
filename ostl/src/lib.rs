// Enable use of ::ostl:: paths within this crate for the proc-macro
extern crate self as ostl;

pub mod ring_buffer;
pub mod stl;
pub mod synchronizer;

// Re-export the stl macro at crate root for convenience
pub use ostl_macros::stl;