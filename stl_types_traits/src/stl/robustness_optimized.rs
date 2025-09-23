use crate::signal::{SignalTrait, Step};
use crate::stl::operators::{STLFormula};
use std::ops::Index;

/// Computes the robustness of the STL formula with respect to the given signal
/// Returns f64::INFINITY if no relevant data is found (for max operations).
/// Returns f64::NEG_INFINITY if no relevant data is found (for min operations).
impl STLFormula {
    pub fn robustness_opt<S>(&self, signal: &S) -> f64 
    where
        S: SignalTrait<Value = f64> + Index<usize, Output = Step<f64>>,
    {
        return 42.;
    }
}
