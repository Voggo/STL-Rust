use crate::signal::SignalBuffer;
use crate::stl::operators::{STLFormula, TimeInterval};
use std::time::Duration;

/// Computes the robustness of the STL formula with respect to the given signal
/// Returns f64::INFINITY if no relevant data is found (for max operations).
/// Returns f64::NEG_INFINITY if no relevant data is found (for min operations).
impl STLFormula {
    pub fn robustness(&self, signal: &SignalBuffer<isize>) -> f64 {
        // Find the signal value at time 't'
        let current_value = signal.steps[0];
        let current_value_at_t = current_value.value as f64;
        let t = current_value.timestamp;

        match self {
            STLFormula::True => f64::INFINITY,      // Always satisfied
            STLFormula::False => f64::NEG_INFINITY, // Never satisfied
            STLFormula::GreaterThan(c) => current_value_at_t - c, // ρ(s_t, x > c) = s(t) - c
            STLFormula::LessThan(c) => c - current_value_at_t, // ρ(s_t, x < c) = c - s(t)

            // Not: ρ(s_t, ¬φ) = -ρ(s_t, φ)
            STLFormula::Not(phi) => -phi.robustness(signal),

            // And: ρ(s_t, φ ∧ ψ) = min(ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::And(phi, psi) => phi.robustness(signal).min(psi.robustness(signal)),

            // Or: ρ(s_t, φ ∨ ψ) = max(ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::Or(phi, psi) => phi.robustness(signal).max(psi.robustness(signal)),

            // Implies: ρ(s_t, φ ⇒ ψ) = max(-ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::Implies(phi, psi) => -phi.robustness(signal).max(psi.robustness(signal)),

            // Eventually: ρ(s_t, ◇_[a,b]φ) = max_{t' ∈ [t+a,t+b]} ρ(s_t', φ)
            STLFormula::Eventually(interval, phi) => {
                let start_time = t + interval.start;
                let end_time = t + interval.end;

                todo!("Iterate over relevant signal points and find the maximum robustness")
            }

            // Always: ρ(s_t, □_[a,b]φ) = min_{t' ∈ [t+a,t+b]} ρ(s_t', φ)
            STLFormula::Always(interval, phi) => {
                let start_time = t + interval.start;
                let end_time = t + interval.end;

                todo!("Iterate over relevant signal points and find the minimum robustness")
            }

            // Until: ρ(s_t, φ U_[a,b] ψ) = max_{t' ∈ [t+a,t+b]} ( min(ρ(s_t', ψ), min_{t'' ∈ [t,t']} ρ(s_t'', φ)) )
            STLFormula::Until(interval, phi, psi) => {
                let lower_bound_t_prime = t + interval.start;
                let upper_bound_t_prime = t + interval.end;

                todo!("Iterate over relevant signal points and compute the until robustness")
            }
        }
    }
}
