use crate::ring_buffer::{RingBufferTrait, Step};
use crate::stl::operators_naive::STLFormula;
use std::ops::Index;

/// Computes the robustness of the STL formula with respect to the given signal
/// Returns f64::INFINITY if no relevant data is found (for max operations).
/// Returns f64::NEG_INFINITY if no relevant data is found (for min operations).
impl STLFormula {
    pub fn robustness_opt<S>(&self, signal: &S) -> f64
    where
        S: RingBufferTrait<Value = f64> + Index<usize, Output = Step<f64>>,
    {
        let current_value = signal[0];
        // Find the signal value at time 't'
        let current_value_at_t = current_value.value as f64;
        let t = current_value.timestamp;

        match self {
            STLFormula::True => f64::INFINITY,      // Always satisfied
            STLFormula::False => f64::NEG_INFINITY, // Never satisfied
            STLFormula::GreaterThan(c) => current_value_at_t - c, // ρ(s_t, x > c) = s(t) - c
            STLFormula::LessThan(c) => c - current_value_at_t, // ρ(s_t, x < c) = c - s(t)

            // Not: ρ(s_t, ¬φ) = -ρ(s_t, φ)
            STLFormula::Not(phi) => -phi.robustness_naive(signal),

            // And: ρ(s_t, φ ∧ ψ) = min(ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::And(phi, psi) => phi
                .robustness_naive(signal)
                .min(psi.robustness_naive(signal)),

            // Or: ρ(s_t, φ ∨ ψ) = max(ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::Or(phi, psi) => phi
                .robustness_naive(signal)
                .max(psi.robustness_naive(signal)),

            // Implies: ρ(s_t, φ ⇒ ψ) = max(-ρ(s_t, φ), ρ(s_t, ψ))
            STLFormula::Implies(phi, psi) => -phi
                .robustness_naive(signal)
                .max(psi.robustness_naive(signal)),

            // Eventually: ρ(s_t, ◇_[a,b]φ) = max_{t' ∈ [t+a,t+b]} ρ(s_t', φ)
            STLFormula::Eventually(interval, phi) => {
                let start_time = t + interval.start;
                let end_time = t + interval.end;

                signal
                    .iter()
                    .filter(|step| step.timestamp >= start_time && step.timestamp <= end_time)
                    .map(|step| {
                        let mut temp_signal = S::new();
                        temp_signal.add_step(step.value, step.timestamp);
                        phi.robustness_naive(&temp_signal)
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            }

            // Always: ρ(s_t, □_[a,b]φ) = min_{t' ∈ [t+a,t+b]} ρ(s_t', φ)
            STLFormula::Globally(interval, phi) => {
                let start_time = t + interval.start;
                let end_time = t + interval.end;

                signal
                    .iter()
                    .filter(|step| step.timestamp >= start_time && step.timestamp <= end_time)
                    .map(|step| {
                        let mut temp_signal = S::new();
                        temp_signal.add_step(step.value, step.timestamp);
                        phi.robustness_naive(&temp_signal)
                    })
                    .fold(f64::INFINITY, f64::min)
            }

            // Until: ρ(s_t, φ U_[a,b] ψ) = max_{t' ∈ [t+a,t+b]} ( min(ρ(s_t', ψ), min_{t'' ∈ [t,t']} ρ(s_t'', φ)) )
            STLFormula::Until(interval, phi, psi) => {
                let lower_bound_t_prime = t + interval.start;
                let upper_bound_t_prime = t + interval.end;

                signal
                    .iter()
                    .filter(|step| {
                        step.timestamp >= lower_bound_t_prime
                            && step.timestamp <= upper_bound_t_prime
                    })
                    .map(|step| {
                        let t_prime = step.timestamp;
                        let mut temp_signal_psi = S::new();
                        temp_signal_psi.add_step(step.value, step.timestamp);
                        let robustness_psi = psi.robustness_naive(&temp_signal_psi);

                        let robustness_phi = signal
                            .iter()
                            .filter(|s| s.timestamp >= t && s.timestamp <= t_prime)
                            .map(|s| {
                                let mut temp_signal_phi = S::new();
                                temp_signal_phi.add_step(s.value, s.timestamp);
                                phi.robustness_naive(&temp_signal_phi)
                            })
                            .fold(f64::INFINITY, f64::min);

                        robustness_psi.min(robustness_phi)
                    })
                    .fold(f64::NEG_INFINITY, f64::max)
            }
        }
    }
}