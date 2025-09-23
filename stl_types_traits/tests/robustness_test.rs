#[cfg(test)]
mod tests {
    use stl_types_traits::{
        signal::SignalBuffer,
        stl::operators::{STLFormula, TimeInterval},
    };
    use std::time::Duration;

    /// Creates a standard signal for testing.
    /// The signal trace is:
    /// t=0, value=2.0
    /// t=1, value=4.0
    /// t=2, value=6.0
    /// t=3, value=4.0
    /// t=4, value=2.0
    fn get_test_signal() -> SignalBuffer<f64> {
        let mut signal = SignalBuffer::new();
        signal.add_step(2.0, Duration::from_secs(0));
        signal.add_step(4.0, Duration::from_secs(1));
        signal.add_step(6.0, Duration::from_secs(2));
        signal.add_step(4.0, Duration::from_secs(3));
        signal.add_step(2.0, Duration::from_secs(4));
        signal
    }

    #[test]
    fn test_atomic_propositions() {
        let signal = get_test_signal(); // Evaluation is at t=0, value=2.0

        let formula_true = STLFormula::True;
        assert_eq!(formula_true.robustness_naive(&signal), f64::INFINITY);

        let formula_false = STLFormula::False;
        assert_eq!(formula_false.robustness_naive(&signal), f64::NEG_INFINITY);

        // value > 1.0  =>  2.0 - 1.0 = 1.0
        let formula_gt = STLFormula::GreaterThan(1.0);
        assert_eq!(formula_gt.robustness_naive(&signal), 1.0);

        // value < 3.0  =>  3.0 - 2.0 = 1.0
        let formula_lt = STLFormula::LessThan(3.0);
        assert_eq!(formula_lt.robustness_naive(&signal), 1.0);

        // value > 3.0  =>  2.0 - 3.0 = -1.0
        let formula_gt_fail = STLFormula::GreaterThan(3.0);
        assert_eq!(formula_gt_fail.robustness_naive(&signal), -1.0);
    }

    #[test]
    fn test_boolean_operators() {
        let signal = get_test_signal(); // Evaluation is at t=0, value=2.0

        let phi = Box::new(STLFormula::GreaterThan(1.0)); // robustness = 1.0
        let psi = Box::new(STLFormula::LessThan(1.0)); // robustness = -1.0

        // Not(phi) => -1.0
        let formula_not = STLFormula::Not(phi.clone());
        assert_eq!(formula_not.robustness_naive(&signal), -1.0);

        // And(phi, psi) => min(1.0, -1.0) = -1.0
        let formula_and = STLFormula::And(phi.clone(), psi.clone());
        assert_eq!(formula_and.robustness_naive(&signal), -1.0);

        // Or(phi, psi) => max(1.0, -1.0) = 1.0
        let formula_or = STLFormula::Or(phi.clone(), psi.clone());
        assert_eq!(formula_or.robustness_naive(&signal), 1.0);

        // Implies(phi, psi) => max(-rob(phi), rob(psi)) => max(-1.0, -1.0) = -1.0
        let formula_implies = STLFormula::Implies(phi, psi);
        assert_eq!(formula_implies.robustness_naive(&signal), -1.0);
    }

    #[test]
    fn test_eventually_operator() {
        let signal = get_test_signal();
        let interval = TimeInterval {
            start: Duration::from_secs(1),
            end: Duration::from_secs(3),
        };
        // Eventually_[1,3] (value > 5.0)
        // We check t=1,2,3. Values are 4,6,4.
        // Robustness values for (value > 5.0) are: -1, 1, -1
        // Max is 1.0
        let phi = Box::new(STLFormula::GreaterThan(5.0));
        let formula = STLFormula::Eventually(interval, phi);
        assert_eq!(formula.robustness_naive(&signal), 1.0);
    }

    #[test]
    fn test_always_operator() {
        let signal = get_test_signal();
        let interval = TimeInterval {
            start: Duration::from_secs(1),
            end: Duration::from_secs(3),
        };
        // Always_[1,3] (value > 3.0)
        // We check t=1,2,3. Values are 4,6,4.
        // Robustness values for (value > 3.0) are: 1, 3, 1
        // Min is 1.0
        let phi = Box::new(STLFormula::GreaterThan(3.0));
        let formula = STLFormula::Always(interval, phi);
        assert_eq!(formula.robustness_naive(&signal), 1.0);
    }

    #[test]
    fn test_until_operator() {
        let signal = get_test_signal();
        let interval = TimeInterval {
            start: Duration::from_secs(0),
            end: Duration::from_secs(2),
        };

        // (value < 3.0) Until_[0,2] (value > 5.0)
        let phi = Box::new(STLFormula::LessThan(3.0));
        let psi = Box::new(STLFormula::GreaterThan(5.0));
        let formula = STLFormula::Until(interval, phi, psi);

        // Calculation: max_{t' in [0,2]} ( min(rob(psi, t'), min_{t'' in [0,t']} rob(phi, t'')) )
        // t'=0: rob(psi,0)=-3. rob(phi,0)=1. min(-3,1)=-3.
        // t'=1: rob(psi,1)=-1. min(rob(phi,0),rob(phi,1))=min(1,-1)=-1. min(-1,-1)=-1.
        // t'=2: rob(psi,2)=1. min(rob(phi,0),rob(phi,1),rob(phi,2))=min(1,-1,-3)=-3. min(1,-3)=-3.
        // max(-3, -1, -3) = -1.0
        assert_eq!(formula.robustness_naive(&signal), -1.0);
    }

    // Might construct an error when signal has no points in the given interval
    // or do the thing with intervals (rosi)
    #[test]
    fn test_temporal_operator_with_no_points_in_interval() {
        let signal = get_test_signal();
        let interval = TimeInterval {
            start: Duration::from_secs(10),
            end: Duration::from_secs(20),
        };

        let phi = Box::new(STLFormula::GreaterThan(0.0));

        // Eventually should return -inf if no points are found
        let eventually = STLFormula::Eventually(interval, phi.clone());
        assert_eq!(eventually.robustness_naive(&signal), f64::NEG_INFINITY);

        // Always should return +inf if no points are found
        let always = STLFormula::Always(interval, phi);
        assert_eq!(always.robustness_naive(&signal), f64::INFINITY);
    }
}