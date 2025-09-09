use std::time::Duration;

pub const THRESHOLD_VALUE: isize = 42;

// struct 'SignalStep' to represent a single step in the signal
pub struct SignalStep {
    value: isize,        // use isize since we may have negative values
    timestamp: Duration, // use built-in duration type
}

// Derive PartialEq to allow comparison in tests, derive Copy to allow copying to make vector quickly.
// Rust requires Copy types to also implement Clone, so we derive both.
#[derive(PartialEq, Clone, Copy)] 
// enum 'Verdict' to represent the result of the validation
pub enum Verdict {
    Ok,
    Violated,
}

// Validate that the signal never exceeds the threshold value for more than a duration threshold
// note: we use a ref to vectorslice because we don't know the size of [SignalStep] at compile time, but references to slices have a known size
pub fn validate_signal(signal: &[SignalStep], duration_threshold: Duration) -> Vec<Verdict> {
    let mut verdict: Vec<Verdict> = Vec::new();
    let mut exceed_start: Option<Duration> = None;

    for s in signal {
        if s.value > THRESHOLD_VALUE {
            if exceed_start.is_none() {
                exceed_start = Some(s.timestamp);
            }
            if let Some(start) = exceed_start {
                if s.timestamp - start > duration_threshold {
                    verdict.push(Verdict::Violated);
                } else {
                    verdict.push(Verdict::Ok);
                }
            }
        } else {
            exceed_start = None;
            verdict.push(Verdict::Ok);
        }
    }
    verdict
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal_1() -> Vec<SignalStep>{
        vec![
            SignalStep {
                value: 10,
                timestamp: Duration::new(0, 0),
            },
            SignalStep {
                value: 20,
                timestamp: Duration::new(1, 0),
            },
            SignalStep {
                value: 30,
                timestamp: Duration::new(2, 0),
            },
            SignalStep {
                value: 50,
                timestamp: Duration::new(3, 0),
            },
            SignalStep {
                value: 60,
                timestamp: Duration::new(4, 0),
            },
            SignalStep {
                value: 40,
                timestamp: Duration::new(5, 0),
            },
            SignalStep {
                value: 30,
                timestamp: Duration::new(6, 0),
            },
            SignalStep {
                value: 20,
                timestamp: Duration::new(7, 0),
            },
        ]
    }
    #[test]
    fn signal_1_full_signal() {
        let signal = signal_1();
        let verdict_vec = vec![Verdict::Ok; 8];
        let res_vec = validate_signal(&signal, Duration::new(5, 0));
        assert!(
            res_vec
                .iter()
                .zip(verdict_vec.iter())
                .all(|verdicts| verdicts.0 == verdicts.1)
        );
    }
    #[test]
    fn signal_1_partial_signal() {
        let signal = signal_1();
        let verdict_vec = vec![Verdict::Ok; 4];
        let res_vec = validate_signal(&signal[0..4], Duration::new(5, 0));
        assert!(
            res_vec
                .iter()
                .zip(verdict_vec.iter())
                .all(|verdicts| verdicts.0 == verdicts.1)
        );
    }

    fn signal_2() -> Vec<SignalStep> {
        vec![
            SignalStep {
                value: 10,
                timestamp: Duration::new(0, 0),
            },
            SignalStep {
                value: 20,
                timestamp: Duration::new(1, 0),
            },
            SignalStep {
                value: 30,
                timestamp: Duration::new(2, 0),
            },
            SignalStep {
                value: 50,
                timestamp: Duration::new(3, 0),
            },
            SignalStep {
                value: 60,
                timestamp: Duration::new(9, 0),
            },
        ]
    }
    #[test]
    fn signal_2_full_signal() {
        let signal = signal_2();
        let mut verdict_vec = vec![Verdict::Ok; 4];
        verdict_vec.push(Verdict::Violated);
        let res_vec = validate_signal(&signal, Duration::new(5, 0));
        assert!(
            res_vec
                .iter()
                .zip(verdict_vec.iter())
                .all(|verdicts| verdicts.0 == verdicts.1)
        );
        
    }
    #[test]
    fn signal_2_partial_signal() {
        let signal = signal_2();
        let verdict_vec = vec![Verdict::Ok; 4];
        let res_vec = validate_signal(&signal[0..4], Duration::new(5, 0));
        assert!(
            res_vec
                .iter()
                .zip(verdict_vec.iter())
                .all(|verdicts| verdicts.0 == verdicts.1)
        );
    }
}   
