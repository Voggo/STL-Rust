#[cfg(test)]
mod tests {
    use ostl::ring_buffer::{RingBuffer, Step};
    use std::time::Duration;

    #[test]
    fn ring_buffer_test() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            value: 1,
            timestamp: Duration::new(0, 0),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::new(0, 0),
        });
        signal.add_step(Step {
            value: 3,
            timestamp: Duration::new(0, 0),
        });

        for i in 0..3 {
            signal.steps.get(i).map(|step| {
                println!(
                    "Step {}: value = {}, timestamp = {:?}",
                    i, step.value, step.timestamp
                );
            });
        }
    }
}
