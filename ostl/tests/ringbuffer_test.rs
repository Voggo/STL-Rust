#[cfg(test)]
mod tests {
    use ostl::ring_buffer::RingBuffer;

    #[test]
    fn ring_buffer_test() {
        let mut signal = RingBuffer::new();
        signal.add_step(1, std::time::Duration::new(0, 0));
        signal.add_step(2, std::time::Duration::new(1, 0));
        signal.add_step(3, std::time::Duration::new(2, 0));

        for i in 0..3 {
            signal.steps.get(i).map(|step| {
                println!("Step {}: value = {}, timestamp = {:?}", i, step.value, step.timestamp);
            });
        }
    }
}
