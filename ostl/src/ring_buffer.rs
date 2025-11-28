use std::{collections::VecDeque, time::Duration};

#[derive(Clone, Debug, PartialEq)]
pub struct Step<T> {
    pub signal: &'static str,
    pub value: T,
    pub timestamp: Duration,
}

impl<T> Step<T> {
    pub fn new(signal: &'static str, value: T, timestamp: Duration) -> Self {
        Step {
            signal,
            value,
            timestamp,
        }
    }
}

pub trait RingBufferTrait {
    // The type of the value stored in the signal
    type Value;

    // The container that holds the steps
    type Container: IntoIterator;

    // A Generic Associated Type for the iterator.
    // The <'a> here links the iterator's lifetime to the lifetime of `&'a self`.
    type Iter<'a>: Iterator<Item = &'a Step<Self::Value>>
    where
        Self: 'a;

    type IterMut<'a>: Iterator<Item = &'a mut Step<Self::Value>>
    where
        Self: 'a;

    fn new() -> Self;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn get_back(&self) -> Option<&Step<Self::Value>>;
    fn get_front(&self) -> Option<&Step<Self::Value>>;
    fn pop_front(&mut self) -> Option<Step<Self::Value>>;

    fn add_step(&mut self, step: Step<Self::Value>);
    fn update_step(&mut self, step: Step<Self::Value>) -> bool;

    /// Prune steps older than `max_age` from the buffer.
    fn prune(&mut self, max_age: Duration);

    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}

#[derive(Clone, Debug)]
pub struct RingBuffer<T> {
    steps: VecDeque<Step<T>>,
}

impl<T> Default for RingBuffer<T>
where
    T: Copy,
 {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> RingBuffer<T>
where
    T: Copy,
{
    pub fn new() -> Self {
        RingBuffer {
            steps: VecDeque::new(),
        }
    }

    pub fn add_step(&mut self, step: Step<T>) {
        self.steps.push_back(step);
    }

    pub fn update_step(&mut self, step: Step<T>) -> bool {
        self.steps.binary_search_by(|s| s.timestamp.cmp(&step.timestamp))
            .map(|index| {
                self.steps[index] = step;
            })
            .is_ok()
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_, Step<T>> {
        self.steps.iter()
    }

}

impl<T> RingBufferTrait for RingBuffer<T>
where
    T: Copy,
{
    type Value = T;
    type Container = VecDeque<Step<T>>;
    type Iter<'a>
        = std::collections::vec_deque::Iter<'a, Step<T>>
    where
        Self: 'a;

    type IterMut<'a>
        = std::collections::vec_deque::IterMut<'a, Step<T>>
    where
        Self: 'a;

    fn new() -> Self {
        Self::new()
    }
    fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
    fn len(&self) -> usize {
        self.steps.len()
    }
    fn get_back(&self) -> Option<&Step<T>> {
        self.steps.back()
    }
    fn get_front(&self) -> Option<&Step<T>> {
        self.steps.front()
    }
    fn pop_front(&mut self) -> Option<Step<Self::Value>> {
        self.steps.pop_front()
    }

    fn add_step(&mut self, step: Step<T>) {
        self.add_step(step)
    }
    fn update_step(&mut self, step: Step<Self::Value>) -> bool {
        self.update_step(step)
    }
    fn prune(&mut self, max_age: Duration) {
        let current_time = match self.get_back() {
            Some(step) => step.timestamp,
            None => return, // Buffer is empty, nothing to prune
        };
        let max_age = current_time.saturating_sub(max_age);
        while let Some(front_step) = self.steps.front() {
            if front_step.timestamp < max_age {
                self.steps.pop_front();
            } else {
                break;
            }
        }
    }

    fn iter<'a>(&'a self) -> Self::Iter<'a> {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_creation() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::new(0, 0),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::new(0, 0),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::new(0, 0),
        });

        for i in 0..3 {
            if let Some(step) = signal.steps.get(i) { assert_eq!(step.value, i + 1) }
        }
    }

    #[test]
    fn ring_prune() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::from_secs(3),
        });

        // Prune steps older than 1 second from the latest timestamp (which is 3 seconds)
        // This should remove the step with timestamp 1 second
        signal.prune(Duration::from_secs(1));

        assert_eq!(signal.len(), 2);
        assert_eq!(signal.get_front().unwrap().value, 2);
        assert_eq!(signal.get_back().unwrap().value, 3);
    }
    #[test]
    fn ring_get_back() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::from_secs(3),
        });
        let back_step = signal.get_back().unwrap();
        assert_eq!(back_step.value, 3);
    }
    #[test]
    fn ring_get_front() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::from_secs(3),
        });
        let front_step = signal.get_front().unwrap();
        assert_eq!(front_step.value, 1);
    }
    #[test]
    fn ring_pop_front() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::from_secs(3),
        });
        let popped_step = signal.pop_front().unwrap();
        assert_eq!(popped_step.value, 1);
        assert_eq!(signal.len(), 2);
        assert_eq!(signal.get_front().unwrap().value, 2);
        assert_eq!(signal.get_back().unwrap().value, 3);
    }
    #[test]
    fn ring_iter() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            signal: "x",
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            signal: "x",
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
            signal: "x",
            value: 3,
            timestamp: Duration::from_secs(3),
        });
        let mut iter = signal.iter();
        assert_eq!(iter.next().unwrap().value, 1);
        assert_eq!(iter.next().unwrap().value, 2);
        assert_eq!(iter.next().unwrap().value, 3);
        assert!(iter.next().is_none());
    }
    #[test]
    fn ring_is_empty() {
        let signal: RingBuffer<bool> = RingBuffer::new();
        assert!(signal.is_empty());
    }

    #[test]
    fn default_ring_buffer() {
        let signal: RingBuffer<i32> = RingBuffer::default();
        assert!(signal.is_empty());
    }
}
