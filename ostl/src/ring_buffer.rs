use std::{collections::VecDeque, ops::Index, time::Duration};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Step<T> {
    pub value: T,
    pub timestamp: Duration,
}

impl<T> Step<T> {
    pub fn new(value: T, timestamp: Duration) -> Self {
        Step { value, timestamp }
    }
}

pub trait RingBufferTrait {
    // The type of the value stored in the signal
    type Value;

    // The container that holds the steps
    type Container: IntoIterator;

    // A Generic Associated Type (GAT) for the iterator.
    // The <'a> here links the iterator's lifetime to the lifetime of `&'a self`.
    type Iter<'a>: Iterator<Item = &'a Step<Self::Value>>
    where
        Self: 'a;

    fn new() -> Self;
    fn is_empty(&self) -> bool;
    fn len(&self) -> usize;
    fn get_back(&self) -> Option<&Step<Self::Value>>;
    fn get_front(&self) -> Option<&Step<Self::Value>>;
    fn pop_front(&mut self) -> Option<Step<Self::Value>>;

    fn add_step(&mut self, step: Step<Self::Value>);
    /// Prune steps older than `max_age` from the buffer.
    /// This method removes all steps with a timestamp less than `current_time - max_age`.
    fn prune(&mut self, max_age: Duration);

    // The iter method now returns the generic iterator type.
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}

#[derive(Clone)]
pub struct RingBuffer<T> {
    pub steps: VecDeque<Step<T>>,
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

    pub fn pop_front(&mut self) -> Option<Step<T>> {
        self.steps.pop_front()
    }



    pub fn iter(&self) -> std::collections::vec_deque::Iter<'_,Step<T>> {
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

    fn new() -> Self {
        RingBuffer::new()
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

impl<T> Index<usize> for RingBuffer<T>
where
    T: Copy,
{
    type Output = Step<T>;
    fn index(&self, index: usize) -> &Self::Output {
        &self.steps[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_creation() {
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
                assert_eq!(step.value, i + 1);
            });
        }
    }

    #[test]
    fn ring_prune() {
        let mut signal = RingBuffer::new();
        signal.add_step(Step {
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
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
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
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
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
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
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
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
            value: 1,
            timestamp: Duration::from_secs(1),
        });
        signal.add_step(Step {
            value: 2,
            timestamp: Duration::from_secs(2),
        });
        signal.add_step(Step {
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
}
