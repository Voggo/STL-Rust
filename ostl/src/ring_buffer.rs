use std::{collections::VecDeque, ops::Index, time::Duration};

#[derive(Clone, Copy, Debug)]
pub struct Step<T> {
    pub value: T,
    pub timestamp: Duration,
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

    fn add_step(&mut self, value: Self::Value, timestamp: Duration);
    fn prune(&mut self, current_time: Duration, max_age: Duration);

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

    pub fn add_step(&mut self, value: T, timestamp: Duration) {
        self.steps.push_back(Step { value, timestamp });
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<Step<T>> {
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

    fn add_step(&mut self, value: T, timestamp: Duration) {
        self.add_step(value, timestamp)
    }

    fn prune(&mut self, current_time: Duration, max_age: Duration) {
        let cutoff_time = current_time.saturating_sub(max_age);

        // remove_while is unstable, so we use a while loop
        while let Some(front_step) = self.steps.front() {
            if front_step.timestamp < cutoff_time {
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
