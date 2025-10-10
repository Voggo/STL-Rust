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
    fn pop_front(&mut self) -> Option<Step<Self::Value>> {
        self.steps.pop_front()
    }

    fn add_step(&mut self, step: Step<T>) {
        self.add_step(step)
    }
    fn prune(&mut self, max_age: Duration) {
        while let Some(front_step) = self.steps.front() {
            if front_step.timestamp <= max_age {
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
