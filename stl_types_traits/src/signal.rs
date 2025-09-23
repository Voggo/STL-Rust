use std::{collections::VecDeque, time::Duration};

#[derive(Clone, Copy, Debug)]
pub struct Step<T> {
    pub value: T,
    pub timestamp: Duration,
}

#[derive(Clone)]
pub struct SignalBuffer<T> {
    pub steps: VecDeque<Step<T>>,
}

pub trait SignalTrait {
    // The type of the value stored in the signal
    type Value;
    
    // The container that holds the steps
    type Container: IntoIterator;

    // A Generic Associated Type (GAT) for the iterator.
    // The <'a> here links the iterator's lifetime to the lifetime of `&'a self`.
    type Iter<'a>: Iterator<Item = &'a Step<Self::Value>> where Self: 'a;

    fn new() -> Self;
    fn add_step(&mut self, value: Self::Value, timestamp: Duration);
    fn prune(&mut self, current_time: Duration, max_age: Duration);
    
    // The iter method now returns the generic iterator type.
    fn iter<'a>(&'a self) -> Self::Iter<'a>;
}

impl<T> SignalBuffer<T> {
    pub fn new() -> Self {
        Self {
            steps: VecDeque::new(),
        }
    }

    pub fn add_step(&mut self, value: T, timestamp: Duration) {
        self.steps.push_back(Step { value, timestamp });
    }

    fn prune(&mut self, current_time: Duration, max_age: Duration) {
        todo!("Implementation for pruning old steps")
    }

    pub fn iter(&self) -> std::collections::vec_deque::Iter<Step<T>> {
        self.steps.iter()
    }
}
