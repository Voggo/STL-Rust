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
