//! Lock-free scheduler queues.
//!
//! We use separate lanes for performance and efficient workers,
//! then allow controlled stealing for spillover.

use super::TaskEnvelope;
use crate::balancer::TaskPriority;
use crate::topology::CpuClass;
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Queue depth snapshot for diagnostics and runtime tuning.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueueDepth {
    pub performance: usize,
    pub efficient: usize,
    pub shared: usize,
    pub total: usize,
}

/// Lock-free task queue with topology-aware lanes.
#[derive(Clone)]
pub struct PriorityTaskQueue {
    performance: Arc<LaneQueue>,
    efficient: Arc<LaneQueue>,
    shared: Arc<LaneQueue>,
}

impl PriorityTaskQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self {
            performance: Arc::new(LaneQueue::new()),
            efficient: Arc::new(LaneQueue::new()),
            shared: Arc::new(LaneQueue::new()),
        }
    }

    /// Push a task into the lane selected by the load balancer.
    pub fn push(&self, task: TaskEnvelope) {
        match task.preferred_class {
            CpuClass::Performance => self.performance.push(task),
            CpuClass::Efficient => self.efficient.push(task),
            CpuClass::Unknown => self.shared.push(task),
        }
    }

    /// Pop a task for a worker class, with controlled cross-lane stealing.
    pub fn pop_for_worker(&self, worker_class: CpuClass) -> Option<TaskEnvelope> {
        match worker_class {
            CpuClass::Performance => self
                .performance
                .pop_high_to_low()
                .or_else(|| self.shared.pop_high_to_low())
                .or_else(|| {
                    self.steal_with_spill_check(
                        &self.efficient,
                        worker_class,
                        [
                            TaskPriority::Critical,
                            TaskPriority::High,
                            TaskPriority::Normal,
                            TaskPriority::Background,
                        ],
                    )
                }),
            CpuClass::Efficient => self
                .shared
                .pop_high_to_low()
                .or_else(|| {
                    self.steal_with_spill_check(
                        &self.performance,
                        worker_class,
                        [
                            // Prefer spillable work first so non-spill critical
                            // tasks do not block stealing on hybrid systems.
                            TaskPriority::High,
                            TaskPriority::Normal,
                            TaskPriority::Background,
                            TaskPriority::Critical,
                        ],
                    )
                })
                .or_else(|| self.efficient.pop_background_to_critical()),
            CpuClass::Unknown => self
                .shared
                .pop_high_to_low()
                .or_else(|| self.performance.pop_high_to_low())
                .or_else(|| self.efficient.pop_background_to_critical()),
        }
    }

    /// Return true when all lanes are empty.
    pub fn is_empty(&self) -> bool {
        self.total_len() == 0
    }

    /// Approximate total queue length.
    pub fn total_len(&self) -> usize {
        self.performance.depth() + self.efficient.depth() + self.shared.depth()
    }

    /// Snapshot lane depths.
    pub fn depth_snapshot(&self) -> QueueDepth {
        let performance = self.performance.depth();
        let efficient = self.efficient.depth();
        let shared = self.shared.depth();
        QueueDepth {
            performance,
            efficient,
            shared,
            total: performance + efficient + shared,
        }
    }

    fn steal_with_spill_check(
        &self,
        source: &LaneQueue,
        worker_class: CpuClass,
        steal_order: [TaskPriority; 4],
    ) -> Option<TaskEnvelope> {
        // Bound retry count to avoid endless churn when only non-spill tasks exist.
        for _ in 0..4 {
            let task = source.pop_in_order(steal_order)?;
            if task.spill_to_any
                || task.preferred_class == CpuClass::Unknown
                || task.preferred_class == worker_class
            {
                return Some(task);
            }

            // Return non-spill tasks to their original lane.
            self.push(task);
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::TaskPayload;

    fn noop_payload() -> TaskPayload {
        TaskPayload::Native(Box::new(|| {}))
    }

    #[test]
    fn efficient_steal_skips_non_spill_critical_barrier() {
        let queue = PriorityTaskQueue::new();

        queue.push(TaskEnvelope::new(
            1,
            TaskPriority::Critical,
            CpuClass::Performance,
            false,
            noop_payload(),
        ));
        queue.push(TaskEnvelope::new(
            2,
            TaskPriority::Critical,
            CpuClass::Performance,
            false,
            noop_payload(),
        ));
        queue.push(TaskEnvelope::new(
            3,
            TaskPriority::High,
            CpuClass::Performance,
            true,
            noop_payload(),
        ));

        let stolen = queue
            .pop_for_worker(CpuClass::Efficient)
            .expect("efficient worker should steal spillable high task");

        assert_eq!(stolen.priority, TaskPriority::High);
        assert!(stolen.spill_to_any);
        assert_eq!(queue.total_len(), 2);
    }
}

struct LaneQueue {
    critical: SegQueue<TaskEnvelope>,
    high: SegQueue<TaskEnvelope>,
    normal: SegQueue<TaskEnvelope>,
    background: SegQueue<TaskEnvelope>,
    depth: AtomicUsize,
}

impl LaneQueue {
    fn new() -> Self {
        Self {
            critical: SegQueue::new(),
            high: SegQueue::new(),
            normal: SegQueue::new(),
            background: SegQueue::new(),
            depth: AtomicUsize::new(0),
        }
    }

    fn push(&self, task: TaskEnvelope) {
        match task.priority {
            TaskPriority::Critical => self.critical.push(task),
            TaskPriority::High => self.high.push(task),
            TaskPriority::Normal => self.normal.push(task),
            TaskPriority::Background => self.background.push(task),
        }
        self.depth.fetch_add(1, Ordering::Relaxed);
    }

    fn pop_high_to_low(&self) -> Option<TaskEnvelope> {
        self.pop_in_order([
            TaskPriority::Critical,
            TaskPriority::High,
            TaskPriority::Normal,
            TaskPriority::Background,
        ])
    }

    fn pop_background_to_critical(&self) -> Option<TaskEnvelope> {
        self.pop_in_order([
            TaskPriority::Background,
            TaskPriority::Normal,
            TaskPriority::High,
            TaskPriority::Critical,
        ])
    }

    fn pop_in_order(&self, order: [TaskPriority; 4]) -> Option<TaskEnvelope> {
        for priority in order {
            if let Some(task) = self.pop_priority(priority) {
                return Some(task);
            }
        }
        None
    }

    fn pop_priority(&self, priority: TaskPriority) -> Option<TaskEnvelope> {
        let popped = match priority {
            TaskPriority::Critical => self.critical.pop(),
            TaskPriority::High => self.high.pop(),
            TaskPriority::Normal => self.normal.pop(),
            TaskPriority::Background => self.background.pop(),
        };

        if popped.is_some() {
            self.depth.fetch_sub(1, Ordering::Relaxed);
        }

        popped
    }

    fn depth(&self) -> usize {
        self.depth.load(Ordering::Relaxed)
    }
}
