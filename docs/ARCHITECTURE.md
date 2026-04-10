# MPS Architecture

## Summary

MPS is a CPU runtime focused on low-latency frame pipelines where render work for frame `N` overlaps physics work for frame `N+1`.

The crate combines:

- A topology-aware task scheduler (`MpsScheduler`)
- A phase-based overlapped dispatcher (`TaskDispatcher`)
- Runtime SIMD kernel selection (`SimdKernelSet`)
- Platform worker bootstrap logic (`worker` module)

## Module Map

- `topology`
  Detects logical/physical cores, vendor, hybrid class map (Performance/Efficient/Unknown), and SIMD capabilities.
- `balancer`
  Converts task priority and core preference into queue routing and spill behavior.
- `scheduler`
  Lock-free queueing, worker orchestration, metrics, and native/WASM task execution.
- `scheduler::dispatcher`
  Physics phase orchestration and render/physics overlap with double-buffered transforms.
- `simd`
  Runtime-dispatched kernels for copy/integrate/AABB operations.
- `worker`
  Thread bootstrap primitives (affinity, nice, optional privileged uplift, wake/wait behavior).
- `thread_pool`
  Legacy pool path maintained for compatibility while `TaskDispatcher` is preferred.

## `MpsScheduler` Data Flow

1. `MpsScheduler::new()` snapshots topology and creates workers.
2. Each task is wrapped into `TaskEnvelope` with priority and preferred class.
3. `PriorityTaskQueue` routes tasks into performance/efficient/shared lanes.
4. Wake routing targets class-specific worker pools when available.
5. Worker threads apply platform QoS/affinity hints and execute native or WASM payloads.
6. Class counters and queue depth snapshots feed scheduler metrics.

Key behavior notes:

- Batch submission has deferred wake support (`submit_batch_native_deferred_wake`) for lower submission jitter.
- `wake_all_workers()` is available for explicit synchronized wake patterns.
- `wait_for_idle()` uses staged spin/yield/sleep backoff to avoid fixed polling penalties.

## `TaskDispatcher` Overlap Flow

`TaskDispatcher` is the recommended path for frame-overlapped physics pipelines.

1. Main thread acquires scene-build ticket for render frame `N`.
2. Main thread triggers physics frame `N+1` with phase callbacks.
3. Dispatcher schedules broadphase/narrowphase/integration jobs on workers.
4. Integration publishes transform write slot as new render-visible slot.
5. Completed frame metadata is queued for the main thread.

This model removes hard ECS stage barriers from the hot path and keeps render and physics ownership separated via double buffering.

## SIMD Strategy

`SimdKernelSet::detect_runtime()` picks one backend at bootstrap:

- `X86Avx512` on x86_64 when AVX-512F is available.
- `Aarch64Neon` on Apple Silicon/AArch64.
- `PowerPcAltivec` on supported PowerPC targets.
- Scalar fallback otherwise.

Current boundary:

- `aabb_overlap_mask` is vectorized on AVX-512 and NEON.
- PowerPC/AltiVec keeps scalar fallback for `aabb_overlap_mask`.

## Platform Worker Behavior

- macOS/iOS
  Uses pthread QoS class hints for class-oriented scheduling behavior.
- Linux
  Applies affinity masks and class-sensitive nice values, with graceful degradation when privileged uplift is unavailable.
- Other targets
  Falls back to neutral worker behavior.

## Non-Goals and Boundaries

- MPS is not a full ECS framework.
- The key/payload analysis utilities are auxiliary tooling and not the primary runtime API surface.
- `thread_pool` is a compatibility path; new integrations should target `TaskDispatcher` first.
