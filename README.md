# MPS

**Multi Processing Scaler** is a standalone bare-metal CPU execution
crate: a topology-aware task scheduler, an overlapped physics dispatcher, and a
runtime-dispatched SIMD kernel surface in one package.

MPS is designed for low-latency frame pipelines where **Render N** and
**Physics N+1** must overlap without forcing a stage-barrier ECS scheduler into
the hot path.

## What MPS Gives You

- **Topology-aware CPU scheduling** with P-core / E-core classification, queue
  routing, and per-class telemetry.
- **Bare-metal overlapped physics dispatch** via `TaskDispatcher`, including
  double-buffered transform snapshots and lock-free chunk queues.
- **Runtime-dispatched SIMD kernels** for `Scalar`, `AVX-512`, `NEON`, and
  `VMX/AltiVec`, selected once during scheduler/dispatcher bootstrap.
- **Linux worker bootstrap controls** with affinity pinning, optional
  `SCHED_FIFO`, nice-value tuning, and a futex-backed wake signal.
- **Native + WASM task execution** through `MpsScheduler` and `WasmTask`.
- **Auxiliary payload/key analysis utilities** (`key_decode`, `key_bruteforce`)
  that use the same scheduler infrastructure.

## Install

Until a crates.io publish happens, consume MPS directly from the standalone
repository tag:

```toml
[dependencies]
mps = { git = "https://github.com/berrycomp/mps", tag = "v0.5.5" }
```

## Quick Start: Priority Scheduler

```rust
use mps::{CorePreference, MpsScheduler, TaskPriority};
use std::time::Duration;

let scheduler = MpsScheduler::new();

let native_id = scheduler.submit_native(
    TaskPriority::High,
    CorePreference::Performance,
    || {
        let mut acc = 0_u64;
        for n in 0..100_000 {
            acc = acc.wrapping_add(n);
        }
        std::hint::black_box(acc);
    },
);

let idle = scheduler.wait_for_idle(Duration::from_secs(5));
let metrics = scheduler.metrics();

println!(
    "native={native_id} idle={idle} completed={} simd={:?}/{} lanes",
    metrics.completed,
    metrics.simd_backend,
    metrics.simd_lanes
);
```

## Quick Start: Overlapped Physics N+1 / SceneBuild N

`TaskDispatcher` is the canonical low-latency path when you want one physics
frame to run on pinned workers while the main thread keeps building the current
render scene.

```rust
use mps::{
    DispatcherPhaseCallbacks, DispatcherTaskContext, DispatcherTransformSample,
    PhysicsDispatchTrigger, TaskDispatcher, TaskDispatcherConfig,
};
use std::sync::Arc;
use std::time::Duration;

let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
    transform_capacity: 65_536,
    default_chunk_size: 256,
    ..TaskDispatcherConfig::default()
})
.expect("dispatcher bootstrap should succeed");

let callbacks = DispatcherPhaseCallbacks::default().with_integration(Arc::new(
    |ctx: &DispatcherTaskContext| {
        for body_index in ctx.work_range.clone() {
            ctx.transforms.write_physics_transform(
                body_index,
                DispatcherTransformSample {
                    position: [body_index as f32, 0.0, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                },
            );
        }
    },
));

let render_ticket = dispatcher.acquire_scene_build(42);
let physics_ticket = dispatcher.trigger_next_physics(
    PhysicsDispatchTrigger::new(43, 0, 0, 30_000, 256),
    callbacks,
)
.expect("frame trigger should succeed");

assert_eq!(render_ticket.render_read_slot, physics_ticket.render_read_slot);
assert!(dispatcher.wait_for_completed_frame(43, Duration::from_millis(250)));
```

## Quick Start: SIMD Kernel Surface

```rust
use mps::{
    AabbBoundsSoaMut, AabbInputSoaRef, PositionSoaMut, SimdKernelSet, VelocitySoaRef,
};

let kernels = SimdKernelSet::detect_runtime();

let mut px = vec![0.0f32; 8];
let mut py = vec![0.0f32; 8];
let mut pz = vec![0.0f32; 8];
let vx = vec![1.0f32; 8];
let vy = vec![2.0f32; 8];
let vz = vec![3.0f32; 8];

kernels.integrate_positions(
    &mut PositionSoaMut {
        position_x: &mut px,
        position_y: &mut py,
        position_z: &mut pz,
    },
    VelocitySoaRef {
        velocity_x: &vx,
        velocity_y: &vy,
        velocity_z: &vz,
    },
    0.5,
);

let mut min_x = vec![0.0f32; 8];
let mut min_y = vec![0.0f32; 8];
let mut min_z = vec![0.0f32; 8];
let mut max_x = vec![0.0f32; 8];
let mut max_y = vec![0.0f32; 8];
let mut max_z = vec![0.0f32; 8];
let half = vec![0.25f32; 8];

kernels.aabb_min_max(
    &mut AabbBoundsSoaMut {
        min_x: &mut min_x,
        min_y: &mut min_y,
        min_z: &mut min_z,
        max_x: &mut max_x,
        max_y: &mut max_y,
        max_z: &mut max_z,
    },
    AabbInputSoaRef {
        center_x: &px,
        center_y: &py,
        center_z: &pz,
        half_x: &half,
        half_y: &half,
        half_z: &half,
    },
);

println!("selected SIMD backend: {:?}", kernels.backend());
```

## Architecture Map

| Module | Role |
| --- | --- |
| `topology` | Detect logical/physical cores, classify P/E/Unknown lanes, snapshot SIMD capability. |
| `balancer` | Convert task priority + core preference into an initial queue lane and spill policy. |
| `scheduler` | Lock-free priority queues, `MpsScheduler`, WASM dispatch, and the modern `TaskDispatcher`. |
| `thread_pool` | Legacy custom physics pool with double-buffered transforms and chunked phase callbacks. |
| `worker` | Linux/macOS worker bootstrap, affinity/priority setup, and futex-backed wake signaling. |
| `simd` | Runtime-selected SoA kernels for transform copy, position integration, and AABB helpers. |
| `key_decode` | Strict `lu:pmta:<payload>` parsing and escaped-base64 normalization utilities. |
| `key_bruteforce` | Parallelized payload scan helpers for the research/tooling side of the crate. |
| `mobile_bridge` | Optional MPS -> MGS workload translation helper used by Tileline mobile paths. |

## Worker and Scheduler Tuning

MPS tries to apply privileged scheduling when the host allows it, then falls
back safely if not.

- `TILELINE_MPS_PRIVILEGED_SCHED=1` forces privileged scheduler uplift checks
  to pass, so workers may request `SCHED_FIFO` and negative nice values.
- `TILELINE_MPS_PRIVILEGED_SCHED=0` forces the unprivileged path.
- `MPS_SIMD_FORCE_SCALAR=1` disables AVX-512 / NEON / AltiVec dispatch and pins
  `SimdKernelSet` to the scalar backend for reproducibility or debugging.

On regular unprivileged Linux sessions, affinity usually still applies, while
real-time policy / negative nice requests are automatically normalized away.

## Run the Examples

```bash
cargo run --example mps_microbench
cargo run --example wasm_task_demo
cargo run --example strict_key_decode -- key.txt
cargo run --example payload_bruteforce -- key.txt candidate_keys.txt
```

## Status and Boundaries

- `TaskDispatcher` is the preferred overlapped frame pipeline for new physics
  integrations.
- `MpsThreadPool` stays available as a simpler compatibility layer while
  callers migrate.
- AVX-512 / NEON / AltiVec dispatch is selected at bootstrap, but non-scalar
  `aabb_overlap_mask` currently falls back to the scalar helper.
- The key/payload utilities are intentionally shipped as auxiliary tooling, not
  as the core runtime API.

## License

MPS is released under **MPL-2.0**. See `LICENSE`.

## Version

Current standalone tag: `v0.5.5`
