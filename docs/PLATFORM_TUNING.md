# Platform Tuning Guide

## Scope

This guide covers practical tuning knobs for macOS Apple Silicon and Linux x86_64 systems running MPS.

## Build-Time Tuning

### Apple Silicon (macOS)

Recommended baseline:

```bash
RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1" cargo build --release
```

Notes:

- `target-cpu=native` maps to the current host CPU (for example `apple-m4`).
- NEON is already baseline on AArch64; no extra NEON flag is needed.

### Linux x86_64 (Ryzen)

Recommended baseline for Zen 4:

```bash
RUSTFLAGS="-C target-cpu=znver4 -C lto=fat -C codegen-units=1" cargo build --release
```

Optional if building on the target machine directly:

```bash
RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1" cargo build --release
```

## Runtime Environment Variables

### `MPS_SIMD_FORCE_SCALAR`

- Location: `simd` runtime detection path.
- Behavior: forces scalar backend by bypassing runtime SIMD selection.
- Use case: debugging, deterministic comparison against vector paths.

Example:

```bash
MPS_SIMD_FORCE_SCALAR=1 cargo run --release --example mps_microbench
```

### `TILELINE_MPS_PRIVILEGED_SCHED`

- Location: Linux worker bootstrap uplift detection.
- Behavior:
  - truthy values (`1`, `true`, `yes`, `on`) force privileged uplift path.
  - falsy values (`0`, `false`, `no`, `off`) force unprivileged path.
- Use case: controlled validation of scheduler uplift behavior.

Example:

```bash
TILELINE_MPS_PRIVILEGED_SCHED=0 cargo run --release --example mps_microbench
```

### Worker idle jitter controls

The scheduler worker loop supports optional idle-path overrides:

- `MPS_IDLE_PRE_PARK_SPINS`
  Number of spin-loop checks before yielding when a worker becomes idle.
- `MPS_IDLE_PRE_PARK_YIELDS`
  Number of cooperative yields before entering park timeout.
- `MPS_IDLE_PARK_MIN_US`
  Minimum park timeout in microseconds for adaptive backoff.
- `MPS_IDLE_PARK_MAX_US`
  Maximum park timeout in microseconds before indefinite park.

Example:

```bash
MPS_IDLE_PRE_PARK_SPINS=128 \
MPS_IDLE_PRE_PARK_YIELDS=32 \
MPS_IDLE_PARK_MIN_US=80 \
MPS_IDLE_PARK_MAX_US=2000 \
cargo run --release --example mps_microbench
```

## Linux Operational Recommendations

- Prefer `performance` governor when collecting benchmark baselines.
- Avoid oversubscription from unrelated workloads during tests.
- Validate whether logical-thread worker count is beneficial for your workload profile.

## macOS Operational Recommendations

- Keep thermal and power conditions stable when comparing runs.
- Use multi-run medians from `mps_microbench` summary instead of single-run peaks.
- Treat `parallel(raw)` and `gflops` together when assessing regressions.

## Quick Diagnostic Commands

Check SIMD backend selected by MPS at startup:

```bash
cargo run --release --example mps_microbench | rg "^\[mps\]"
```

Scalar vs vector A/B run:

```bash
cargo run --release --example mps_microbench
MPS_SIMD_FORCE_SCALAR=1 cargo run --release --example mps_microbench
```
