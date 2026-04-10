# MPS Documentation Index

This folder contains detailed engineering documentation for MPS.

## Core Documents

- [Architecture](ARCHITECTURE.md)
  Runtime structure, scheduler and dispatcher flow, SIMD dispatch strategy, and platform worker behavior.
- [Performance and Benchmarking](PERFORMANCE_BENCHMARKS.md)
  Standard benchmark methodology, microbench output interpretation, and reproducibility guidance.
- [Platform Tuning](PLATFORM_TUNING.md)
  macOS and Linux build/runtime tuning, environment variables, and troubleshooting patterns.
- [Release Checklist](RELEASE_CHECKLIST.md)
  Release readiness gates and suggested go/no-go criteria for v0.6 and v1.0 milestones.

## Suggested Reading Order

1. Architecture
2. Performance and Benchmarking
3. Platform Tuning
4. Release Checklist

## Fast Start

- Run tests: `cargo test --all-targets`
- Run microbench: `cargo run --release --example mps_microbench`
- Force scalar SIMD (diagnostics): `MPS_SIMD_FORCE_SCALAR=1 cargo run --release --example mps_microbench`
