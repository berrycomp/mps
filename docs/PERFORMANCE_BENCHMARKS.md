# Performance and Benchmarking

## Goals

This document standardizes how to benchmark MPS and interpret results.

Primary goals:

- Compare builds and scheduler changes reproducibly.
- Separate workload behavior from one-off timing noise.
- Track throughput and efficiency using stable summary metrics.

## Microbench Entry Point

Use:

```bash
cargo run --release --example mps_microbench
```

The benchmark now includes:

- Warmup and measured runs
- Per-run output (`elapsed`, `score`, `parallel(raw)`, `parallel(norm)`, `gflops`)
- Summary statistics with coefficient of variation (CV) and `p95`
- Optional machine-readable JSON summary output

## Runtime Configuration

`mps_microbench` supports env-based tuning:

- `MPS_BENCH_WARMUP_RUNS` (default: `2`)
- `MPS_BENCH_MEASURED_RUNS` (default: `7`)
- `MPS_BENCH_TASKS_PER_CORE` (default: `240`)
- `MPS_BENCH_ITER_SCALE` (default: `4`)
- `MPS_BENCH_OUTPUT_JSON` (default: `0`)

Example:

```bash
MPS_BENCH_WARMUP_RUNS=3 \
MPS_BENCH_MEASURED_RUNS=10 \
MPS_BENCH_TASKS_PER_CORE=320 \
MPS_BENCH_ITER_SCALE=4 \
cargo run --release --example mps_microbench
```

Emit JSON summary (for regression dashboards or CI parsing):

```bash
MPS_BENCH_OUTPUT_JSON=1 cargo run --release --example mps_microbench
```

## Build Profiles

For comparative runs, use consistent build flags.

macOS Apple Silicon:

```bash
RUSTFLAGS="-C target-cpu=native -C lto=fat -C codegen-units=1" \
cargo run --release --example mps_microbench
```

Linux Ryzen (Zen 4):

```bash
RUSTFLAGS="-C target-cpu=znver4 -C lto=fat -C codegen-units=1" \
cargo run --release --example mps_microbench
```

## Output Interpretation

Important fields:

- `elapsed`
  Wall-time for one measured execution window.
- `multicore score`
  Composite throughput score used as a relative regression signal.
- `parallel(raw)`
  Raw aggregate execution efficiency against logical core count.
- `parallel(norm)`
  Capacity-normalized efficiency (useful for heterogeneous core setups).
- `gflops`
  Estimated floating-point throughput based on benchmark-defined FLOP accounting.
- `E-core share`
  Portion of execution time attributed to efficient cores.
- `CV`
  Stability indicator. Lower is better.
- `p95`
  Tail behavior indicator for non-mean latency/throughput swings.

Practical interpretation:

- Use median values for cross-run comparisons.
- Treat CV under `3%` as stable for local performance iteration.
- Investigate scheduler wake/routing changes if `parallel(raw)` drops while `gflops` also drops.

## Reproducibility Checklist

Before comparing two commits:

1. Keep the same `RUSTFLAGS`.
2. Keep benchmark env vars unchanged.
3. Close heavy background workloads.
4. Use the same power/governor mode.
5. Compare median and CV, not single-run peak.

## Troubleshooting Noise

- High elapsed CV with normal parallel values:
  Usually host-level jitter (power governor, thermal transitions, background load).
- High parallel CV:
  Often queue/wake imbalance or inconsistent worker scheduling behavior.
- Good parallel but low GFLOPS:
  Usually per-core compute kernel efficiency or memory bandwidth/latency limits.
