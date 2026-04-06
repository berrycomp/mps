use mps::{
    ClassExecutionMetrics, CorePreference, MpsScheduler, NativeTask, SchedulerMetrics,
    TaskPriority,
};
use std::thread;
use std::time::{Duration, Instant};

const FLOPS_PER_ITERATION: u64 = 9;
const DEFAULT_WARMUP_RUNS: usize = 2;
const DEFAULT_MEASURED_RUNS: usize = 7;
const DEFAULT_TASKS_PER_CORE: usize = 240;
const DEFAULT_ITER_SCALE: u64 = 4;
const PARK_SETTLE_MS: u64 = 5;

#[derive(Debug, Clone, Copy)]
struct BenchConfig {
    warmup_runs: usize,
    measured_runs: usize,
    tasks_per_core: usize,
    iter_scale: u64,
}

impl BenchConfig {
    fn from_env() -> Self {
        Self {
            warmup_runs: parse_env_usize("MPS_BENCH_WARMUP_RUNS", DEFAULT_WARMUP_RUNS),
            measured_runs: parse_env_usize("MPS_BENCH_MEASURED_RUNS", DEFAULT_MEASURED_RUNS)
                .max(1),
            tasks_per_core: parse_env_usize("MPS_BENCH_TASKS_PER_CORE", DEFAULT_TASKS_PER_CORE)
                .max(1),
            iter_scale: parse_env_u64("MPS_BENCH_ITER_SCALE", DEFAULT_ITER_SCALE).max(1),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WorkloadCounts {
    critical: usize,
    high: usize,
    normal: usize,
    background: usize,
}

impl WorkloadCounts {
    fn from_total(total_tasks: usize) -> Self {
        let critical = total_tasks / 10;
        let high = total_tasks * 2 / 10;
        let normal = total_tasks * 5 / 10;
        let background = total_tasks - critical - high - normal;
        Self {
            critical,
            high,
            normal,
            background,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct MultiCoreScore {
    score: u64,
    tier: &'static str,
    tasks_per_sec: f64,
    work_units_per_sec: f64,
    gflops: f64,
    parallel_efficiency_pct: f64,
    parallel_efficiency_raw_pct: f64,
    e_core_share_pct: f64,
}

#[derive(Debug, Clone, Copy)]
struct BenchSample {
    idle: bool,
    elapsed_ms: f64,
    score: MultiCoreScore,
    metrics: SchedulerMetrics,
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    mean: f64,
    median: f64,
    min: f64,
    max: f64,
    stddev: f64,
    cv_pct: f64,
}

fn main() {
    let config = BenchConfig::from_env();
    let scheduler = MpsScheduler::new();
    let topology = scheduler.topology();

    let logical = topology.logical_cores.max(1);
    let total_tasks = logical.saturating_mul(config.tasks_per_core);
    let counts = WorkloadCounts::from_total(total_tasks);

    println!(
        "Topology => logical: {}, physical: {}, hybrid: {}, P-cores: {}, E-cores: {}",
        topology.logical_cores,
        topology.physical_cores,
        topology.has_hybrid,
        topology.performance_cores,
        topology.efficient_cores
    );
    println!(
        "Task distribution => critical: {}, high: {}, normal: {}, background: {}",
        counts.critical, counts.high, counts.normal, counts.background
    );
    println!(
        "Config => warmup: {}, measured: {}, tasks/core: {}, iter-scale: {}",
        config.warmup_runs, config.measured_runs, config.tasks_per_core, config.iter_scale
    );

    let effective_cores = effective_core_capacity(topology);
    let total_runs = config.warmup_runs + config.measured_runs;
    let mut samples = Vec::with_capacity(config.measured_runs);

    for run in 0..total_runs {
        let sample = run_once(&scheduler, topology, counts, config.iter_scale, effective_cores);
        let is_warmup = run < config.warmup_runs;
        let run_no = if is_warmup {
            run + 1
        } else {
            run + 1 - config.warmup_runs
        };

        let label = if is_warmup { "Warmup" } else { "Run" };
        println!(
            "{label:>6} {run_no:02} => idle: {}, elapsed: {:.3}ms, score: {} [{}], parallel(raw): {:.2}%, parallel(norm): {:.2}%, gflops: {:.2}, e-share: {:.2}%",
            sample.idle,
            sample.elapsed_ms,
            sample.score.score,
            sample.score.tier,
            sample.score.parallel_efficiency_raw_pct,
            sample.score.parallel_efficiency_pct,
            sample.score.gflops,
            sample.score.e_core_share_pct
        );

        if !is_warmup {
            samples.push(sample);
        }
    }

    if samples.is_empty() {
        eprintln!("No measured runs were collected.");
        return;
    }

    print_summary(&samples);
}

fn run_once(
    scheduler: &MpsScheduler,
    topology: &mps::CpuTopology,
    counts: WorkloadCounts,
    iter_scale: u64,
    effective_cores: f64,
) -> BenchSample {
    let critical_base = 30_000_u64.saturating_mul(iter_scale);
    let high_base = 24_000_u64.saturating_mul(iter_scale);
    let normal_base = 18_000_u64.saturating_mul(iter_scale);
    let background_base = 12_000_u64.saturating_mul(iter_scale);

    let (critical_batch, critical_work_units, critical_flops) =
        make_work_batch(counts.critical, critical_base);
    let (high_batch, high_work_units, high_flops) = make_work_batch(counts.high, high_base);
    let (normal_batch, normal_work_units, normal_flops) =
        make_work_batch(counts.normal, normal_base);
    let (background_batch, background_work_units, background_flops) =
        make_work_batch(counts.background, background_base);

    let total_work_units = critical_work_units
        .saturating_add(high_work_units)
        .saturating_add(normal_work_units)
        .saturating_add(background_work_units);
    let total_flops = critical_flops
        .saturating_add(high_flops)
        .saturating_add(normal_flops)
        .saturating_add(background_flops);

    thread::sleep(Duration::from_millis(PARK_SETTLE_MS));

    let before = scheduler.metrics();
    let _ = scheduler.submit_batch_native_deferred_wake(
        TaskPriority::Critical,
        CorePreference::Performance,
        critical_batch,
    );
    let _ = scheduler.submit_batch_native_deferred_wake(
        TaskPriority::High,
        CorePreference::Performance,
        high_batch,
    );
    let _ = scheduler.submit_batch_native_deferred_wake(
        TaskPriority::Normal,
        CorePreference::Auto,
        normal_batch,
    );
    let _ = scheduler.submit_batch_native_deferred_wake(
        TaskPriority::Background,
        CorePreference::Efficient,
        background_batch,
    );

    let started = Instant::now();
    scheduler.wake_all_workers();
    let idle = scheduler.wait_for_idle(Duration::from_secs(60));
    let elapsed = started.elapsed();
    let after = scheduler.metrics();

    let metrics = metrics_delta(after, before);
    let score = compute_multicore_score(
        &metrics,
        elapsed,
        topology.logical_cores.max(1),
        effective_cores,
        total_work_units,
        total_flops,
    );

    BenchSample {
        idle,
        elapsed_ms: elapsed.as_secs_f64() * 1_000.0,
        score,
        metrics,
    }
}

fn print_summary(samples: &[BenchSample]) {
    let elapsed_values: Vec<f64> = samples.iter().map(|sample| sample.elapsed_ms).collect();
    let score_values: Vec<f64> = samples.iter().map(|sample| sample.score.score as f64).collect();
    let tasks_per_sec_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.score.tasks_per_sec)
        .collect();
    let work_units_per_sec_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.score.work_units_per_sec)
        .collect();
    let gflops_values: Vec<f64> = samples.iter().map(|sample| sample.score.gflops).collect();
    let parallel_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.score.parallel_efficiency_pct)
        .collect();
    let parallel_raw_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.score.parallel_efficiency_raw_pct)
        .collect();
    let e_share_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.score.e_core_share_pct)
        .collect();
    let p_runtime_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.metrics.performance.execution_ms())
        .collect();
    let e_runtime_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.metrics.efficient.execution_ms())
        .collect();
    let u_runtime_values: Vec<f64> = samples
        .iter()
        .map(|sample| sample.metrics.unknown.execution_ms())
        .collect();

    let elapsed_stats = summarize(&elapsed_values);
    let score_stats = summarize(&score_values);
    let tasks_stats = summarize(&tasks_per_sec_values);
    let work_units_stats = summarize(&work_units_per_sec_values);
    let gflops_stats = summarize(&gflops_values);
    let parallel_stats = summarize(&parallel_values);
    let parallel_raw_stats = summarize(&parallel_raw_values);
    let e_share_stats = summarize(&e_share_values);
    let p_runtime_stats = summarize(&p_runtime_values);
    let e_runtime_stats = summarize(&e_runtime_values);
    let u_runtime_stats = summarize(&u_runtime_values);

    let median_score = score_stats.median.round().max(0.0) as u64;
    println!("---");
    println!(
        "Stability => elapsed cv: {:.2}%, parallel(raw) cv: {:.2}%, gflops cv: {:.2}%",
        elapsed_stats.cv_pct, parallel_raw_stats.cv_pct, gflops_stats.cv_pct
    );
    println!(
        "Elapsed(ms) => mean: {:.3}, median: {:.3}, min: {:.3}, max: {:.3}, stddev: {:.3}",
        elapsed_stats.mean,
        elapsed_stats.median,
        elapsed_stats.min,
        elapsed_stats.max,
        elapsed_stats.stddev
    );
    println!("Multicore score => {} [{}]", median_score, score_tier(median_score));
    println!(
        "Throughput => tasks/s: {:.0}, work units/s: {:.2}M, gflops: {:.2}",
        tasks_stats.median,
        work_units_stats.median / 1_000_000.0,
        gflops_stats.median
    );
    println!(
        "Efficiency => parallel(raw): {:.2}%, parallel(norm): {:.2}%, E-core share: {:.2}%",
        parallel_raw_stats.median, parallel_stats.median, e_share_stats.median
    );
    println!(
        "Class runtime(ms) => P: {:.3}, E: {:.3}, U: {:.3}",
        p_runtime_stats.median, e_runtime_stats.median, u_runtime_stats.median
    );
}

fn summarize(values: &[f64]) -> Stats {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    let len = sorted.len();
    let sum: f64 = sorted.iter().sum();
    let mean = sum / len as f64;
    let median = if len % 2 == 1 {
        sorted[len / 2]
    } else {
        (sorted[len / 2 - 1] + sorted[len / 2]) * 0.5
    };
    let min = sorted[0];
    let max = sorted[len - 1];

    let variance = sorted
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / len as f64;
    let stddev = variance.sqrt();
    let cv_pct = if mean.abs() <= f64::EPSILON {
        0.0
    } else {
        (stddev / mean) * 100.0
    };

    Stats {
        mean,
        median,
        min,
        max,
        stddev,
        cv_pct,
    }
}

fn metrics_delta(after: SchedulerMetrics, before: SchedulerMetrics) -> SchedulerMetrics {
    SchedulerMetrics {
        submitted: after.submitted.saturating_sub(before.submitted),
        completed: after.completed.saturating_sub(before.completed),
        failed: after.failed.saturating_sub(before.failed),
        queue_depth: after.queue_depth,
        performance: class_metrics_delta(after.performance, before.performance),
        efficient: class_metrics_delta(after.efficient, before.efficient),
        unknown: class_metrics_delta(after.unknown, before.unknown),
        simd_backend: after.simd_backend,
        simd_lanes: after.simd_lanes,
    }
}

fn class_metrics_delta(
    after: ClassExecutionMetrics,
    before: ClassExecutionMetrics,
) -> ClassExecutionMetrics {
    ClassExecutionMetrics {
        executed_tasks: after.executed_tasks.saturating_sub(before.executed_tasks),
        execution_time_ns: after.execution_time_ns.saturating_sub(before.execution_time_ns),
    }
}

fn make_work_batch(count: usize, base_iterations: u64) -> (Vec<NativeTask>, u64, u64) {
    let total_iterations = (count as u64)
        .saturating_mul(base_iterations)
        .saturating_add(sum_mod_sequence(count as u64, 97));
    let total_flops = total_iterations.saturating_mul(FLOPS_PER_ITERATION);

    let tasks = (0..count)
        .map(|task_index| {
            Box::new(move || {
                let mut a = 1.0_f64 + task_index as f64 * 1e-6;
                let mut b = 0.5_f64 + task_index as f64 * 5e-7;
                let mut c = 0.25_f64 + task_index as f64 * 2e-7;
                let mut acc = 0.0_f64;
                let iterations = base_iterations + (task_index as u64 % 97);

                for _ in 0..iterations {
                    // 9 FLOPs/iteration:
                    // a = a*k + b (2), b = b*k + c (2), c = c*k + a (2), acc += a*b + c (3)
                    a = a * 1.000_000_119_209_29 + b;
                    b = b * 0.999_999_880_790_71 + c;
                    c = c * 1.000_000_357_627_87 + a;
                    acc += a * b + c;
                }

                std::hint::black_box((a, b, c, acc));
            }) as NativeTask
        })
        .collect();

    (tasks, total_iterations, total_flops)
}

fn sum_mod_sequence(count: u64, modulo: u64) -> u64 {
    if modulo == 0 {
        return 0;
    }

    let full_cycles = count / modulo;
    let remainder = count % modulo;
    let cycle_sum = (modulo.saturating_sub(1)).saturating_mul(modulo) / 2;
    let remainder_sum = remainder.saturating_sub(1).saturating_mul(remainder) / 2;

    full_cycles
        .saturating_mul(cycle_sum)
        .saturating_add(remainder_sum)
}

fn compute_multicore_score(
    metrics: &SchedulerMetrics,
    elapsed: Duration,
    logical_cores: usize,
    effective_cores: f64,
    total_work_units: u64,
    total_flops: u64,
) -> MultiCoreScore {
    let elapsed_secs = elapsed.as_secs_f64().max(1e-9);
    let tasks_per_sec = metrics.completed as f64 / elapsed_secs;
    let work_units_per_sec = total_work_units as f64 / elapsed_secs;
    let gflops = total_flops as f64 / elapsed_secs / 1_000_000_000.0;

    let parallel_efficiency_raw_pct = metrics.parallel_efficiency_pct(elapsed, logical_cores);
    let parallel_efficiency_pct =
        parallel_efficiency_pct_for_capacity(metrics, elapsed, effective_cores);
    let e_core_share_pct = metrics.e_core_share_pct();

    // Scoring model:
    // - base term: workload throughput
    // - boost term: multicore parallel efficiency
    let base_throughput_score = work_units_per_sec / 10_000.0;
    let efficiency_boost = 0.70 + (parallel_efficiency_pct / 100.0) * 0.30;
    let score = (base_throughput_score * efficiency_boost).round().max(0.0) as u64;

    MultiCoreScore {
        score,
        tier: score_tier(score),
        tasks_per_sec,
        work_units_per_sec,
        gflops,
        parallel_efficiency_pct,
        parallel_efficiency_raw_pct,
        e_core_share_pct,
    }
}

fn effective_core_capacity(topology: &mps::CpuTopology) -> f64 {
    // Apple hybrid SoCs have materially slower E-cores than P-cores.
    // Normalizing by an effective-core capacity produces a steadier
    // multicore efficiency signal than treating all logical cores as equal.
    if topology.has_hybrid && topology.vendor.as_deref() == Some("Apple") {
        let p = topology.performance_cores.max(1) as f64;
        let e = topology.efficient_cores as f64;
        p + e * 0.66
    } else {
        topology.logical_cores.max(1) as f64
    }
}

fn parallel_efficiency_pct_for_capacity(
    metrics: &SchedulerMetrics,
    elapsed: Duration,
    effective_cores: f64,
) -> f64 {
    let elapsed_ns = elapsed.as_nanos() as f64;
    let core_capacity = effective_cores.max(1.0);
    if elapsed_ns <= 0.0 {
        return 0.0;
    }

    let theoretical_ns = elapsed_ns * core_capacity;
    let actual_ns = metrics.aggregate_execution_ns() as f64;
    (actual_ns / theoretical_ns).clamp(0.0, 1.0) * 100.0
}

fn score_tier(score: u64) -> &'static str {
    match score {
        220_000.. => "S",
        160_000.. => "A",
        110_000.. => "B",
        70_000.. => "C",
        _ => "D",
    }
}

fn parse_env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn parse_env_u64(key: &str, default: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default)
}
