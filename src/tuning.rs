//! Performance-profile tuning and compression configuration for MPS.

/// High-level performance profile exposed to the runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpsPerformanceProfile {
    /// Conservative settings for typical workloads.
    Balanced,
    /// Higher worker aggression and larger queues.
    Aggressive,
    /// Maximum throughput, lowest latency tuning.
    Heimdall,
}

/// Knobs applied to a [`TaskDispatcherConfig`] via [`apply_tuning`].
#[derive(Debug, Clone, Copy)]
pub struct MpsTuningProfile {
    /// Spin iterations before yielding.
    pub spin_iterations: u32,
    /// Yield iterations before futex wait.
    pub yield_iterations: u32,
    /// Nice value for performance cores.
    pub performance_nice: i32,
    /// Nice value for efficient cores.
    pub efficient_nice: i32,
    /// Nice value for unknown-class cores.
    pub unknown_nice: i32,
    /// Use strict one-core affinity on Linux.
    pub strict_affinity: bool,
    /// Attempt `SCHED_FIFO` on Linux for worker threads.
    pub enable_realtime_policy: bool,
    /// Multiplier applied to the default queue capacity.
    pub queue_capacity_multiplier: usize,
    /// Chunk size used when the trigger requests zero.
    pub default_chunk_size: usize,
}

impl MpsTuningProfile {
    /// Build a tuning profile from a runtime-facing performance profile.
    pub fn from_profile(profile: MpsPerformanceProfile) -> Self {
        match profile {
            MpsPerformanceProfile::Balanced => Self {
                spin_iterations: 2_048,
                yield_iterations: 64,
                performance_nice: -8,
                efficient_nice: -4,
                unknown_nice: -6,
                strict_affinity: true,
                enable_realtime_policy: true,
                queue_capacity_multiplier: 1,
                default_chunk_size: 256,
            },
            MpsPerformanceProfile::Aggressive => Self {
                spin_iterations: 4_096,
                yield_iterations: 32,
                performance_nice: -10,
                efficient_nice: -6,
                unknown_nice: -8,
                strict_affinity: true,
                enable_realtime_policy: true,
                queue_capacity_multiplier: 2,
                default_chunk_size: 512,
            },
            MpsPerformanceProfile::Heimdall => Self {
                spin_iterations: 8_192,
                yield_iterations: 16,
                performance_nice: -12,
                efficient_nice: -8,
                unknown_nice: -10,
                strict_affinity: true,
                enable_realtime_policy: true,
                queue_capacity_multiplier: 4,
                default_chunk_size: 1_024,
            },
        }
    }
}

/// Compression settings carried by the dispatcher config.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpsCompressionConfig {
    /// Whether transform/compression pass is enabled.
    pub enabled: bool,
    /// Compression level hint (codec-specific).
    pub level: u8,
}

impl MpsCompressionConfig {
    /// Read compression settings from well-known `TILELINE_*` environment variables.
    pub fn from_tileline_env() -> Self {
        let enabled = std::env::var("TILELINE_MPS_COMPRESSION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let level = std::env::var("TILELINE_MPS_COMPRESSION_LEVEL")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(3);
        Self { enabled, level }
    }
}

/// Metrics snapshot for the compression sub-system.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompressionMetrics {
    /// Total compressed bytes written.
    pub compressed_bytes: u64,
    /// Total uncompressed bytes fed in.
    pub uncompressed_bytes: u64,
    /// Observed compression ratio (0.0 when no data).
    pub compression_ratio: f32,
}
