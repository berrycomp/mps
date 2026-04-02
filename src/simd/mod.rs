//! Runtime-dispatched SIMD kernels for MPS.
//!
//! The API is intentionally small and safe: callers pass SoA slices and the
//! selected backend executes the hot kernels through a dispatch table chosen
//! once during scheduler/dispatcher initialization.

mod scalar;

#[cfg(target_arch = "aarch64")]
mod aarch64_neon;
#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
mod powerpc_altivec;
#[cfg(target_arch = "x86_64")]
mod x86_64_avx512;

/// Runtime-selected SIMD backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdBackendKind {
    /// Portable scalar fallback.
    Scalar,
    /// x86_64 AVX-512 backend.
    X86Avx512,
    /// AArch64 NEON backend.
    Aarch64Neon,
    /// PowerPC VMX/AltiVec backend.
    PowerPcAltivec,
}

impl Default for SimdBackendKind {
    fn default() -> Self {
        Self::Scalar
    }
}

/// Snapshot of SIMD capabilities detected for the current process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SimdCapabilities {
    /// Selected runtime backend.
    pub backend: SimdBackendKind,
    /// Number of f32 lanes processed per vector chunk.
    pub simd_lanes: usize,
    /// Whether AVX-512F is available on this host.
    pub avx512f: bool,
    /// Whether NEON is available on this host.
    pub neon: bool,
    /// Whether VMX/AltiVec is available on this host.
    pub altivec: bool,
}

impl Default for SimdCapabilities {
    fn default() -> Self {
        Self {
            backend: SimdBackendKind::Scalar,
            simd_lanes: 1,
            avx512f: false,
            neon: false,
            altivec: false,
        }
    }
}

/// Read-only transform data in SoA form.
#[derive(Debug, Clone, Copy)]
pub struct TransformSoaRef<'a> {
    pub position_x: &'a [f32],
    pub position_y: &'a [f32],
    pub position_z: &'a [f32],
    pub rotation_x: &'a [f32],
    pub rotation_y: &'a [f32],
    pub rotation_z: &'a [f32],
    pub rotation_w: &'a [f32],
}

/// Mutable transform data in SoA form.
#[derive(Debug)]
pub struct TransformSoaMut<'a> {
    pub position_x: &'a mut [f32],
    pub position_y: &'a mut [f32],
    pub position_z: &'a mut [f32],
    pub rotation_x: &'a mut [f32],
    pub rotation_y: &'a mut [f32],
    pub rotation_z: &'a mut [f32],
    pub rotation_w: &'a mut [f32],
}

/// Read-only velocity data in SoA form.
#[derive(Debug, Clone, Copy)]
pub struct VelocitySoaRef<'a> {
    pub velocity_x: &'a [f32],
    pub velocity_y: &'a [f32],
    pub velocity_z: &'a [f32],
}

/// Mutable position data in SoA form.
#[derive(Debug)]
pub struct PositionSoaMut<'a> {
    pub position_x: &'a mut [f32],
    pub position_y: &'a mut [f32],
    pub position_z: &'a mut [f32],
}

/// Read-only AABB center/half-extent SoA slices.
#[derive(Debug, Clone, Copy)]
pub struct AabbInputSoaRef<'a> {
    pub center_x: &'a [f32],
    pub center_y: &'a [f32],
    pub center_z: &'a [f32],
    pub half_x: &'a [f32],
    pub half_y: &'a [f32],
    pub half_z: &'a [f32],
}

/// Mutable AABB min/max SoA slices.
#[derive(Debug)]
pub struct AabbBoundsSoaMut<'a> {
    pub min_x: &'a mut [f32],
    pub min_y: &'a mut [f32],
    pub min_z: &'a mut [f32],
    pub max_x: &'a mut [f32],
    pub max_y: &'a mut [f32],
    pub max_z: &'a mut [f32],
}

/// Read-only AABB min/max SoA slices.
#[derive(Debug, Clone, Copy)]
pub struct AabbBoundsSoaRef<'a> {
    pub min_x: &'a [f32],
    pub min_y: &'a [f32],
    pub min_z: &'a [f32],
    pub max_x: &'a [f32],
    pub max_y: &'a [f32],
    pub max_z: &'a [f32],
}

type CopyTransformSoaFn = unsafe fn(
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    usize,
);

type IntegratePositionsFn = unsafe fn(
    *mut f32,
    *mut f32,
    *mut f32,
    *const f32,
    *const f32,
    *const f32,
    f32,
    usize,
);

type AabbMinMaxFn = unsafe fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    *mut f32,
    usize,
);

type AabbOverlapMaskFn = unsafe fn(
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *const f32,
    *mut u32,
    usize,
);

/// Concrete kernel table selected once per process/runtime instance.
#[derive(Clone, Copy)]
pub struct SimdKernelSet {
    capabilities: SimdCapabilities,
    copy_transform_soa_fn: CopyTransformSoaFn,
    integrate_positions_fn: IntegratePositionsFn,
    aabb_min_max_fn: AabbMinMaxFn,
    aabb_overlap_mask_fn: AabbOverlapMaskFn,
}

impl std::fmt::Debug for SimdKernelSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SimdKernelSet")
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

impl Default for SimdKernelSet {
    fn default() -> Self {
        Self::scalar()
    }
}

impl SimdKernelSet {
    /// Build a scalar-only kernel table.
    pub fn scalar() -> Self {
        Self::for_capabilities(SimdCapabilities::default())
    }

    /// Build the best kernel table supported by the current process.
    pub fn detect_runtime() -> Self {
        Self::for_capabilities(detect_runtime_simd())
    }

    /// Build a kernel table from an explicit capability snapshot.
    pub fn for_capabilities(capabilities: SimdCapabilities) -> Self {
        match capabilities.backend {
            SimdBackendKind::Scalar => Self {
                capabilities,
                copy_transform_soa_fn: scalar::copy_transform_soa,
                integrate_positions_fn: scalar::integrate_positions,
                aabb_min_max_fn: scalar::aabb_min_max,
                aabb_overlap_mask_fn: scalar::aabb_overlap_mask,
            },
            #[cfg(target_arch = "x86_64")]
            SimdBackendKind::X86Avx512 => Self {
                capabilities,
                copy_transform_soa_fn: x86_64_avx512::copy_transform_soa,
                integrate_positions_fn: x86_64_avx512::integrate_positions,
                aabb_min_max_fn: x86_64_avx512::aabb_min_max,
                aabb_overlap_mask_fn: scalar::aabb_overlap_mask,
            },
            #[cfg(target_arch = "aarch64")]
            SimdBackendKind::Aarch64Neon => Self {
                capabilities,
                copy_transform_soa_fn: aarch64_neon::copy_transform_soa,
                integrate_positions_fn: aarch64_neon::integrate_positions,
                aabb_min_max_fn: aarch64_neon::aabb_min_max,
                aabb_overlap_mask_fn: scalar::aabb_overlap_mask,
            },
            #[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
            SimdBackendKind::PowerPcAltivec => Self {
                capabilities,
                copy_transform_soa_fn: powerpc_altivec::copy_transform_soa,
                integrate_positions_fn: powerpc_altivec::integrate_positions,
                aabb_min_max_fn: powerpc_altivec::aabb_min_max,
                aabb_overlap_mask_fn: scalar::aabb_overlap_mask,
            },
            #[allow(unreachable_patterns)]
            _ => Self::scalar(),
        }
    }

    /// Return the selected backend kind.
    pub fn backend(&self) -> SimdBackendKind {
        self.capabilities.backend
    }

    /// Return the number of f32 lanes processed per vector step.
    pub fn lanes_f32(&self) -> usize {
        self.capabilities.simd_lanes
    }

    /// Return the full capability snapshot.
    pub fn capabilities(&self) -> SimdCapabilities {
        self.capabilities
    }

    /// Copy a transform SoA batch from `src` to `dst`.
    pub fn copy_transform_soa(&self, dst: &mut TransformSoaMut<'_>, src: TransformSoaRef<'_>) -> usize {
        let len = min_transform_len_mut(dst, &src);
        if len == 0 {
            return 0;
        }

        unsafe {
            (self.copy_transform_soa_fn)(
                dst.position_x.as_mut_ptr(),
                dst.position_y.as_mut_ptr(),
                dst.position_z.as_mut_ptr(),
                dst.rotation_x.as_mut_ptr(),
                dst.rotation_y.as_mut_ptr(),
                dst.rotation_z.as_mut_ptr(),
                dst.rotation_w.as_mut_ptr(),
                src.position_x.as_ptr(),
                src.position_y.as_ptr(),
                src.position_z.as_ptr(),
                src.rotation_x.as_ptr(),
                src.rotation_y.as_ptr(),
                src.rotation_z.as_ptr(),
                src.rotation_w.as_ptr(),
                len,
            );
        }
        len
    }

    /// Integrate `position += velocity * dt` over a SoA batch.
    pub fn integrate_positions(
        &self,
        positions: &mut PositionSoaMut<'_>,
        velocities: VelocitySoaRef<'_>,
        dt_seconds: f32,
    ) -> usize {
        let len = min_position_velocity_len(positions, &velocities);
        if len == 0 {
            return 0;
        }

        unsafe {
            (self.integrate_positions_fn)(
                positions.position_x.as_mut_ptr(),
                positions.position_y.as_mut_ptr(),
                positions.position_z.as_mut_ptr(),
                velocities.velocity_x.as_ptr(),
                velocities.velocity_y.as_ptr(),
                velocities.velocity_z.as_ptr(),
                dt_seconds,
                len,
            );
        }
        len
    }

    /// Compute AABB min/max bounds from centers and half extents.
    pub fn aabb_min_max(&self, dst: &mut AabbBoundsSoaMut<'_>, src: AabbInputSoaRef<'_>) -> usize {
        let len = min_aabb_input_len(dst, &src);
        if len == 0 {
            return 0;
        }

        unsafe {
            (self.aabb_min_max_fn)(
                src.center_x.as_ptr(),
                src.center_y.as_ptr(),
                src.center_z.as_ptr(),
                src.half_x.as_ptr(),
                src.half_y.as_ptr(),
                src.half_z.as_ptr(),
                dst.min_x.as_mut_ptr(),
                dst.min_y.as_mut_ptr(),
                dst.min_z.as_mut_ptr(),
                dst.max_x.as_mut_ptr(),
                dst.max_y.as_mut_ptr(),
                dst.max_z.as_mut_ptr(),
                len,
            );
        }
        len
    }

    /// Compute pairwise overlap mask for two AABB SoA batches.
    pub fn aabb_overlap_mask(
        &self,
        a: AabbBoundsSoaRef<'_>,
        b: AabbBoundsSoaRef<'_>,
        output_mask: &mut [u32],
    ) -> usize {
        let len = min_aabb_overlap_len(&a, &b, output_mask);
        if len == 0 {
            return 0;
        }

        unsafe {
            (self.aabb_overlap_mask_fn)(
                a.min_x.as_ptr(),
                a.min_y.as_ptr(),
                a.min_z.as_ptr(),
                a.max_x.as_ptr(),
                a.max_y.as_ptr(),
                a.max_z.as_ptr(),
                b.min_x.as_ptr(),
                b.min_y.as_ptr(),
                b.min_z.as_ptr(),
                b.max_x.as_ptr(),
                b.max_y.as_ptr(),
                b.max_z.as_ptr(),
                output_mask.as_mut_ptr(),
                len,
            );
        }
        len
    }
}

/// Detect the best SIMD backend available for this process.
pub fn detect_runtime_simd() -> SimdCapabilities {
    if std::env::var_os("MPS_SIMD_FORCE_SCALAR").is_some() {
        return SimdCapabilities::default();
    }

    detect_arch_runtime_simd()
}

#[cfg(target_arch = "x86_64")]
fn detect_arch_runtime_simd() -> SimdCapabilities {
    if std::arch::is_x86_feature_detected!("avx512f") {
        return SimdCapabilities {
            backend: SimdBackendKind::X86Avx512,
            simd_lanes: 16,
            avx512f: true,
            neon: false,
            altivec: false,
        };
    }

    SimdCapabilities::default()
}

#[cfg(target_arch = "aarch64")]
fn detect_arch_runtime_simd() -> SimdCapabilities {
    if std::arch::is_aarch64_feature_detected!("neon") {
        return SimdCapabilities {
            backend: SimdBackendKind::Aarch64Neon,
            simd_lanes: 4,
            avx512f: false,
            neon: true,
            altivec: false,
        };
    }

    SimdCapabilities::default()
}

#[cfg(target_arch = "powerpc")]
fn detect_arch_runtime_simd() -> SimdCapabilities {
    if std::arch::is_powerpc_feature_detected!("altivec") {
        return SimdCapabilities {
            backend: SimdBackendKind::PowerPcAltivec,
            simd_lanes: 4,
            avx512f: false,
            neon: false,
            altivec: true,
        };
    }

    SimdCapabilities::default()
}

#[cfg(target_arch = "powerpc64")]
fn detect_arch_runtime_simd() -> SimdCapabilities {
    if std::arch::is_powerpc64_feature_detected!("altivec") {
        return SimdCapabilities {
            backend: SimdBackendKind::PowerPcAltivec,
            simd_lanes: 4,
            avx512f: false,
            neon: false,
            altivec: true,
        };
    }

    SimdCapabilities::default()
}

#[cfg(not(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "powerpc",
    target_arch = "powerpc64"
)))]
fn detect_arch_runtime_simd() -> SimdCapabilities {
    SimdCapabilities::default()
}

fn min_transform_len_mut(dst: &TransformSoaMut<'_>, src: &TransformSoaRef<'_>) -> usize {
    dst.position_x
        .len()
        .min(dst.position_y.len())
        .min(dst.position_z.len())
        .min(dst.rotation_x.len())
        .min(dst.rotation_y.len())
        .min(dst.rotation_z.len())
        .min(dst.rotation_w.len())
        .min(src.position_x.len())
        .min(src.position_y.len())
        .min(src.position_z.len())
        .min(src.rotation_x.len())
        .min(src.rotation_y.len())
        .min(src.rotation_z.len())
        .min(src.rotation_w.len())
}

fn min_position_velocity_len(
    positions: &PositionSoaMut<'_>,
    velocities: &VelocitySoaRef<'_>,
) -> usize {
    positions
        .position_x
        .len()
        .min(positions.position_y.len())
        .min(positions.position_z.len())
        .min(velocities.velocity_x.len())
        .min(velocities.velocity_y.len())
        .min(velocities.velocity_z.len())
}

fn min_aabb_input_len(dst: &AabbBoundsSoaMut<'_>, src: &AabbInputSoaRef<'_>) -> usize {
    dst.min_x
        .len()
        .min(dst.min_y.len())
        .min(dst.min_z.len())
        .min(dst.max_x.len())
        .min(dst.max_y.len())
        .min(dst.max_z.len())
        .min(src.center_x.len())
        .min(src.center_y.len())
        .min(src.center_z.len())
        .min(src.half_x.len())
        .min(src.half_y.len())
        .min(src.half_z.len())
}

fn min_aabb_overlap_len(a: &AabbBoundsSoaRef<'_>, b: &AabbBoundsSoaRef<'_>, output: &[u32]) -> usize {
    output
        .len()
        .min(a.min_x.len())
        .min(a.min_y.len())
        .min(a.min_z.len())
        .min(a.max_x.len())
        .min(a.max_y.len())
        .min(a.max_z.len())
        .min(b.min_x.len())
        .min(b.min_y.len())
        .min(b.min_z.len())
        .min(b.max_x.len())
        .min(b.max_y.len())
        .min(b.max_z.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1.0e-5;

    #[test]
    fn scalar_copy_and_integrate_match_expected_values() {
        let kernels = SimdKernelSet::scalar();
        let source = [0.0_f32, 1.0, 2.0, 3.0, 4.0];
        let ry = [0.5_f32; 5];
        let rz = [1.0_f32; 5];
        let rw = [1.0_f32; 5];
        let mut px = [0.0_f32; 5];
        let mut py = [0.0_f32; 5];
        let mut pz = [0.0_f32; 5];
        let mut rx = [0.0_f32; 5];
        let mut dst_ry = [0.0_f32; 5];
        let mut dst_rz = [0.0_f32; 5];
        let mut dst_rw = [0.0_f32; 5];

        let copied = kernels.copy_transform_soa(
            &mut TransformSoaMut {
                position_x: &mut px,
                position_y: &mut py,
                position_z: &mut pz,
                rotation_x: &mut rx,
                rotation_y: &mut dst_ry,
                rotation_z: &mut dst_rz,
                rotation_w: &mut dst_rw,
            },
            TransformSoaRef {
                position_x: &source,
                position_y: &source,
                position_z: &source,
                rotation_x: &source,
                rotation_y: &ry,
                rotation_z: &rz,
                rotation_w: &rw,
            },
        );
        assert_eq!(copied, 5);
        assert_eq!(px, source);
        assert_eq!(dst_ry, ry);

        let velocity = [2.0_f32; 5];
        let integrated = kernels.integrate_positions(
            &mut PositionSoaMut {
                position_x: &mut px,
                position_y: &mut py,
                position_z: &mut pz,
            },
            VelocitySoaRef {
                velocity_x: &velocity,
                velocity_y: &velocity,
                velocity_z: &velocity,
            },
            0.25,
        );
        assert_eq!(integrated, 5);
        assert!((px[4] - 4.5).abs() <= EPSILON);
        assert!((py[1] - 1.5).abs() <= EPSILON);
        assert!((pz[0] - 0.5).abs() <= EPSILON);
    }

    #[test]
    fn scalar_aabb_helpers_match_overlap_expectations() {
        let kernels = SimdKernelSet::scalar();
        let center = [0.0_f32, 2.0, 8.0, -4.0];
        let half = [1.0_f32, 1.5, 0.5, 2.0];
        let mut min_x = [0.0_f32; 4];
        let mut min_y = [0.0_f32; 4];
        let mut min_z = [0.0_f32; 4];
        let mut max_x = [0.0_f32; 4];
        let mut max_y = [0.0_f32; 4];
        let mut max_z = [0.0_f32; 4];

        let written = kernels.aabb_min_max(
            &mut AabbBoundsSoaMut {
                min_x: &mut min_x,
                min_y: &mut min_y,
                min_z: &mut min_z,
                max_x: &mut max_x,
                max_y: &mut max_y,
                max_z: &mut max_z,
            },
            AabbInputSoaRef {
                center_x: &center,
                center_y: &center,
                center_z: &center,
                half_x: &half,
                half_y: &half,
                half_z: &half,
            },
        );
        assert_eq!(written, 4);

        let mut mask = [0_u32; 4];
        let tested = kernels.aabb_overlap_mask(
            AabbBoundsSoaRef {
                min_x: &min_x,
                min_y: &min_y,
                min_z: &min_z,
                max_x: &max_x,
                max_y: &max_y,
                max_z: &max_z,
            },
            AabbBoundsSoaRef {
                min_x: &[-0.5, 10.0, 7.8, -6.0],
                min_y: &[-0.5, 10.0, 7.8, -6.0],
                min_z: &[-0.5, 10.0, 7.8, -6.0],
                max_x: &[0.5, 11.0, 8.2, -5.5],
                max_y: &[0.5, 11.0, 8.2, -5.5],
                max_z: &[0.5, 11.0, 8.2, -5.5],
            },
            &mut mask,
        );
        assert_eq!(tested, 4);
        assert_eq!(mask, [1, 0, 1, 1]);
    }

    #[test]
    fn runtime_dispatch_returns_a_valid_kernel_table() {
        let kernels = SimdKernelSet::detect_runtime();
        assert!(kernels.lanes_f32() >= 1);
        match kernels.backend() {
            SimdBackendKind::Scalar
            | SimdBackendKind::X86Avx512
            | SimdBackendKind::Aarch64Neon
            | SimdBackendKind::PowerPcAltivec => {}
        }
    }
}
