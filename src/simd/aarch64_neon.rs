#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::{
    float32x4_t, uint32x4_t, vaddq_f32, vandq_u32, vcgeq_f32, vcleq_f32, vdupq_n_f32, vfmaq_f32,
    vld1q_f32, vshrq_n_u32, vst1q_f32, vst1q_u32, vsubq_f32,
};

const LANES: usize = 4;

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn copy_transform_soa(
    dst_px: *mut f32,
    dst_py: *mut f32,
    dst_pz: *mut f32,
    dst_rx: *mut f32,
    dst_ry: *mut f32,
    dst_rz: *mut f32,
    dst_rw: *mut f32,
    src_px: *const f32,
    src_py: *const f32,
    src_pz: *const f32,
    src_rx: *const f32,
    src_ry: *const f32,
    src_rz: *const f32,
    src_rw: *const f32,
    len: usize,
) {
    let mut index = 0;
    while index + LANES <= len {
        vst1q_f32(dst_px.add(index), vld1q_f32(src_px.add(index)));
        vst1q_f32(dst_py.add(index), vld1q_f32(src_py.add(index)));
        vst1q_f32(dst_pz.add(index), vld1q_f32(src_pz.add(index)));
        vst1q_f32(dst_rx.add(index), vld1q_f32(src_rx.add(index)));
        vst1q_f32(dst_ry.add(index), vld1q_f32(src_ry.add(index)));
        vst1q_f32(dst_rz.add(index), vld1q_f32(src_rz.add(index)));
        vst1q_f32(dst_rw.add(index), vld1q_f32(src_rw.add(index)));
        index += LANES;
    }
    crate::simd::scalar::copy_transform_soa(
        dst_px.add(index),
        dst_py.add(index),
        dst_pz.add(index),
        dst_rx.add(index),
        dst_ry.add(index),
        dst_rz.add(index),
        dst_rw.add(index),
        src_px.add(index),
        src_py.add(index),
        src_pz.add(index),
        src_rx.add(index),
        src_ry.add(index),
        src_rz.add(index),
        src_rw.add(index),
        len - index,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn integrate_positions(
    position_x: *mut f32,
    position_y: *mut f32,
    position_z: *mut f32,
    velocity_x: *const f32,
    velocity_y: *const f32,
    velocity_z: *const f32,
    dt_seconds: f32,
    len: usize,
) {
    let dt: float32x4_t = vdupq_n_f32(dt_seconds);
    let mut index = 0;
    while index + LANES <= len {
        let px = vld1q_f32(position_x.add(index));
        let py = vld1q_f32(position_y.add(index));
        let pz = vld1q_f32(position_z.add(index));
        let vx = vld1q_f32(velocity_x.add(index));
        let vy = vld1q_f32(velocity_y.add(index));
        let vz = vld1q_f32(velocity_z.add(index));

        vst1q_f32(position_x.add(index), vfmaq_f32(px, vx, dt));
        vst1q_f32(position_y.add(index), vfmaq_f32(py, vy, dt));
        vst1q_f32(position_z.add(index), vfmaq_f32(pz, vz, dt));
        index += LANES;
    }
    crate::simd::scalar::integrate_positions(
        position_x.add(index),
        position_y.add(index),
        position_z.add(index),
        velocity_x.add(index),
        velocity_y.add(index),
        velocity_z.add(index),
        dt_seconds,
        len - index,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn aabb_min_max(
    center_x: *const f32,
    center_y: *const f32,
    center_z: *const f32,
    half_x: *const f32,
    half_y: *const f32,
    half_z: *const f32,
    min_x: *mut f32,
    min_y: *mut f32,
    min_z: *mut f32,
    max_x: *mut f32,
    max_y: *mut f32,
    max_z: *mut f32,
    len: usize,
) {
    let mut index = 0;
    while index + LANES <= len {
        let cx = vld1q_f32(center_x.add(index));
        let cy = vld1q_f32(center_y.add(index));
        let cz = vld1q_f32(center_z.add(index));
        let hx = vld1q_f32(half_x.add(index));
        let hy = vld1q_f32(half_y.add(index));
        let hz = vld1q_f32(half_z.add(index));

        vst1q_f32(min_x.add(index), vsubq_f32(cx, hx));
        vst1q_f32(min_y.add(index), vsubq_f32(cy, hy));
        vst1q_f32(min_z.add(index), vsubq_f32(cz, hz));
        vst1q_f32(max_x.add(index), vaddq_f32(cx, hx));
        vst1q_f32(max_y.add(index), vaddq_f32(cy, hy));
        vst1q_f32(max_z.add(index), vaddq_f32(cz, hz));
        index += LANES;
    }
    crate::simd::scalar::aabb_min_max(
        center_x.add(index),
        center_y.add(index),
        center_z.add(index),
        half_x.add(index),
        half_y.add(index),
        half_z.add(index),
        min_x.add(index),
        min_y.add(index),
        min_z.add(index),
        max_x.add(index),
        max_y.add(index),
        max_z.add(index),
        len - index,
    );
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn aabb_overlap_mask(
    a_min_x: *const f32,
    a_min_y: *const f32,
    a_min_z: *const f32,
    a_max_x: *const f32,
    a_max_y: *const f32,
    a_max_z: *const f32,
    b_min_x: *const f32,
    b_min_y: *const f32,
    b_min_z: *const f32,
    b_max_x: *const f32,
    b_max_y: *const f32,
    b_max_z: *const f32,
    out_mask: *mut u32,
    len: usize,
) {
    let mut index = 0;
    while index + LANES <= len {
        let ax_min = vld1q_f32(a_min_x.add(index));
        let ay_min = vld1q_f32(a_min_y.add(index));
        let az_min = vld1q_f32(a_min_z.add(index));
        let ax_max = vld1q_f32(a_max_x.add(index));
        let ay_max = vld1q_f32(a_max_y.add(index));
        let az_max = vld1q_f32(a_max_z.add(index));

        let bx_min = vld1q_f32(b_min_x.add(index));
        let by_min = vld1q_f32(b_min_y.add(index));
        let bz_min = vld1q_f32(b_min_z.add(index));
        let bx_max = vld1q_f32(b_max_x.add(index));
        let by_max = vld1q_f32(b_max_y.add(index));
        let bz_max = vld1q_f32(b_max_z.add(index));

        let x_le = vcleq_f32(ax_min, bx_max);
        let x_ge = vcgeq_f32(ax_max, bx_min);
        let y_le = vcleq_f32(ay_min, by_max);
        let y_ge = vcgeq_f32(ay_max, by_min);
        let z_le = vcleq_f32(az_min, bz_max);
        let z_ge = vcgeq_f32(az_max, bz_min);
        let overlap_mask: uint32x4_t = vandq_u32(
            vandq_u32(x_le, x_ge),
            vandq_u32(vandq_u32(y_le, y_ge), vandq_u32(z_le, z_ge)),
        );
        let normalized = vshrq_n_u32(overlap_mask, 31);
        vst1q_u32(out_mask.add(index), normalized);
        index += LANES;
    }

    crate::simd::scalar::aabb_overlap_mask(
        a_min_x.add(index),
        a_min_y.add(index),
        a_min_z.add(index),
        a_max_x.add(index),
        a_max_y.add(index),
        a_max_z.add(index),
        b_min_x.add(index),
        b_min_y.add(index),
        b_min_z.add(index),
        b_max_x.add(index),
        b_max_y.add(index),
        b_max_z.add(index),
        out_mask.add(index),
        len - index,
    );
}
