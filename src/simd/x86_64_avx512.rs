#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{
    _mm512_add_ps, _mm512_loadu_ps, _mm512_mul_ps, _mm512_set1_ps, _mm512_storeu_ps,
    _mm512_sub_ps,
};

const LANES: usize = 16;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
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
        _mm512_storeu_ps(dst_px.add(index), _mm512_loadu_ps(src_px.add(index)));
        _mm512_storeu_ps(dst_py.add(index), _mm512_loadu_ps(src_py.add(index)));
        _mm512_storeu_ps(dst_pz.add(index), _mm512_loadu_ps(src_pz.add(index)));
        _mm512_storeu_ps(dst_rx.add(index), _mm512_loadu_ps(src_rx.add(index)));
        _mm512_storeu_ps(dst_ry.add(index), _mm512_loadu_ps(src_ry.add(index)));
        _mm512_storeu_ps(dst_rz.add(index), _mm512_loadu_ps(src_rz.add(index)));
        _mm512_storeu_ps(dst_rw.add(index), _mm512_loadu_ps(src_rw.add(index)));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
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
    let dt = _mm512_set1_ps(dt_seconds);
    let mut index = 0;
    while index + LANES <= len {
        let px = _mm512_loadu_ps(position_x.add(index));
        let py = _mm512_loadu_ps(position_y.add(index));
        let pz = _mm512_loadu_ps(position_z.add(index));
        let vx = _mm512_loadu_ps(velocity_x.add(index));
        let vy = _mm512_loadu_ps(velocity_y.add(index));
        let vz = _mm512_loadu_ps(velocity_z.add(index));

        _mm512_storeu_ps(position_x.add(index), _mm512_add_ps(px, _mm512_mul_ps(vx, dt)));
        _mm512_storeu_ps(position_y.add(index), _mm512_add_ps(py, _mm512_mul_ps(vy, dt)));
        _mm512_storeu_ps(position_z.add(index), _mm512_add_ps(pz, _mm512_mul_ps(vz, dt)));
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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
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
        let cx = _mm512_loadu_ps(center_x.add(index));
        let cy = _mm512_loadu_ps(center_y.add(index));
        let cz = _mm512_loadu_ps(center_z.add(index));
        let hx = _mm512_loadu_ps(half_x.add(index));
        let hy = _mm512_loadu_ps(half_y.add(index));
        let hz = _mm512_loadu_ps(half_z.add(index));

        _mm512_storeu_ps(min_x.add(index), _mm512_sub_ps(cx, hx));
        _mm512_storeu_ps(min_y.add(index), _mm512_sub_ps(cy, hy));
        _mm512_storeu_ps(min_z.add(index), _mm512_sub_ps(cz, hz));
        _mm512_storeu_ps(max_x.add(index), _mm512_add_ps(cx, hx));
        _mm512_storeu_ps(max_y.add(index), _mm512_add_ps(cy, hy));
        _mm512_storeu_ps(max_z.add(index), _mm512_add_ps(cz, hz));
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
