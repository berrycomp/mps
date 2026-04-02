const LANES: usize = 4;

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
#[target_feature(enable = "altivec")]
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
        *dst_px.add(index) = *src_px.add(index);
        *dst_px.add(index + 1) = *src_px.add(index + 1);
        *dst_px.add(index + 2) = *src_px.add(index + 2);
        *dst_px.add(index + 3) = *src_px.add(index + 3);

        *dst_py.add(index) = *src_py.add(index);
        *dst_py.add(index + 1) = *src_py.add(index + 1);
        *dst_py.add(index + 2) = *src_py.add(index + 2);
        *dst_py.add(index + 3) = *src_py.add(index + 3);

        *dst_pz.add(index) = *src_pz.add(index);
        *dst_pz.add(index + 1) = *src_pz.add(index + 1);
        *dst_pz.add(index + 2) = *src_pz.add(index + 2);
        *dst_pz.add(index + 3) = *src_pz.add(index + 3);

        *dst_rx.add(index) = *src_rx.add(index);
        *dst_rx.add(index + 1) = *src_rx.add(index + 1);
        *dst_rx.add(index + 2) = *src_rx.add(index + 2);
        *dst_rx.add(index + 3) = *src_rx.add(index + 3);

        *dst_ry.add(index) = *src_ry.add(index);
        *dst_ry.add(index + 1) = *src_ry.add(index + 1);
        *dst_ry.add(index + 2) = *src_ry.add(index + 2);
        *dst_ry.add(index + 3) = *src_ry.add(index + 3);

        *dst_rz.add(index) = *src_rz.add(index);
        *dst_rz.add(index + 1) = *src_rz.add(index + 1);
        *dst_rz.add(index + 2) = *src_rz.add(index + 2);
        *dst_rz.add(index + 3) = *src_rz.add(index + 3);

        *dst_rw.add(index) = *src_rw.add(index);
        *dst_rw.add(index + 1) = *src_rw.add(index + 1);
        *dst_rw.add(index + 2) = *src_rw.add(index + 2);
        *dst_rw.add(index + 3) = *src_rw.add(index + 3);
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

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
#[target_feature(enable = "altivec")]
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
    let mut index = 0;
    while index + LANES <= len {
        *position_x.add(index) += *velocity_x.add(index) * dt_seconds;
        *position_x.add(index + 1) += *velocity_x.add(index + 1) * dt_seconds;
        *position_x.add(index + 2) += *velocity_x.add(index + 2) * dt_seconds;
        *position_x.add(index + 3) += *velocity_x.add(index + 3) * dt_seconds;

        *position_y.add(index) += *velocity_y.add(index) * dt_seconds;
        *position_y.add(index + 1) += *velocity_y.add(index + 1) * dt_seconds;
        *position_y.add(index + 2) += *velocity_y.add(index + 2) * dt_seconds;
        *position_y.add(index + 3) += *velocity_y.add(index + 3) * dt_seconds;

        *position_z.add(index) += *velocity_z.add(index) * dt_seconds;
        *position_z.add(index + 1) += *velocity_z.add(index + 1) * dt_seconds;
        *position_z.add(index + 2) += *velocity_z.add(index + 2) * dt_seconds;
        *position_z.add(index + 3) += *velocity_z.add(index + 3) * dt_seconds;
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

#[cfg(any(target_arch = "powerpc", target_arch = "powerpc64"))]
#[target_feature(enable = "altivec")]
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
        for lane in 0..LANES {
            let slot = index + lane;
            let cx = *center_x.add(slot);
            let cy = *center_y.add(slot);
            let cz = *center_z.add(slot);
            let hx = *half_x.add(slot);
            let hy = *half_y.add(slot);
            let hz = *half_z.add(slot);

            *min_x.add(slot) = cx - hx;
            *min_y.add(slot) = cy - hy;
            *min_z.add(slot) = cz - hz;
            *max_x.add(slot) = cx + hx;
            *max_y.add(slot) = cy + hy;
            *max_z.add(slot) = cz + hz;
        }
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
