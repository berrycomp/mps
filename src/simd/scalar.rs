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
    for index in 0..len {
        *dst_px.add(index) = *src_px.add(index);
        *dst_py.add(index) = *src_py.add(index);
        *dst_pz.add(index) = *src_pz.add(index);
        *dst_rx.add(index) = *src_rx.add(index);
        *dst_ry.add(index) = *src_ry.add(index);
        *dst_rz.add(index) = *src_rz.add(index);
        *dst_rw.add(index) = *src_rw.add(index);
    }
}

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
    for index in 0..len {
        *position_x.add(index) += *velocity_x.add(index) * dt_seconds;
        *position_y.add(index) += *velocity_y.add(index) * dt_seconds;
        *position_z.add(index) += *velocity_z.add(index) * dt_seconds;
    }
}

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
    for index in 0..len {
        let cx = *center_x.add(index);
        let cy = *center_y.add(index);
        let cz = *center_z.add(index);
        let hx = *half_x.add(index);
        let hy = *half_y.add(index);
        let hz = *half_z.add(index);

        *min_x.add(index) = cx - hx;
        *min_y.add(index) = cy - hy;
        *min_z.add(index) = cz - hz;
        *max_x.add(index) = cx + hx;
        *max_y.add(index) = cy + hy;
        *max_z.add(index) = cz + hz;
    }
}

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
    for index in 0..len {
        let overlaps = *a_min_x.add(index) <= *b_max_x.add(index)
            && *a_max_x.add(index) >= *b_min_x.add(index)
            && *a_min_y.add(index) <= *b_max_y.add(index)
            && *a_max_y.add(index) >= *b_min_y.add(index)
            && *a_min_z.add(index) <= *b_max_z.add(index)
            && *a_max_z.add(index) >= *b_min_z.add(index);
        *out_mask.add(index) = u32::from(overlaps);
    }
}
