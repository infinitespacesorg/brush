//! External-device render test.
//!
//! Stage 1 spike of the rosy-falcon Brush integration plan: prove that an
//! outside caller (PlayBox / Bevy / any embedder) can supply its own
//! `wgpu::Adapter` + `wgpu::Device` + `wgpu::Queue`, hand them to
//! `brush_process::burn_init_device`, and successfully render splats through
//! `brush_render::render_splats` against the externally-bound Burn backend.
//!
//! This is the load-bearing assertion of the spike. If this passes,
//! `brush-render` is usable as a library crate from outside Brush's binary
//! entry point (`burn_init_setup` is *not* called — we go around it).

#![cfg(not(target_family = "wasm"))]

use brush_process::burn_init_device;
use brush_render::{
    MainBackend, TextureMode,
    camera::Camera,
    gaussian_splats::{SplatRenderMode, Splats, render_splats},
};
use glam::{Quat, Vec3};
use wgpu::{
    DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor, MemoryHints,
    PowerPreference, RequestAdapterOptions, Trace,
};

#[tokio::test(flavor = "multi_thread")]
async fn external_wgpu_device_renders_splats() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1. Create wgpu Instance/Adapter/Device/Queue OURSELVES — i.e. exactly
    //    what an embedder (PlayBox's Bevy renderer) would do. Crucially, we
    //    do NOT call `brush_process::burn_init_setup`; that path constructs
    //    its own wgpu through `burn_wgpu::init_setup_async`. We are proving
    //    the OTHER path — externally-supplied wgpu via `burn_init_device`.
    // Match the live brush-process / brush-ui setup exactly: use the
    // "no display handle" descriptor (we're rendering offscreen).
    let instance = Instance::new(InstanceDescriptor::new_without_display_handle());

    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("no compatible wgpu adapter — is a GPU/driver available?");

    let adapter_info = adapter.get_info();
    println!(
        "external wgpu adapter: {} ({:?}, backend {:?})",
        adapter_info.name, adapter_info.device_type, adapter_info.backend,
    );

    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: Some("external-device-render-test"),
            // Mirror brush-ui's eframe path: take all features the adapter
            // offers except MAPPABLE_PRIMARY_BUFFERS (which Burn actively
            // disables).
            required_features: adapter
                .features()
                .difference(Features::MAPPABLE_PRIMARY_BUFFERS),
            required_limits: adapter.limits(),
            memory_hints: MemoryHints::MemoryUsage,
            trace: Trace::Off,
            // SAFETY: Burn's wgpu kernels need passthrough shaders, just
            // like the live Brush UI path enables them.
            experimental_features: unsafe { ExperimentalFeatures::enabled() },
        })
        .await
        .expect("request_device failed on externally-created adapter");

    // 2. Bind Burn to the externally-created device. This is the spike's
    //    critical call — it proves Brush's renderer can plug into a wgpu
    //    that the embedder owns.
    let burn_device = burn_init_device(adapter, device, queue);

    // 3. Build a small synthetic splat scene. Six splats placed near the
    //    origin so a camera at z=-5 (looking down +Z, which is the
    //    convention proven out by `brush-render::tests::renders_many_splats`)
    //    sees them. Bright colors and high opacity so at least some pixels
    //    MUST come back non-transparent.
    let positions: Vec<f32> = vec![
        // x, y, z — six splats arranged in a tight cluster near origin
        -0.3, -0.3, 0.0, // bottom-left
        0.3, -0.3, 0.0, // bottom-right
        -0.3, 0.3, 0.0, // top-left
        0.3, 0.3, 0.0, // top-right
        0.0, 0.0, 0.5, // front-center (closer to camera)
        0.0, 0.0, -0.5, // back-center (farther from camera)
    ];
    let num_splats = positions.len() / 3;

    // Identity rotations.
    let rotations: Vec<f32> = (0..num_splats).flat_map(|_| [1.0, 0.0, 0.0, 0.0]).collect();

    // log_scale = ln(0.15) ≈ -1.9 — splats roughly 0.15 world units across,
    // big enough to span multiple pixels at 256x256 / FOV 0.5.
    let log_scales: Vec<f32> = (0..num_splats).flat_map(|_| [-1.9_f32, -1.9, -1.9]).collect();

    // SH degree 0 → one [r,g,b] DC coefficient per splat, in SH-DC space
    // (~0.5 DC ≈ mid-gray after sigmoid+SH evaluation; well above zero).
    let sh_coeffs: Vec<f32> = (0..num_splats).flat_map(|_| [0.5_f32, 0.5, 0.5]).collect();

    // raw_opacity = 4.0 → sigmoid(4.0) ≈ 0.98 — strongly visible.
    let raw_opacities: Vec<f32> = vec![4.0; num_splats];

    let splats = Splats::<MainBackend>::from_raw(
        positions,
        rotations,
        log_scales,
        sh_coeffs,
        raw_opacities,
        SplatRenderMode::Default,
        &burn_device,
    );
    assert_eq!(splats.num_splats() as usize, num_splats);

    // 4. Render to a 256x256 offscreen target. Camera placement matches the
    //    in-tree positive-render tests (e.g. `renders_many_splats`):
    //    z=-5, identity rotation, looking down +Z toward the origin cluster.
    let camera = Camera::new(
        Vec3::new(0.0, 0.0, -5.0),
        Quat::IDENTITY,
        0.5,
        0.5,
        glam::vec2(0.5, 0.5),
    );
    let img_size = glam::uvec2(256, 256);

    let (output, aux) = render_splats(
        splats,
        &camera,
        img_size,
        Vec3::ZERO, // black background — non-zero alpha must come from splats
        None,
        TextureMode::Float,
    )
    .await;

    assert!(
        aux.num_visible > 0,
        "render produced num_visible=0 against externally-supplied wgpu device — \
         likely means burn_init_device did not bind the backend correctly"
    );

    // 5. Read back the rendered tensor and assert at least one pixel has
    //    nonzero alpha. This is the "did anything render" check.
    let pixels = output
        .to_data_async()
        .await
        .expect("failed to read back rendered tensor")
        .to_vec::<f32>()
        .expect("rendered tensor was not f32");

    let total_pixels = (img_size.x * img_size.y) as usize;
    assert_eq!(
        pixels.len(),
        total_pixels * 4,
        "expected RGBA, got {} floats for {}x{}",
        pixels.len(),
        img_size.x,
        img_size.y,
    );

    // Sanity: no NaNs/infs.
    assert!(
        pixels.iter().all(|v| v.is_finite()),
        "rendered tensor contained NaN/inf — backend binding likely garbage"
    );

    // Count nonzero-alpha pixels; report for debugging.
    let alpha_pixels: Vec<f32> = pixels.chunks_exact(4).map(|c| c[3]).collect();
    let nonzero_alpha_count = alpha_pixels.iter().filter(|a| **a > 1e-3).count();
    let max_alpha = alpha_pixels.iter().fold(0.0_f32, |m, a| m.max(*a));

    println!(
        "external-device render: {} of {} pixels had alpha > 1e-3 (max alpha {:.4}, num_visible {})",
        nonzero_alpha_count, total_pixels, max_alpha, aux.num_visible,
    );

    assert!(
        nonzero_alpha_count > 0,
        "no pixel had alpha > 1e-3 — splats did not render through the \
         externally-created wgpu device (max alpha was {max_alpha})"
    );
}
