//! External-driven training smoke test.
//!
//! Stage 1 spike of the rosy-falcon Brush integration plan: prove that
//! `brush-train` (via `brush_process::create_process`) can be driven by an
//! external caller — load a Nerfstudio dataset, run a minimal training
//! schedule, and emit a PLY to disk — without going through Brush's `brush`
//! binary entry point or the egui/eframe UI.
//!
//! This mirrors the FFI embedding pattern in `brush-app/src/ffi.rs`
//! (`train_and_save`), but stays inside Rust — no FFI boundary, no C
//! callback shim. If this passes, `brush-train` is usable as a library crate
//! from outside Brush's binary entry point.
//!
//! The test reuses the existing test fixture at
//! `brush-app/tests/data/test_dataset/` (a tiny Nerfstudio dataset: a single
//! 50x50 view + an init.ply). Copying the fixture into a `tempfile::tempdir`
//! keeps the test self-contained — no writes to the source tree.

#![cfg(not(target_family = "wasm"))]

use std::path::{Path, PathBuf};

use brush_process::config::TrainStreamConfig;
use brush_process::message::{ProcessMessage, TrainMessage};
use brush_process::{burn_init_setup, create_process};
use brush_vfs::DataSource;
use tokio_stream::StreamExt;

/// Recursively copy a directory tree. The fixture is small (one PNG, one
/// init.ply, one transforms.json), so a naive recursion is fine.
fn copy_dir_recursive(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&from, &to)?;
        } else {
            std::fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread")]
async fn external_caller_drives_training_to_ply() {
    let _ = env_logger::builder().is_test(true).try_init();

    // 1. Stage the existing Nerfstudio test fixture into a tempdir so the
    //    test is hermetic (no writes to the source tree, no leftover files
    //    on failure).
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let fixture_src = Path::new(manifest_dir)
        .join("..")
        .join("brush-app")
        .join("tests")
        .join("data")
        .join("test_dataset");
    assert!(
        fixture_src.exists(),
        "test fixture missing at {}",
        fixture_src.display()
    );

    let tempdir = tempfile::Builder::new()
        .prefix("brush_external_train_")
        .tempdir()
        .expect("could not create tempdir");
    let dataset_dir = tempdir.path().join("dataset");
    copy_dir_recursive(&fixture_src, &dataset_dir).expect("failed to stage test fixture");

    // 2. Configure a minimum-iteration training run. Numbers mirror the
    //    in-fork FFI test (`brush-app/tests/integration.rs::test_train_and_save_ffi_short`),
    //    which is already proven to drive end-to-end training to a PLY in
    //    CI: total_train_iters=10, refine_every=5, export_every=10,
    //    max_resolution=50.
    //
    //    `export_path = "./output/"` is resolved by train_stream.rs relative
    //    to the dataset's PARENT directory, so the PLY lands at:
    //        <tempdir>/output/export_<iter>.ply
    let output_dir = tempdir.path().join("output");
    let mut config = TrainStreamConfig::default();
    config.train_config.total_train_iters = 10;
    config.train_config.refine_every = 5;
    config.process_config.export_every = 10;
    config.process_config.export_path = "./output/".to_owned();
    config.process_config.export_name = "export_{iter}.ply".to_owned();
    config.process_config.eval_save_to_disk = false;
    config.load_config.max_resolution = 50;

    // 3. Bind Burn to its default wgpu setup. Unlike the external_device
    //    sibling test, the smoke path uses `burn_init_setup` (Brush's own
    //    setup) — the spike's claim here is that `brush-train` can be DRIVEN
    //    by an external caller, not that it can render against an external
    //    device. The latter is covered separately.
    let _device = burn_init_setup().await;

    let dataset_path_str = dataset_dir
        .to_str()
        .expect("dataset path was non-UTF8")
        .to_owned();
    let source = DataSource::Path(dataset_path_str.clone());

    let mut process = create_process(source, async move |_| config);

    // 4. Drive the ProcessStream to completion. Track which TrainMessages
    //    fire so we can fail informatively if training stalls or completes
    //    abnormally.
    let mut saw_train_config = false;
    let mut saw_dataset = false;
    let mut saw_done_training = false;
    let mut last_iter: u32 = 0;
    let mut warnings: Vec<String> = Vec::new();

    while let Some(msg_result) = process.stream.next().await {
        let msg = msg_result.expect("ProcessStream yielded an error during external training run");
        match msg {
            ProcessMessage::NewProcess => {
                println!("smoke: NewProcess");
            }
            ProcessMessage::StartLoading {
                name,
                training,
                base_path,
                ..
            } => {
                println!(
                    "smoke: StartLoading name={name:?} training={training} base_path={base_path:?}"
                );
                assert!(training, "test fixture must be loaded as training, not single-PLY");
            }
            ProcessMessage::SplatsUpdated {
                num_splats, frame, ..
            } => {
                println!("smoke: SplatsUpdated frame={frame} num_splats={num_splats}");
            }
            ProcessMessage::TrainMessage(TrainMessage::TrainConfig { .. }) => {
                saw_train_config = true;
                println!("smoke: TrainConfig emitted");
            }
            ProcessMessage::TrainMessage(TrainMessage::Dataset { .. }) => {
                saw_dataset = true;
                println!("smoke: Dataset loaded");
            }
            ProcessMessage::TrainMessage(TrainMessage::TrainStep { iter, .. }) => {
                last_iter = iter;
                if iter <= 3 || iter == 10 || iter % 5 == 0 {
                    println!("smoke: TrainStep iter={iter}");
                }
            }
            ProcessMessage::TrainMessage(TrainMessage::RefineStep {
                cur_splat_count,
                iter,
            }) => {
                println!("smoke: RefineStep iter={iter} cur_splat_count={cur_splat_count}");
            }
            ProcessMessage::TrainMessage(TrainMessage::EvalResult {
                iter,
                avg_psnr,
                avg_ssim,
            }) => {
                println!("smoke: EvalResult iter={iter} psnr={avg_psnr:.3} ssim={avg_ssim:.3}");
            }
            ProcessMessage::TrainMessage(TrainMessage::DoneTraining) => {
                saw_done_training = true;
                println!("smoke: DoneTraining");
            }
            ProcessMessage::Warning { error } => {
                let msg = error.to_string();
                eprintln!("smoke: Warning: {msg}");
                warnings.push(msg);
            }
            ProcessMessage::DoneLoading => {
                println!("smoke: DoneLoading");
            }
        }
    }

    assert!(saw_train_config, "training never emitted TrainConfig");
    assert!(saw_dataset, "training never emitted Dataset");
    assert!(
        saw_done_training,
        "training never emitted DoneTraining (last iter seen: {last_iter}, warnings: {warnings:?})"
    );
    assert!(
        last_iter >= 1,
        "training emitted no TrainStep messages — loop never ran"
    );

    // 5. Assert the export PLY exists on disk after completion.
    let exported_files = list_files_recursive(&output_dir).unwrap_or_else(|e| {
        panic!(
            "expected output dir {} to exist after training but couldn't read it: {e}",
            output_dir.display()
        )
    });
    let ply_files: Vec<&PathBuf> = exported_files
        .iter()
        .filter(|p| p.extension().is_some_and(|ext| ext == "ply"))
        .collect();

    assert!(
        !ply_files.is_empty(),
        "training completed but emitted no .ply files in {} (saw {} files: {:?})",
        output_dir.display(),
        exported_files.len(),
        exported_files,
    );

    let ply = ply_files[0];
    let metadata = std::fs::metadata(ply).expect("could not stat exported PLY");
    println!(
        "smoke: exported PLY {} ({} bytes)",
        ply.display(),
        metadata.len()
    );
    assert!(
        metadata.len() > 100,
        "exported PLY at {} is suspiciously small ({} bytes)",
        ply.display(),
        metadata.len(),
    );
}

fn list_files_recursive(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            out.extend(list_files_recursive(&path)?);
        } else {
            out.push(path);
        }
    }
    Ok(out)
}
