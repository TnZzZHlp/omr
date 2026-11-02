//! Regression tests over the stored snapshots.
//!
//! Every scan under `tests/` has a companion JSON file with the reading that was
//! produced when the snapshot was taken. The tests recompute the reading and
//! compare it field by field, so any change in registration, cell measurement or
//! decision surfaces as a diff instead of as a silently different answer.
//!
//! Regenerate the snapshots with:
//!
//! ```text
//! cargo run --release --bin snapshot
//! ```

mod common;

use common::{RegistrationSnapshot, Snapshot};
use omr::{Gray, Options, Recognition, Template};
use std::path::{Path, PathBuf};

fn manifest() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn cases_root() -> PathBuf {
    manifest().join("tests").join("cases")
}

/// Page index from the scan file name: `1.jpg` is page 0.
fn page_of(path: &Path) -> usize {
    path.file_stem()
        .and_then(|stem| stem.to_string_lossy().parse::<usize>().ok())
        .map_or(0, |n| n.saturating_sub(1))
}

/// The single template of a suite.
fn template_of(suite: &Path) -> PathBuf {
    let mut found: Vec<PathBuf> = std::fs::read_dir(suite)
        .expect("suite reads")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == "mb"))
        .collect();
    found.sort();
    found.into_iter().next().expect("suite has a template")
}

/// Snapshot path of a scan, mirroring the layout used by the generator.
fn expected_of(scan: &Path, suite: &Path) -> PathBuf {
    let relative = scan.strip_prefix(suite).expect("scan is inside its suite");
    let mut path = suite.join("expected");
    for component in relative.components() {
        let name = component.as_os_str().to_string_lossy().into_owned();
        if name == "data" {
            continue;
        }
        path.push(name);
    }
    path.set_extension("json");
    path
}

/// A reading, or the reason the sheet could not be read.
enum Read {
    Ok(Box<(Template, Gray, Recognition)>),
    Failed(String),
}

fn recognise(scan: &Path, template_path: &Path) -> Read {
    let template = Template::parse_file(template_path).expect("template parses");
    let image = Gray::open(scan).expect("scan loads");
    match omr::register_scan(&template, page_of(scan), &image, None, Options::default()) {
        Ok(recognition) => Read::Ok(Box::new((template, image, recognition))),
        Err(error) => Read::Failed(format!("{error}")),
    }
}

/// Snapshot of a sheet that could not be registered, matching the generator.
fn failed(page: usize, reason: String) -> Snapshot {
    Snapshot {
        page,
        template_id: String::new(),
        registration: RegistrationSnapshot {
            mode: "failed".to_owned(),
            max_residual: -1.0,
            rotation_deg: 0.0,
            scale_x: 0.0,
            scale_y: 0.0,
        },
        page_number: None,
        candidate_number: None,
        barcode: None,
        absent: None,
        selections: Vec::new(),
        multiple: Vec::new(),
        review_count: 0,
        registration_failed: Some(reason),
        overlay: "none".to_owned(),
    }
}

/// Compares one scan against its stored snapshot.
fn check(scan: &Path, template_path: &Path, suite: &Path) {
    let expected_path = expected_of(scan, suite);
    let expected = Snapshot::load(&expected_path)
        .unwrap_or_else(|e| panic!("{}: {e}", expected_path.display()));
    let actual = match recognise(scan, template_path) {
        Read::Failed(reason) => failed(page_of(scan), reason),
        Read::Ok(parts) => {
            let (template, image, recognition) = *parts;
            let rgb = omr::draw::to_rgb(image.width(), image.height(), image.pixels());
            let annotated = omr::annotate::annotate(
                &template,
                page_of(scan),
                &recognition,
                &rgb,
                omr::Style::default(),
            );
            let overlay = omr::annotate::overlay_digest(&annotated, &image);
            Snapshot::from_parts(&recognition, overlay)
        }
    };

    assert_eq!(
        actual,
        expected,
        "\n{} no longer matches {}\nrun `cargo run --release --bin snapshot` after checking the difference\n",
        scan.display(),
        expected_path.display()
    );
}

/// Every suite on disk, with its template and its scans.
fn suites() -> Vec<(PathBuf, PathBuf, Vec<PathBuf>)> {
    let mut out = Vec::new();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(cases_root())
        .expect("cases directory reads")
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();

    for suite in dirs {
        if !suite
            .read_dir()
            .expect("suite reads")
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|e| e == "mb"))
        {
            continue;
        }
        let template = template_of(&suite);
        let mut scans = Vec::new();
        collect_scans(&suite, &mut scans);
        scans.sort();
        out.push((suite, template, scans));
    }
    out
}

fn collect_scans(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir)
        .expect("directory reads")
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "expected") {
                continue;
            }
            collect_scans(&path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("jpg") || e.eq_ignore_ascii_case("jpeg"))
        {
            out.push(path);
        }
    }
}

#[test]
fn every_scan_matches_its_snapshot() {
    let mut checked = 0;
    for (suite, template, scans) in suites() {
        assert!(!scans.is_empty(), "{} has no scans", suite.display());
        for scan in scans {
            check(&scan, &template, &suite);
            checked += 1;
        }
    }
    assert!(checked >= 30, "expected the full corpus, found {checked}");
}

/// A missing snapshot is a test failure, not a silent pass.
#[test]
fn every_scan_has_a_snapshot() {
    for (suite, _, scans) in suites() {
        for scan in scans {
            let path = expected_of(&scan, &suite);
            assert!(
                path.is_file(),
                "{} has no snapshot; run `cargo run --release --bin snapshot`",
                scan.display()
            );
        }
    }
}

/// Every scan also has a stored preview of its annotation, so a reviewer can look
/// at what the reader saw without regenerating anything.
#[test]
fn every_scan_has_a_preview() {
    for (suite, _, scans) in suites() {
        for scan in scans {
            let snapshot = expected_of(&scan, &suite);
            // A sheet that could not be registered has no annotation to preview.
            let stored = Snapshot::load(&snapshot).expect("snapshot reads");
            if stored.registration_failed.is_some() {
                continue;
            }
            let preview = snapshot.with_extension("png");
            assert!(
                preview.is_file(),
                "{} has no preview; run `cargo run --release --bin snapshot`",
                scan.display()
            );
        }
    }
}

/// The stored annotation must keep the scan's own resolution: a downscaled or
/// lossy copy cannot be compared against the sheet it came from.
#[test]
fn previews_keep_the_scan_resolution() {
    for (suite, template, scans) in suites() {
        for scan in scans {
            let stored = Snapshot::load(&expected_of(&scan, &suite)).expect("snapshot reads");
            if stored.registration_failed.is_some() {
                continue;
            }
            let source = image::open(&scan).expect("scan opens");
            let preview = image::open(expected_of(&scan, &suite).with_extension("png"))
                .expect("preview opens");
            assert_eq!(
                (preview.width(), preview.height()),
                (source.width(), source.height()),
                "{} is not stored at the scan resolution",
                scan.display()
            );
            let _ = &template;
        }
    }
}
