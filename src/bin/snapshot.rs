//! Regenerates the stored snapshots and annotated scans under `tests/*/expected/`.
//!
//! The fixtures name their scans after the template page they belong to: the
//! file `tests/01/data/2/1.jpg` is page 1 of the second sheet. Each scan gets a
//! `json` reading and a `png` rendering of the same reading, both placed under
//! `expected/` with the layout of the scan, so a regression shows up as a diff of
//! either file.
//!
//! ```text
//! cargo run --release --bin snapshot -- tests
//! ```

use omr::{Gray, Options, Recognition, Template};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Snapshot {
    page: usize,
    template_id: String,
    registration: RegistrationSnapshot,
    #[serde(skip_serializing_if = "Option::is_none")]
    page_number: Option<Choice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    candidate_number: Option<Candidate>,
    #[serde(skip_serializing_if = "Option::is_none")]
    barcode: Option<Barcode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    absent: Option<bool>,
    selections: Vec<Choice>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    multiple: Vec<MultipleChoice>,
    review_count: usize,
    /// Set when the sheet could not be registered at all, in which case there are
    /// no readings; the reason is kept for diagnosis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    registration_failed: Option<String>,
    /// Fingerprint of the annotation drawn over the scan.
    overlay: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct RegistrationSnapshot {
    mode: String,
    max_residual: f64,
    rotation_deg: f64,
    scale_x: f64,
    scale_y: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Choice {
    #[serde(skip_serializing_if = "Option::is_none")]
    question: Option<u32>,
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    selected: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    also_marked: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    needs_review: bool,
    confidence: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct MultipleChoice {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    question: Option<u32>,
    selected: Vec<String>,
    needs_review: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Candidate {
    value: String,
    digits: Vec<String>,
    needs_review: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Barcode {
    text: String,
    check_ok: bool,
}

/// Confidence is stored to two decimals so a rounding change does not look like
/// a regression.
fn bucket(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn snapshot(recognition: &Recognition, overlay: String) -> Snapshot {
    let registration = &recognition.registration;
    Snapshot {
        page: recognition.page,
        template_id: recognition.template_id.clone(),
        registration: RegistrationSnapshot {
            mode: format!("{:?}", registration.mode),
            max_residual: (registration.max_residual * 10.0).round() / 10.0,
            rotation_deg: (registration.rotation_deg * 100.0).round() / 100.0,
            scale_x: (registration.scale_x * 1000.0).round() / 1000.0,
            scale_y: (registration.scale_y * 1000.0).round() / 1000.0,
        },
        page_number: recognition.page_number.as_ref().map(|number| Choice {
            question: None,
            name: "PNO".to_owned(),
            selected: number.selected.clone(),
            also_marked: Vec::new(),
            reason: number.reason.map(|r| format!("{r:?}")),
            needs_review: number.needs_review,
            confidence: bucket(number.confidence),
        }),
        candidate_number: recognition
            .candidate_number
            .as_ref()
            .map(|candidate| Candidate {
                value: candidate.value.clone(),
                digits: candidate
                    .columns
                    .iter()
                    .map(|column| column.selected.clone().unwrap_or_else(|| ".".to_owned()))
                    .collect(),
                needs_review: candidate.needs_review,
            }),
        barcode: recognition.barcode.as_ref().map(|barcode| Barcode {
            text: barcode.text.clone(),
            check_ok: barcode.check_ok,
        }),
        absent: recognition.absent.as_ref().map(|absent| absent.marked),
        selections: recognition
            .selections
            .iter()
            .map(|selection| Choice {
                question: selection.question,
                name: selection.name.clone(),
                selected: selection.selected.clone(),
                also_marked: selection.also_marked.clone(),
                reason: selection.reason.map(|r| format!("{r:?}")),
                needs_review: selection.needs_review,
                confidence: bucket(selection.confidence),
            })
            .collect(),
        multiple: recognition
            .multiple
            .iter()
            .map(|reading| MultipleChoice {
                name: reading.name.clone(),
                question: reading.question,
                selected: reading.selected.clone(),
                needs_review: reading.needs_review,
            })
            .collect(),
        review_count: recognition.review_count,
        registration_failed: None,
        overlay,
    }
}

/// Snapshot for a sheet that could not be registered.
fn failed_snapshot(page: usize, reason: String) -> Snapshot {
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

/// Writes a snapshot, creating its directory.
fn write_snapshot(path: &Path, data: &Snapshot) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = serde_json::to_string_pretty(data)?;
    text.push('\n');
    std::fs::write(path, text)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The annotated preview is part of the stored result, so it is written unless
    // it is switched off.
    let mut draw = true;
    let mut root = PathBuf::from("tests/cases");
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--no-draw" => draw = false,
            other => root = PathBuf::from(other),
        }
    }

    // A suite is any directory holding a template, at the root or one level down.
    let mut suites: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(&root)?.filter_map(Result::ok) {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if find_template(&path)?.is_some() {
            suites.push(path);
            continue;
        }
        for inner in std::fs::read_dir(&path)?.filter_map(Result::ok) {
            let inner = inner.path();
            if inner.is_dir() && find_template(&inner)?.is_some() {
                suites.push(inner);
            }
        }
    }
    suites.sort();

    let mut written = 0;
    for suite_root in suites {
        let Some(template_path) = find_template(&suite_root)? else {
            continue;
        };
        let template = Template::parse_file(&template_path)?;
        let mut scans = Vec::new();
        collect_scans(&suite_root, &mut scans)?;
        scans.sort();

        for scan_path in scans {
            let Some(page) = page_of(&scan_path) else {
                eprintln!(
                    "skip {}: the file name does not name a page",
                    scan_path.display()
                );
                continue;
            };
            let scan = Gray::open(&scan_path)?;
            let recognition =
                match omr::register_scan(&template, page, &scan, None, Options::default()) {
                    Ok(recognition) => recognition,
                    Err(error) => {
                        // A sheet whose corners cannot be found is a real outcome, not
                        // a crash: record it so the test pins the refusal too.
                        let data = failed_snapshot(page, format!("{error}"));
                        write_snapshot(&expected_path(&scan_path), &data)?;
                        println!("{} -> failed: {error}", scan_path.display());
                        written += 1;
                        continue;
                    }
                };
            let rgb = omr::draw::to_rgb(scan.width(), scan.height(), scan.pixels());
            let annotated =
                omr::annotate::annotate(&template, page, &recognition, &rgb, omr::Style::default());
            let overlay = omr::annotate::overlay_digest(&annotated, &scan);
            let data = snapshot(&recognition, overlay);
            let out = expected_path(&scan_path);
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut text = serde_json::to_string_pretty(&data)?;
            text.push('\n');
            std::fs::write(&out, text)?;

            if draw {
                // Full size and lossless: the stored image is meant to be read and
                // compared against the sheet, so any scaling or lossy codec would
                // make the thing it exists to show less legible.
                let image_path = out.with_extension("png");
                omr::annotate::save_annotated(&annotated, &image_path)?;
                println!(
                    "{} -> {} + {}",
                    scan_path.display(),
                    out.display(),
                    image_path.display()
                );
            } else {
                println!("{} -> {}", scan_path.display(), out.display());
            }
            written += 1;
        }
    }

    eprintln!("wrote {written} snapshots");
    Ok(())
}

/// The single `.mb` file of a suite.
fn find_template(dir: &Path) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    let mut found = Vec::new();
    for entry in std::fs::read_dir(dir)?.filter_map(Result::ok) {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "mb") {
            found.push(path);
        }
    }
    found.sort();
    Ok(found.into_iter().next())
}

fn collect_scans(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in std::fs::read_dir(dir)?.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "expected") {
                continue;
            }
            collect_scans(&path, out)?;
        } else if path.extension().is_some_and(|e| {
            e.eq_ignore_ascii_case("jpg")
                || e.eq_ignore_ascii_case("jpeg")
                || e.eq_ignore_ascii_case("png")
        }) {
            out.push(path);
        }
    }
    Ok(())
}

/// Page index from the scan file name: `1.jpg` is page 0.
fn page_of(path: &Path) -> Option<usize> {
    let stem = path.file_stem()?.to_string_lossy();
    stem.parse::<usize>().ok().map(|n| n.saturating_sub(1))
}

/// Mirrors the layout of a scan under `expected/`, dropping a leading `data`
/// directory and the extension.
///
/// `tests/01/data/2/1.jpg` -> `tests/01/expected/2/1.json`
/// `tests/02/1/3.jpg`      -> `tests/02/expected/1/3.json`
fn expected_path(scan: &Path) -> PathBuf {
    let suite = suite_of(scan);
    let relative = scan.strip_prefix(&suite).unwrap_or(scan);
    let mut parts: Vec<String> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    // The scan file itself loses its extension.
    if let Some(last) = parts.pop() {
        let stem = Path::new(&last)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or(last);
        parts.push(stem);
    }
    if parts.first().is_some_and(|p| p == "data") {
        parts.remove(0);
    }
    let mut path = suite.join("expected");
    for part in parts {
        path.push(part);
    }
    path.set_extension("json");
    path
}

/// Directory of the suite a scan belongs to: the first ancestor holding an `.mb`.
fn suite_of(scan: &Path) -> PathBuf {
    let mut current = scan.parent();
    while let Some(dir) = current {
        let has_template = std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(Result::ok)
            .any(|entry| entry.path().extension().is_some_and(|e| e == "mb"));
        if has_template {
            return dir.to_path_buf();
        }
        current = dir.parent();
    }
    scan.parent().unwrap_or(scan).to_path_buf()
}
