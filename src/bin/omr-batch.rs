//! Recognises every scan in a directory against one template, in parallel.
//!
//! ```text
//! omr-batch <template.mb> <directory> [-p page] [-o out-dir] [--jobs n]
//!     [--json] [--scale n] [--no-cells] [--no-registration] [--fail-fast]
//! ```

use clap::Parser;
use omr::{annotate, draw, recognize, Gray, Options, Recognition, Style, Template};
use rayon::prelude::*;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Recognise every scan in a directory against one template.
#[derive(Parser)]
#[command(name = "omr-batch", version, about)]
struct Args {
    /// Template file (.mb).
    template: PathBuf,
    /// Directory scanned recursively for .jpg/.jpeg/.png images.
    directory: PathBuf,
    /// Zero-based template page index.
    #[arg(short, long, default_value_t = 0)]
    page: usize,
    /// Output directory for annotated images (mirrors the input layout).
    #[arg(short, long)]
    out: Option<PathBuf>,
    /// Worker threads (default: all cores).
    #[arg(short, long)]
    jobs: Option<usize>,
    /// Label and line scale.
    #[arg(long, default_value_t = 2)]
    scale: u32,
    /// Do not outline every candidate cell.
    #[arg(long)]
    no_cells: bool,
    /// Do not draw the fitted page corners.
    #[arg(long)]
    no_registration: bool,
    /// Print one JSON object per scan on stdout.
    #[arg(long)]
    json: bool,
    /// Stop scheduling new scans after the first failure.
    #[arg(long)]
    fail_fast: bool,
}

/// What one scan produced: a reading, or the reason it has none.
struct Outcome {
    scan: PathBuf,
    image: Option<PathBuf>,
    reading: Option<Recognition>,
    reason: Option<String>,
}

impl Outcome {
    fn ok(&self) -> bool {
        self.reason.is_none()
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match run(args) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::FAILURE,
        Err(error) => {
            eprintln!("omr-batch: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<usize, Box<dyn std::error::Error>> {
    if !args.template.is_file() {
        return Err(format!("template not found: {}", args.template.display()).into());
    }
    if !args.directory.is_dir() {
        return Err(format!("not a directory: {}", args.directory.display()).into());
    }
    if let Some(jobs) = args.jobs {
        rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build_global()
            .map_err(|e| e.to_string())?;
    }

    let template = Template::parse_file(&args.template)?;
    let mut scans = Vec::new();
    collect(&args.directory, &mut scans);
    scans.sort();
    if scans.is_empty() {
        return Err(format!("no images under {}", args.directory.display()).into());
    }

    let style = Style {
        scale: args.scale,
        show_all_cells: !args.no_cells,
        show_registration: !args.no_registration,
    };
    let stop = AtomicUsize::new(0);

    let outcomes: Vec<Outcome> = scans
        .par_iter()
        .map(|scan| {
            if stop.load(Ordering::Relaxed) == 1 {
                return Outcome {
                    scan: scan.clone(),
                    image: None,
                    reading: None,
                    reason: None,
                };
            }
            let outcome = process(&args, &template, &style, scan);
            if !outcome.ok() && args.fail_fast {
                stop.store(1, Ordering::Relaxed);
            }
            outcome
        })
        .collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for outcome in &outcomes {
        if args.json {
            writeln!(out, "{}", json_line(outcome))?;
        } else {
            writeln!(out, "{}", describe(outcome))?;
        }
    }

    let failed = outcomes.iter().filter(|o| !o.ok()).count();
    eprintln!(
        "{} scans: {} read, {} failed",
        scans.len(),
        scans.len() - failed,
        failed
    );
    Ok(failed)
}

fn process(args: &Args, template: &Template, style: &Style, scan: &Path) -> Outcome {
    let outcome = match Gray::open(scan) {
        Ok(image) => match recognize::register_scan(template, args.page, &image, None, Options::default()) {
            Ok(recognition) => {
                let image_path = if let Some(root) = &args.out {
                    let path = output_path(root, &args.directory, scan);
                    let rgb = draw::to_rgb(image.width(), image.height(), image.pixels());
                    let annotated = annotate::annotate(template, args.page, &recognition, &rgb, *style);
                    match annotated.save(&path) {
                        Ok(()) => Some(path),
                        Err(e) => {
                            return Outcome {
                                scan: scan.to_path_buf(),
                                image: None,
                                reading: None,
                                reason: Some(format!("cannot write {}: {e}", path.display())),
                            };
                        }
                    }
                } else {
                    None
                };
                Outcome {
                    scan: scan.to_path_buf(),
                    image: image_path,
                    reading: Some(recognition),
                    reason: None,
                }
            }
            Err(e) => Outcome {
                scan: scan.to_path_buf(),
                image: None,
                reading: None,
                reason: Some(e.to_string()),
            },
        },
        Err(e) => Outcome {
            scan: scan.to_path_buf(),
            image: None,
            reading: None,
            reason: Some(e.to_string()),
        },
    };
    outcome
}

/// `<out>/<relative path>.annotated.png`, mirroring the layout under the input
/// directory so nested corpora cannot collide.
fn output_path(root: &Path, base: &Path, scan: &Path) -> PathBuf {
    let stem = scan
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "scan".to_owned());
    let relative = scan.strip_prefix(base).unwrap_or(Path::new(&stem));
    let target = root.join(relative);
    if let Some(parent) = target.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    target.with_file_name(format!("{stem}.annotated.png"))
}

fn describe(outcome: &Outcome) -> String {
    let path = outcome.scan.display();
    match (&outcome.reading, &outcome.reason) {
        (Some(reading), _) => {
            let answers: String = reading
                .selections
                .iter()
                .map(|s| s.selected.clone().unwrap_or_else(|| "?".to_owned()))
                .collect();
            let barcode = reading
                .barcode
                .as_ref()
                .map(|b| b.text.as_str())
                .unwrap_or("-");
            let image = outcome
                .image
                .as_ref()
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "-".to_owned());
            format!(
                "{path}: {} {answers} barcode {barcode} review {} -> {image}",
                reading.registration.mode.as_str(),
                reading.review_count
            )
        }
        (None, Some(reason)) => format!("{path}: {reason}"),
        (None, None) => format!("{path}: skipped"),
    }
}

fn json_line(outcome: &Outcome) -> String {
    let scan = outcome.scan.display().to_string();
    match (&outcome.reading, &outcome.reason) {
        (Some(reading), _) => format!(
            "{}",
            serde_json::json!({"scan": scan, "result": reading})
        ),
        (None, Some(reason)) => serde_json::json!({"scan": scan, "error": reason}).to_string(),
        (None, None) => serde_json::json!({"scan": scan, "error": "skipped"}).to_string(),
    }
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out);
        } else if is_scan(&path) {
            out.push(path);
        }
    }
}

fn is_scan(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if name.starts_with('.') || name.ends_with(".annotated.png") {
        return false;
    }
    path.extension().is_some_and(|e| {
        matches!(
            e.to_ascii_lowercase().to_str(),
            Some("jpg") | Some("jpeg") | Some("png")
        )
    })
}
