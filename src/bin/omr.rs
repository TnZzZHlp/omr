//! Reads a scanned answer sheet and writes an annotated copy of the scan.
//!
//! ```text
//! omr <template.mb> <scan.jpg> [-p page] [-o out.png] [--crop x0,y0,x1,y1]
//!     [--scale n] [--no-cells] [--no-registration] [--json]
//! ```

use omr::{annotate, draw, recognize, Gray, Options, Rect, Style, Template};
use std::path::PathBuf;
use std::process::ExitCode;

const USAGE: &str = "\
usage: omr <template.mb> <scan.jpg> [options]

options:
  -p, --page <n>        zero based template page index (default 0)
  -o, --out <path>      output image, defaults to <scan>.annotated.png
      --crop <rect>     x0,y0,x1,y1 in scan pixels, for a spread of two pages
      --scale <n>       label and line scale (default 2)
      --no-cells        do not outline every candidate cell
      --no-registration do not draw the fitted page corners
      --json            print the readings as JSON on stdout
  -h, --help            show this message
";

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("omr: {error}");
            ExitCode::FAILURE
        }
    }
}

struct Args {
    template: PathBuf,
    scan: PathBuf,
    page: usize,
    out: Option<PathBuf>,
    crop: Option<Rect>,
    style: Style,
    json: bool,
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let Some(args) = parse_args()? else {
        print!("{USAGE}");
        return Ok(());
    };

    let template = Template::parse_file(&args.template)?;
    let scan = Gray::open(&args.scan)?;
    let recognition =
        recognize::register_scan(&template, args.page, &scan, args.crop, Options::default())?;

    let rgb = draw::to_rgb(scan.width(), scan.height(), scan.pixels());
    let annotated = annotate::annotate(&template, args.page, &recognition, &rgb, args.style);

    let out = args
        .out
        .unwrap_or_else(|| default_output(args.scan.as_path()));
    annotated
        .save(&out)
        .map_err(|e| format!("cannot write {}: {e}", out.display()))?;

    eprintln!(
        "registered {:.2} px residual at {:.2} deg, {} of {} questions need review",
        recognition.registration.max_residual,
        recognition.registration.rotation_deg,
        recognition.review_count,
        recognition.selections.len()
    );
    for selection in &recognition.selections {
        eprintln!(
            "  Q{:>2} {} {:.2}{}",
            selection.question.unwrap_or(0),
            selection.selected.clone().unwrap_or_else(|| "-".to_owned()),
            selection.confidence,
            selection
                .reason
                .map(|r| format!(" ({r:?})"))
                .unwrap_or_default()
        );
    }
    if let Some(number) = &recognition.page_number {
        eprintln!(
            "  page number {}",
            number.selected.clone().unwrap_or_else(|| "?".to_owned())
        );
    }
    if let Some(absent) = &recognition.absent {
        eprintln!(
            "  absent mark {}",
            if absent.marked { "set" } else { "clear" }
        );
    }
    if let Some(barcode) = &recognition.barcode {
        eprintln!(
            "  barcode {} (check {})",
            barcode.text,
            if barcode.check_ok { "ok" } else { "failed" }
        );
    }
    eprintln!("wrote {}", out.display());

    if args.json {
        println!("{}", serde_json::to_string_pretty(&recognition)?);
    }
    Ok(())
}

fn default_output(scan: &std::path::Path) -> PathBuf {
    let stem = scan
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "scan".to_owned());
    scan.with_file_name(format!("{stem}.annotated.png"))
}

fn parse_args() -> Result<Option<Args>, Box<dyn std::error::Error>> {
    let mut positional: Vec<String> = Vec::new();
    let mut page = 0usize;
    let mut out = None;
    let mut crop = None;
    let mut style = Style::default();
    let mut json = false;

    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "-h" | "--help" => return Ok(None),
            "-p" | "--page" => {
                page = args
                    .next()
                    .ok_or("--page needs a number")?
                    .parse()
                    .map_err(|_| "--page needs a number")?;
            }
            "-o" | "--out" => out = args.next().map(PathBuf::from),
            "--crop" => {
                let raw = args.next().ok_or("--crop needs x0,y0,x1,y1")?;
                let parts: Vec<f64> = raw
                    .split(',')
                    .filter_map(|p| p.trim().parse().ok())
                    .collect();
                if parts.len() != 4 {
                    return Err("--crop needs x0,y0,x1,y1".into());
                }
                crop = Some(Rect::new(parts[0], parts[1], parts[2], parts[3]));
            }
            "--scale" => {
                style.scale = args
                    .next()
                    .ok_or("--scale needs a number")?
                    .parse()
                    .map_err(|_| "--scale needs a number")?;
            }
            "--no-cells" => style.show_all_cells = false,
            "--no-registration" => style.show_registration = false,
            "--json" => json = true,
            other if other.starts_with('-') => {
                return Err(format!("unknown option {other}").into());
            }
            other => positional.push(other.to_owned()),
        }
    }

    if positional.len() < 2 {
        return Err("expected a template and a scan".into());
    }
    Ok(Some(Args {
        template: PathBuf::from(&positional[0]),
        scan: PathBuf::from(&positional[1]),
        page,
        out,
        crop,
        style,
        json,
    }))
}
