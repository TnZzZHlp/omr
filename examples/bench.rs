//! Timing harness for the test corpus.
//!
//! ```text
//! cargo run --release --example bench [rounds]
//! ```

use omr::{Gray, Options, Template};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = PathBuf::from("tests/cases");
    let rounds: usize = std::env::args()
        .nth(1)
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let mut jobs = Vec::new();
    for suite in std::fs::read_dir(&root)?
        .filter_map(Result::ok)
        .map(|e| e.path())
    {
        if !suite.is_dir() {
            continue;
        }
        let Some(template_path) = find(&suite, "mb") else {
            continue;
        };
        let mut scans = Vec::new();
        collect(&suite, &mut scans);
        scans.sort();
        for scan in scans {
            jobs.push((template_path.clone(), scan));
        }
    }
    jobs.sort_by(|a, b| a.1.cmp(&b.1));
    println!("{} scans, {rounds} rounds\n", jobs.len());

    let mut total_parse = Duration::ZERO;
    let mut total_bytes = 0usize;
    let mut total_pixels = 0usize;
    let mut per_scan: Vec<(PathBuf, Duration, Duration, usize)> = Vec::new();

    for (template_path, scan) in &jobs {
        let template = Template::parse_file(template_path)?;
        let page = scan
            .file_stem()
            .and_then(|s| s.to_string_lossy().parse::<usize>().ok())
            .map_or(0, |n| n.saturating_sub(1));

        let start = Instant::now();
        let image = Gray::open(scan)?;
        let decode = start.elapsed();

        // Two warm-up runs, then the best of `rounds`: the numbers then describe
        // the steady state instead of a cold first pass.
        for _ in 0..2 {
            let _ = omr::register_scan(&template, page, &image, None, Options::default());
        }
        let mut best = Duration::MAX;
        for _ in 0..rounds {
            let start = Instant::now();
            let recognition = omr::register_scan(&template, page, &image, None, Options::default());
            let elapsed = start.elapsed();
            // A scan whose corners are rejected is a defined outcome, not a failure.
            let _ = recognition;
            best = best.min(elapsed);
        }

        let start = Instant::now();
        let _ = Template::parse_file(template_path)?;
        total_parse += start.elapsed();

        total_bytes += std::fs::metadata(scan)?.len() as usize;
        let pixels = image.width() * image.height();
        total_pixels += pixels;
        per_scan.push((scan.clone(), best, decode, pixels));
    }

    per_scan.sort_by_key(|(_, d, _, _)| std::cmp::Reverse(*d));
    println!("slowest recognitions:");
    for (scan, d, _, px) in per_scan.iter().take(5) {
        println!(
            "  {:38} {:7.2} ms  {:.2} MP",
            scan.display().to_string().replace("tests/cases/", ""),
            d.as_secs_f64() * 1000.0,
            *px as f64 / 1e6
        );
    }
    println!("fastest:");
    for (scan, d, _, px) in per_scan.iter().rev().take(3) {
        println!(
            "  {:38} {:7.2} ms  {:.2} MP",
            scan.display().to_string().replace("tests/cases/", ""),
            d.as_secs_f64() * 1000.0,
            *px as f64 / 1e6
        );
    }

    let n = per_scan.len() as f64;
    let recognise_ms: f64 = per_scan
        .iter()
        .map(|(_, d, _, _)| d.as_secs_f64() * 1000.0)
        .sum();
    let decode_ms: f64 = per_scan
        .iter()
        .map(|(_, _, d, _)| d.as_secs_f64() * 1000.0)
        .sum();
    println!(
        "\nrecognition: mean {:.2} ms/scan, total {:.2} s for {} scans, {:.0} MP/s",
        recognise_ms / n,
        recognise_ms / 1000.0,
        per_scan.len(),
        (total_pixels as f64 / 1e6) / (recognise_ms / 1000.0)
    );
    println!(
        "jpeg decode: mean {:.2} ms/scan, total {:.2} s",
        decode_ms / n,
        decode_ms / 1000.0
    );
    println!(
        "template parse {:.2} ms per file, {:.2} MB of input",
        total_parse.as_secs_f64() * 1000.0 / n,
        total_bytes as f64 / 1e6
    );
    Ok(())
}

fn find(dir: &Path, extension: &str) -> Option<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|e| e == extension))
        .collect();
    found.sort();
    found.into_iter().next()
}

fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir)
        .expect("dir reads")
        .filter_map(Result::ok)
    {
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|n| n == "expected") {
                continue;
            }
            collect(&path, out);
        } else if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("jpg"))
        {
            out.push(path);
        }
    }
}
