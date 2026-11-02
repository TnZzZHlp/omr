//! Reads a scanned answer sheet and prints the result as JSON.
//!
//! ```text
//! cargo run --example recognize -- 1.mb 77000e72a7ae415d9b44bdab0663caa7.jpg 0
//! cargo run --example recognize -- 1.mb 679aa3558b2c45eab1eec5a6f78a60db.jpg 1 0,0,1150,1603
//! ```
//!
//! Arguments: template, scan, template page index (0 based), optional
//! `x0,y0,x1,y1` crop in scan pixels when one file holds both pages.

use omr::{register_scan, Gray, Options, Rect, Template};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let template_path = args
        .next()
        .ok_or("usage: recognize <template.mb> <scan> <page> [x0,y0,x1,y1]")?;
    let scan_path = args.next().ok_or("missing scan path")?;
    let page: usize = args.next().unwrap_or_else(|| "0".into()).parse()?;
    let crop = args
        .next()
        .map(|raw| -> Result<Rect, Box<dyn std::error::Error>> {
            let parts: Vec<f64> = raw
                .split(',')
                .filter_map(|p| p.trim().parse().ok())
                .collect();
            if parts.len() != 4 {
                return Err("crop must be x0,y0,x1,y1".into());
            }
            Ok(Rect::new(parts[0], parts[1], parts[2], parts[3]))
        })
        .transpose()?;

    let template = Template::parse_file(&template_path)?;
    let scan = Gray::open(&scan_path)?;
    let recognition = register_scan(&template, page, &scan, crop, Options::default())?;

    eprintln!(
        "page {}: {} corner residual {:.2} px, rotation {:.3} deg",
        page + 1,
        recognition.registration.corners.len(),
        recognition.registration.max_residual,
        recognition.registration.rotation_deg,
    );
    for selection in &recognition.selections {
        eprintln!(
            "  Q{:>2} -> {:<4} confidence {:.2}{}",
            selection.question.unwrap_or(0),
            selection.selected.clone().unwrap_or_else(|| "-".into()),
            selection.confidence,
            if selection.needs_review {
                " needs review"
            } else {
                ""
            }
        );
    }
    println!("{}", serde_json::to_string_pretty(&recognition)?);
    Ok(())
}
