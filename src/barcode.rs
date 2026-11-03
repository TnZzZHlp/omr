//! Barcode decoding for the candidate-number region (`KSH`).
//!
//! The sheets carry Code 128, subset C: a start-C symbol, digit pairs, a
//! modulo-103 check symbol and a stop pattern. Decoding therefore needs the
//! element widths in modules, recovered from the column ink profile of the
//! barcode band.

use crate::error::{Error, Result};
use rayon::prelude::*;
use crate::geometry::{Homography, Point, Rect};
use crate::gray::Gray;

/// Symbol patterns of Code 128 in element widths, indexed by symbol value.
const PATTERNS: [&str; 107] = [
    "212222", "222122", "222221", "121223", "121322", "131222", "122213", "122312", "132212",
    "221213", "221312", "231212", "112232", "122132", "122231", "113222", "123122", "123221",
    "223211", "221132", "221231", "213212", "223112", "312131", "311222", "321122", "321221",
    "312212", "322112", "322211", "212123", "212321", "232121", "111323", "131123", "131321",
    "112313", "132113", "132311", "211313", "231113", "231311", "112133", "112331", "132131",
    "113123", "113321", "133121", "313121", "211331", "231131", "213113", "213311", "213131",
    "311123", "311321", "331121", "312113", "312311", "332111", "314111", "221411", "431111",
    "111224", "111422", "121124", "121421", "141122", "141221", "112214", "112412", "122114",
    "122411", "142112", "142211", "241211", "221114", "413111", "241112", "134111", "111242",
    "121142", "121241", "114212", "124112", "124211", "411212", "421112", "421211", "212141",
    "214121", "412121", "111143", "111341", "131141", "114113", "114311", "411113", "411311",
    "113141", "114131", "311141", "411131", "211412", "211214", "211232", "2331112",
];

const START_C: usize = 105;
const STOP: usize = 106;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BarcodeResult {
    /// Decoded digits, empty when decoding failed.
    pub text: String,
    /// True when the check symbol validated.
    pub check_ok: bool,
    /// Symbol values in order, start and check included.
    pub values: Vec<i32>,
    /// Largest deviation between an element and its module grid, in modules.
    pub width_error: f64,
    /// Why decoding failed, when it did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl BarcodeResult {
    fn failed(error: impl Into<String>) -> Self {
        Self {
            text: String::new(),
            check_ok: false,
            values: Vec::new(),
            width_error: f64::INFINITY,
            error: Some(error.into()),
        }
    }
}

/// Decodes the barcode inside `area` of the template page.
///
/// The block also holds the human readable number and the candidate name, and a
/// scan may be scaled or slightly blurred, so no single band and threshold is
/// trustworthy on its own. Several hypotheses are decoded and the first result
/// whose check symbol validates is returned, which is how a real scanner avoids
/// guessing between equally plausible element widths.
pub fn decode_block(scan: &Gray, transform: &Homography, area: Rect) -> Result<BarcodeResult> {
    if area.width() < 8.0 || area.height() < 4.0 {
        return Err(Error::Decode("barcode area is too small".into()));
    }
    let rows = area.height().round().max(1.0) as usize;
    let cols = area.width().round().max(1.0) as usize;

    // The lattice steps are constant under an affine transform: the x taps of
    // the bilinear blend are one lattice shared by every row, each row only
    // picks two scan rows and a weight. That removes the per-sample transform
    // apply, floor and clamp and leaves plain indexed blends; only a genuinely
    // projective fit falls back to per-point sampling.
    let affine = transform.is_affine();
    let step_x = transform.m[0] / transform.m[8];
    // Structure of arrays: the blend loop loads three contiguous lanes and
    // vectorizes; a tuple-of-three interleaves padding between lanes.
    let col_taps: Option<(Vec<u32>, Vec<u32>, Vec<f64>)> = if affine {
        let (mut lefts, mut rights, mut weights) =
            (Vec::with_capacity(cols), Vec::with_capacity(cols), Vec::with_capacity(cols));
        let mut px = transform.apply(Point::new(area.x0, area.y0)).x;
        for _ in 0..cols {
            let x0 = ((px - 0.5).floor() as isize).clamp(0, scan.width() as isize - 1);
            let fx = px - 0.5 - x0 as f64;
            let x1 = (x0 + 1).min(scan.width() as isize - 1);
            lefts.push(x0 as u32);
            rights.push(x1 as u32);
            weights.push(fx);
            px += step_x;
        }
        Some((lefts, rights, weights))
    } else {
        None
    };
    let col_taps = col_taps.as_ref();
    let mut ink = vec![0.0f64; rows * cols];
    let pixels = scan.pixels();
    let stride = scan.width();
    ink.par_chunks_mut(cols)
        .enumerate()
        .for_each(|(y, line)| {
        let ty = area.y0 + y as f64;
        match col_taps {
            Some((lefts, rights, weights)) => {
                let sy = transform.apply(Point::new(area.x0, ty)).y - 0.5;
                let row_y = sy.floor();
                let fy = sy - row_y;
                let y0 = (row_y as isize).clamp(0, scan.height() as isize - 1) as usize;
                let y1 = (row_y as isize + 1).clamp(0, scan.height() as isize - 1) as usize;
                let (row_top, row_bottom) = (y0 * stride, y1 * stride);
                let top_w = 1.0 - fy;
                let scale = 1.0 / 255.0;
                for (x, ((&xa, &xb), &fx)) in
                    lefts.iter().zip(rights.iter()).zip(weights.iter()).enumerate()
                {
                    let value = &mut line[x];
                    let w_l = 1.0 - fx;
                    let top = (f64::from(pixels[row_top + xa as usize]) * w_l
                        + f64::from(pixels[row_top + xb as usize]) * fx) * top_w;
                    let bottom = (f64::from(pixels[row_bottom + xa as usize]) * w_l
                        + f64::from(pixels[row_bottom + xb as usize]) * fx) * fy;
                    *value = 255.0 * scale - (top + bottom) * scale;
                }
            }
            None => {
                for (x, value) in line.iter_mut().enumerate() {
                    let tx = area.x0 + x as f64;
                    let p = transform.apply(Point::new(tx, ty));
                    *value = (255.0 - scan.sample(p)) / 255.0;
                }
            }
        }
    });

    // A barcode row changes from ink to paper many times across the block; a row
    // of text changes a few times and a blank row not at all. Counting those
    // transitions therefore finds the symbol rows even when a line of text is
    // darker overall.
    let transitions: Vec<usize> = ink
        .chunks(cols)
        .map(|line| {
            line.windows(2)
                .filter(|pair| (pair[0] > 0.5) != (pair[1] > 0.5))
                .count()
        })
        .collect();
    let busiest = transitions.iter().copied().max().unwrap_or(0);
    if busiest < 10 {
        return Err(Error::Decode(
            "the barcode block contains no bar pattern".into(),
        ));
    }

    let mut bands: Vec<(usize, usize)> = Vec::new();
    for fraction in [0.8f64, 0.65, 0.5] {
        let threshold = (busiest as f64 * fraction) as usize;
        let mut start_row: Option<usize> = None;
        for (y, count) in transitions.iter().enumerate() {
            if *count >= threshold {
                start_row.get_or_insert(y);
            } else if let Some(begin) = start_row.take() {
                if y - begin >= 6 {
                    bands.push((begin, y - 1));
                }
            }
        }
        if let Some(begin) = start_row {
            if rows - begin >= 6 {
                bands.push((begin, rows - 1));
            }
        }
    }
    bands.sort_by_key(|(a, b)| b - a);
    bands.dedup();
    if bands.is_empty() {
        return Err(Error::Decode("no barcode band found in the block".into()));
    }

    let mut best: Option<BarcodeResult> = None;
    for (top, bottom) in bands {
        // The symbol is printed inside a wide block and its exact left edge
        // varies with the printer, so the horizontal extent is measured from the
        // band rather than taken from the template.
        let mut column_ink = vec![0.0f64; cols];
        for line in ink[top * cols..(bottom + 1) * cols].chunks(cols) {
            for (target, value) in column_ink.iter_mut().zip(line) {
                *target += *value;
            }
        }
        let row_count = (bottom - top + 1) as f64;
        for value in &mut column_ink {
            *value /= row_count;
        }
        let tallest = column_ink.iter().copied().fold(0.0, f64::max);
        let edges: Vec<usize> = (0..cols)
            .filter(|x| column_ink[*x] > tallest * 0.25)
            .collect();
        let (Some(&first), Some(&last)) = (edges.first(), edges.last()) else {
            continue;
        };
        let mut profile = vec![0.0f64; cols];
        for line in ink[top * cols..(bottom + 1) * cols].chunks(cols) {
            for (target, value) in profile.iter_mut().zip(line) {
                *target += value;
            }
        }
        let count = (bottom - top + 1) as f64;
        for value in &mut profile {
            *value /= count;
        }
        let window = &profile[first..=last];
        for level in [0.35, 0.5, 0.6] {
            for chunk in run_lengths(window, level) {
                // A symbol can still have a few stray elements at either end, so
                // the decoder tries the chunk with its edges trimmed as well.
                let candidate = decode_elements(&[chunk]);
                if candidate.check_ok {
                    return Ok(candidate);
                }
                if candidate.error.is_none()
                    && best
                        .as_ref()
                        .map_or(true, |b| candidate.width_error < b.width_error)
                {
                    best = Some(candidate);
                }
            }
        }
    }

    best.ok_or_else(|| Error::Decode("the barcode could not be decoded".into()))
}

/// Alternating bar and space widths, measured at the Otsu level of the profile.
///
/// The profile of a barcode is close to two-valued, so the level that best
/// separates ink from paper is a stable place to cut; a fraction of the peak
/// would instead follow the noise of a single bright column.
/// Measures the alternating bar and space widths of `profile` at the given
/// fraction of its peak, returning one candidate chunk per contiguous run.
/// Widths are in profile columns, normalised later against the module count.
fn run_lengths(profile: &[f64], level: f64) -> Vec<Vec<f64>> {
    let peak = profile.iter().copied().fold(0.0, f64::max);
    let floor = profile.iter().copied().fold(f64::MAX, f64::min);
    if peak - floor < 0.1 {
        return Vec::new();
    }
    let threshold = peak * level;

    let Some(first) = profile.iter().position(|v| *v > threshold) else {
        return Vec::new();
    };
    let Some(last) = profile.iter().rposition(|v| *v > threshold) else {
        return Vec::new();
    };
    let slice = &profile[first..=last];

    let mut runs: Vec<(bool, f64)> = Vec::new();
    let mut is_bar = slice[0] > threshold;
    let mut width = 0.0;
    for index in 0..slice.len() {
        let value = slice[index] > threshold;
        if value == is_bar {
            width += 1.0;
            continue;
        }
        let (a, b) = (slice[index - 1], slice[index]);
        let t = if (b - a).abs() < 1e-9 {
            0.5
        } else {
            ((threshold - a) / (b - a)).clamp(0.0, 1.0)
        };
        runs.push((is_bar, width - 1.0 + t));
        is_bar = value;
        width = 1.0 - t;
    }
    runs.push((is_bar, width));
    runs.retain(|(_, w)| *w > 0.05);

    // A gap much wider than one module separates the symbol from the number and
    // the name printed around it. The module width is unknown before decoding, so
    // it is estimated from the narrow elements of the profile; a gap also has to
    // stand out from the widest element, which is four modules.
    let mut narrow: Vec<f64> = runs.iter().map(|(_, w)| *w).collect();
    narrow.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let module = narrow
        .get(narrow.len() / 4)
        .copied()
        .unwrap_or(1.0)
        .max(1.0);
    let gap_limit = (module * 6.0).max(12.0);

    // The leading and trailing blank of the band belongs to the surrounding
    // paper, not to the symbol, so it is dropped as well.
    let mut chunks: Vec<Vec<f64>> = Vec::new();
    let mut current: Vec<f64> = Vec::new();
    for (bar, w) in runs {
        if !bar && w > gap_limit {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
        } else {
            current.push(w);
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }

    // Trim the slivers that a noisy profile leaves at either end and the blank
    // elements that belong to the paper; neither is part of the symbol.
    for chunk in &mut chunks {
        while chunk.len() > 1 && (chunk[0] < module * 0.45 || chunk[0] > module * 3.0) {
            chunk.remove(0);
        }
        while chunk.len() > 1
            && chunk
                .last()
                .is_some_and(|w| *w < module * 0.45 || *w > module * 3.0)
        {
            chunk.pop();
        }
    }
    chunks.retain(|c| !c.is_empty());
    chunks
}

/// Picks the element run that fits a Code 128 grid and decodes it.
///
/// A symbol occupies six elements and eleven modules, the stop pattern seven
/// elements and thirteen modules, so an `n` symbol code has `6n + 7` elements
/// spanning `11n + 13` modules. Scan noise blurs the outer edges of the band, so
/// a few leading and trailing elements are dropped until the count matches, and
/// the best fit wins.
/// Decodes Code 128 element widths into values, trying the edge trims until
/// the element count matches a legal symbol layout.
fn decode_elements(chunks: &[Vec<f64>]) -> BarcodeResult {
    let mut best: Option<(f64, Vec<i32>)> = None;
    for chunk in chunks {
        for drop_front in 0..=2usize {
            for drop_back in 0..=2usize {
                if chunk.len() < drop_front + drop_back + 13 {
                    continue;
                }
                let body = &chunk[drop_front..chunk.len() - drop_back];
                let Some(symbols) = body.len().checked_sub(7).map(|n| n / 6) else {
                    continue;
                };
                if symbols < 2 || body.len() != 6 * symbols + 7 {
                    continue;
                }
                let modules = 11 * symbols + 13;
                let total: f64 = body.iter().sum();
                let module = total / modules as f64;
                if module <= 0.0 {
                    continue;
                }
                let quantized: Vec<i32> =
                    body.iter().map(|w| (w / module).round() as i32).collect();
                if quantized.iter().any(|q| !(1..=4).contains(q)) {
                    continue;
                }
                if quantized.iter().sum::<i32>() as usize != modules {
                    continue;
                }
                let error = body
                    .iter()
                    .zip(&quantized)
                    .map(|(w, q)| (w - module * f64::from(*q)).abs())
                    .fold(0.0f64, f64::max)
                    / module;
                if best.as_ref().map_or(true, |b| error < b.0) {
                    best = Some((error, quantized));
                }
            }
        }
    }

    let Some((width_error, quantized)) = best else {
        return BarcodeResult::failed("element widths do not fit a Code 128 grid");
    };

    let mut values = Vec::new();
    let mut index = 0;
    while index + 6 <= quantized.len() {
        // The stop pattern is one element longer than a regular symbol.
        let stop_start = quantized.len() - 7;
        if index == stop_start {
            let pattern: String = quantized[index..]
                .iter()
                .map(|v| char::from(b'0' + *v as u8))
                .collect();
            if pattern == PATTERNS[STOP] {
                values.push(STOP as i32);
            } else {
                return BarcodeResult {
                    text: String::new(),
                    check_ok: false,
                    values,
                    width_error,
                    error: Some(format!("unexpected stop pattern {pattern}")),
                };
            }
            break;
        }
        let pattern: String = quantized[index..index + 6]
            .iter()
            .map(|v| char::from(b'0' + *v as u8))
            .collect();
        let Some(value) = PATTERNS.iter().position(|p| *p == pattern) else {
            return BarcodeResult {
                text: String::new(),
                check_ok: false,
                values,
                width_error,
                error: Some(format!("unknown symbol pattern {pattern}")),
            };
        };
        values.push(value as i32);
        index += 6;
    }

    decode_values(values, width_error)
}

/// Validates a decoded value sequence: start symbol, check digit (modulo 103
/// over the data values), and stop pattern. `width_error` is the residual of
/// the width fit, kept so the caller can prefer the tightest of several
/// hypotheses.
fn decode_values(values: Vec<i32>, width_error: f64) -> BarcodeResult {
    let Some(&first) = values.first() else {
        return BarcodeResult::failed("barcode has no start symbol");
    };
    if first as usize != START_C {
        return BarcodeResult {
            text: String::new(),
            check_ok: false,
            values,
            width_error,
            error: Some("barcode does not start with Code 128 subset C".into()),
        };
    }

    let payload: Vec<i32> = values
        .iter()
        .copied()
        .skip(1)
        .take_while(|v| (0..100).contains(v))
        .collect();
    let Some((&check, data)) = payload.split_last() else {
        return BarcodeResult::failed("barcode carries no data symbols");
    };

    let mut sum = first;
    for (position, value) in data.iter().enumerate() {
        sum += value * (position as i32 + 1);
    }
    let check_ok = sum % 103 == check;

    BarcodeResult {
        text: data.iter().map(|value| format!("{value:02}")).collect(),
        check_ok,
        values,
        width_error,
        error: if check_ok {
            None
        } else {
            Some(format!(
                "check symbol {check} does not match computed {}",
                sum % 103
            ))
        },
    }
}
