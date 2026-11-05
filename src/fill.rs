//! Measurement and classification of filled cells.
//!
//! Two strategies are needed because the template declares two shapes of block.
//!
//! * Grouped cells (questions) are compared against a column baseline: the
//!   printed bracket and option letter appear in every row of the column, a
//!   pencil mark does not, so subtracting the clean-row median leaves only the
//!   pencil. Blocks whose first-row cells sit within a few pixels of each
//!   other form one group even when the template declares them separately;
//!   a column whose rows are mostly marked has no clean reference and falls
//!   back to the absolute measure.
//! * Solo cells (page number, absence bracket, candidate digits) have no
//!   siblings, so the interior of the cell is measured directly. Those boxes
//!   are enclosed and an empty one keeps a bright interior, which separates
//!   cleanly.

use crate::geometry::{Homography, Point, Rect};
use crate::gray::Gray;
use crate::template::Block;
use rayon::prelude::*;
use serde::Serialize;

/// Size every cell is resampled to before comparison.
pub const PATCH_W: usize = 28;
pub const PATCH_H: usize = 14;

#[derive(Clone, Debug, Serialize)]
pub struct CellMeasure {
    /// Fraction of the cell darker than its column baseline by the ink margin.
    pub residual: f64,
    /// `residual` after erosion, so thin printed strokes do not survive.
    pub solid: f64,
    /// Mean level of the cell, kept for diagnostics.
    pub mean_level: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct MeasureOptions {
    /// Gray levels below the reference that count as pencil ink.
    pub ink_margin: f64,
    /// Erosion kernel size applied to the residual mask.
    pub erode: usize,
    /// Fraction of each side ignored when measuring a single cell directly.
    pub solo_inset: f64,
}

impl Default for MeasureOptions {
    fn default() -> Self {
        Self {
            ink_margin: 32.0,
            erode: 2,
            solo_inset: 0.30,
        }
    }
}

/// Why a decision needs a human look.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewReason {
    /// Ink is present but weaker than a confident mark.
    Faint,
    /// More than one cell carries enough ink to be a mark.
    Multiple,
    /// Ink sits near the decision boundary, so a blank cannot be confirmed.
    NearBlank,
}

/// The outcome of [`decide`] for one question.
#[derive(Clone, Debug, Serialize)]
pub struct Decision {
    /// Index into the measured cell list, `None` when the group is blank.
    pub best: Option<usize>,
    /// Other cells that also carry ink, strongest first.
    pub alternates: Vec<usize>,
    pub confidence: f64,
    pub needs_review: bool,
    pub reason: Option<ReviewReason>,
}

/// Decision thresholds. The defaults are calibrated on the test corpus: real
/// marks measure 0.4-0.8 after baseline subtraction or 0.15-0.7 solo, printed
/// letters 0.2 or less, baseline speckle under 0.1.
#[derive(Clone, Copy, Debug)]
pub struct DecideOptions {
    /// Score above which a cell counts as marked at all.
    pub min_score: f64,
    /// A rival counts only when it reaches this fraction of the best score, so
    /// that the light halo around a pencil mark is not reported as a second one.
    pub rival_ratio: f64,
    /// Score expected from a normal mark; winners below it stay unconfirmed.
    pub confident_score: f64,
    /// Relative separation `(best - second) / best` for a clean single mark.
    pub min_separation: f64,
}

impl Default for DecideOptions {
    fn default() -> Self {
        Self {
            min_score: 0.008,
            rival_ratio: 0.45,
            confident_score: 0.040,
            min_separation: 0.25,
        }
    }
}

/// A cell resampled on a fixed lattice, in 0..255 ink levels.
///
/// Cells are only comparable on a common grid: template heights differ by a
/// pixel between rows, and the baseline machinery needs matching positions.
/// The lattice is `PATCH_W` x `PATCH_H` for multi-row groups; solo cells keep
/// their native size.
#[derive(Clone, Debug)]
pub struct Patch {
    /// Resampled gray levels, row-major, 0 = black.
    values: Vec<f64>,
    pub width: usize,
    pub height: usize,
}

impl Patch {
    /// Mean gray level; 255 for an empty patch.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            255.0
        } else {
            self.values.iter().sum::<f64>() / self.values.len() as f64
        }
    }
}

/// Samples `rect` on the standard `PATCH_W` x `PATCH_H` lattice in template
/// space.
pub fn sample_patch(scan: &Gray, transform: &Homography, rect: Rect) -> Patch {
    sample_patch_sized(scan, transform, rect, PATCH_W, PATCH_H)
}

/// Samples a rectangle on an explicit lattice.
///
/// Cells of one column differ by a pixel in height in the template, so they are
/// resampled on a shared lattice and compared at matching positions.
/// Samples a rectangle on an explicit lattice in template space.
pub fn sample_patch_sized(
    scan: &Gray,
    transform: &Homography,
    rect: Rect,
    width: usize,
    height: usize,
) -> Patch {
    // A scan finer than the template needs a small average, otherwise a single
    // bilinear lookup can land on the wrong side of a thin printed stroke.
    let radius = ((rect.width() / width as f64) * 0.5 - 0.35).clamp(0.0, 1.5);
    let (width, height) = (width.max(1), height.max(1));
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        let ty = rect.y0 + (y as f64 + 0.5) * rect.height() / height as f64;
        for x in 0..width {
            let tx = rect.x0 + (x as f64 + 0.5) * rect.width() / width as f64;
            values.push(average_around(
                scan,
                transform.apply(Point::new(tx, ty)),
                radius,
            ));
        }
    }
    Patch {
        values,
        width,
        height,
    }
}

/// Mean of the scan over a small disc around `p`.
///
/// Sampling a downscaled scan at a single point picks one of the original
/// pixels, so thin strokes alias away; averaging keeps their weight.
fn average_around(scan: &Gray, p: Point, radius: f64) -> f64 {
    if radius <= 0.35 {
        return scan.sample(p);
    }
    let step = radius.max(0.5);
    let mut sum = 0.0;
    let mut count = 0.0;
    let steps = 3isize;
    for iy in -steps..=steps {
        for ix in -steps..=steps {
            let dx = ix as f64 * step / steps as f64;
            let dy = iy as f64 * step / steps as f64;
            if dx * dx + dy * dy > radius * radius + 1e-9 {
                continue;
            }
            sum += scan.sample(Point::new(p.x + dx, p.y + dy));
            count += 1.0;
        }
    }
    if count > 0.0 {
        sum / count
    } else {
        scan.sample(p)
    }
}

/// Per-pixel median of several patches, i.e. the printed appearance with
/// marks and noise voted away. Requires at least two patches of equal size.
pub fn median_patch(patches: &[Patch]) -> Patch {
    let size = patches
        .first()
        .map_or(PATCH_W * PATCH_H, |p| p.values.len());
    let mut values = Vec::with_capacity(size);
    let mut column: Vec<f64> = Vec::with_capacity(patches.len());
    for index in 0..size {
        column.clear();
        for patch in patches {
            if let Some(value) = patch.values.get(index) {
                column.push(*value);
            }
        }
        column.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.push(column.get(column.len() / 2).copied().unwrap_or(255.0));
    }
    Patch {
        values,
        width: patches.first().map_or(PATCH_W, |p| p.width),
        height: patches.first().map_or(PATCH_H, |p| p.height),
    }
}

/// Baseline for one column of a multi-row block: the median of the clean
/// rows, which is the printed appearance of that column.
///
/// A filled cell is much darker than the printed bracket, so cells that are
/// already clearly inked are left out before taking the median; otherwise a
/// strongly marked row would darken the reference and shrink its own contrast.
pub fn column_baseline(patches: &[&Patch]) -> Option<Patch> {
    if patches.len() < 3 {
        return None;
    }
    // A printed letter darkens a row's mean by well under 100 levels even in
    // bold fonts; a pencil mark doubles that. Rows within this band of the
    // brightest row are clean and vote for the baseline.
    // Score each row by its deep-ink fraction: printed strokes barely move
    // it, a pencil mark dominates it. Rows like the cleanest one vote.
    let fractions: Vec<f64> = patches.iter().map(|p| deep_ink_fraction(p)).collect();
    let cleanest = fractions.iter().copied().fold(f64::MAX, f64::min);
    let clean: Vec<&Patch> = patches
        .iter()
        .zip(&fractions)
        .filter(|(_, fraction)| **fraction - cleanest < 0.25)
        .map(|(patch, _)| *patch)
        .collect();

    // Fall back to every row when marking is so heavy that nothing is clean.
    let usable: Vec<&Patch> = if clean.len() >= 2 {
        clean
    } else {
        patches.to_vec()
    };
    let owned: Vec<Patch> = usable.iter().map(|p| (*p).clone()).collect();
    Some(median_patch(&owned))
}

/// Measures one cell against its column baseline: a pixel counts as ink when
/// it is darker than the baseline by the ink margin. The eroded fraction is
/// the score that decides the reading.
fn measure_against(patch: &Patch, baseline: &Patch, options: MeasureOptions) -> CellMeasure {
    let total = patch.values.len().max(1) as f64;
    let mask: Vec<bool> = patch
        .values
        .iter()
        .zip(&baseline.values)
        .map(|(value, base)| base - value > options.ink_margin)
        .collect();
    let solid = erode(
        &mask,
        PATCH_W,
        patch.values.len() / PATCH_W.max(1),
        options.erode.max(1),
    );
    CellMeasure {
        residual: mask.iter().filter(|v| **v).count() as f64 / total,
        solid: solid.iter().filter(|v| **v).count() as f64 / total,
        mean_level: patch.mean(),
    }
}

/// Measures a single cell on its own.
///
/// Page-number boxes and the absence bracket are printed shapes, so the cell is
/// judged by how dark its interior is: an empty box keeps a bright centre even
/// though its outline is ink, a filled one does not.
fn measure_solo(patch: &Patch, options: MeasureOptions) -> CellMeasure {
    // A single cell is judged on its whole area. An empty option keeps a mostly
    // bright interior inside its printed bracket, while a marked one is filled in
    // solidly, so the mean level separates them without any background estimate.
    let paper = patch.values.iter().copied().fold(0.0f64, f64::max);
    let cutoff = paper - options.ink_margin;
    let mask: Vec<bool> = patch.values.iter().map(|v| *v < cutoff).collect();
    let solid = erode(&mask, patch.width, patch.height, options.erode.max(1));
    let total = patch.values.len().max(1) as f64;
    CellMeasure {
        residual: mask.iter().filter(|v| **v).count() as f64 / total,
        solid: solid.iter().filter(|v| **v).count() as f64 / total,
        mean_level: patch.mean(),
    }
}

/// Measures a group of sibling blocks against per-position baselines.
///
/// Siblings are read as a stack: cell `i` of one block is the same printed
/// shape as cell `i` of the others. For question rows that is option A of
/// every row; for the digit columns of a candidate number grid it is the glyph
/// for 0 of every column, so the same machinery removes both the printed
/// letter and the printed digit.
///
/// Every block is resampled on the shared lattice, each position gets a
/// baseline from the clean (unmarked) siblings, and each cell is measured as
/// "how much darker than the baseline" — which removes the printed letter,
/// bracket and digit exactly. Rows are returned in the order of `siblings`.
pub fn measure_group(
    scan: &Gray,
    transform: &Homography,
    siblings: &[&Block],
    options: MeasureOptions,
) -> Vec<Vec<CellMeasure>> {
    let lattice_h = PATCH_H.max(
        siblings
            .iter()
            .flat_map(|b| b.cells.iter())
            .map(|c| c.rect.height().round() as usize)
            .max()
            .unwrap_or(PATCH_H),
    );
    let patches: Vec<Vec<Patch>> = siblings
        .par_iter()
        .map(|block| {
            block
                .cells
                .iter()
                .map(|cell| sample_patch_sized(scan, transform, cell.rect, PATCH_W, lattice_h))
                .collect()
        })
        .collect();

    // Question blocks declare the same options in every row and digit columns
    // declare the same digits in every column, so a cell's position in its
    // block is the shape it prints; the number of positions is therefore the
    // cell count, not the highest column index.
    let positions = siblings
        .iter()
        .map(|b| b.cells.len())
        .max()
        .unwrap_or(0);
    let baselines: Vec<Option<Patch>> = (0..positions)
        .map(|column| {
            let stacked: Vec<&Patch> = patches.iter().filter_map(|row| row.get(column)).collect();
            column_baseline(&stacked)
        })
        .collect();

    // A column whose rows are almost all marked has no clean reference: its
    // median is the mark itself, and subtracting it erases real answers (a
    // column of five C marks would read blank). Measure such columns on
    // their own, like ungrouped blocks.
    let clean_counts: Vec<usize> = (0..positions)
        .map(|column| {
            let stacked: Vec<&Patch> = patches.iter().filter_map(|row| row.get(column)).collect();
            column_clean_count(&stacked)
        })
        .collect();

    patches
        .par_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .map(|(column, patch)| match baselines.get(column).and_then(|b| b.as_ref()) {
                    Some(baseline) if clean_counts.get(column).is_some_and(|&n| n >= 2) => {
                        measure_against(patch, baseline, options)
                    }
                    _ => measure_solo(patch, options),
                })
                .collect()
        })
        .collect()
}

/// Fraction of a patch that is deeply darkened relative to its own brightest
/// pixels: near zero for printed strokes, large for a pencil mark.
fn deep_ink_fraction(patch: &Patch) -> f64 {
    let paper = patch.values.iter().copied().fold(0.0f64, f64::max);
    let total = patch.values.len().max(1) as f64;
    patch
        .values
        .iter()
        .filter(|v| paper - **v > 60.0)
        .count() as f64
        / total
}

/// Number of rows light enough to vote for a column baseline: the same test
/// `column_baseline` uses to pick its clean rows.
fn column_clean_count(patches: &[&Patch]) -> usize {
    if patches.len() < 3 {
        return 0;
    }
    let fractions: Vec<f64> = patches.iter().map(|p| deep_ink_fraction(p)).collect();
    let cleanest = fractions.iter().copied().fold(f64::MAX, f64::min);
    fractions
        .iter()
        .filter(|f| **f - cleanest < 0.25)
        .count()
}

/// Samples a rectangle at its native template resolution.
///
/// A single cell is small, so resampling it onto a fixed lattice loses a whole
/// row or column of pixels and can shift the printed bracket across the cell
/// boundary. Sampling one point per template unit keeps the box aligned.
/// Samples a rectangle at its native template resolution, one sample per
/// pixel of the cell. A filled box has crisp edges, and bilinear blending
/// would soften exactly the boundary that separates a marked cell from its
/// printed bracket, so no resampling happens here.
pub fn sample_native(scan: &Gray, transform: &Homography, rect: Rect) -> Patch {
    let width = rect.width().round().max(2.0) as usize;
    let height = rect.height().round().max(2.0) as usize;
    let mut values = Vec::with_capacity(width * height);
    for y in 0..height {
        let ty = rect.y0 + (y as f64 + 0.5) * rect.height() / height as f64;
        for x in 0..width {
            let tx = rect.x0 + (x as f64 + 0.5) * rect.width() / width as f64;
            let p = transform.apply(Point::new(tx, ty));
            // One sample per pixel of the cell: a filled box has crisp edges and
            // bilinear blending would soften exactly the boundary that separates
            // a marked cell from its printed bracket.
            values.push(f64::from(
                scan.get(p.x.round().max(0.0) as usize, p.y.round().max(0.0) as usize),
            ));
        }
    }
    Patch {
        values,
        width,
        height,
    }
}

/// Measures every cell of a block on its own, in parallel.
pub fn measure_solo_block(
    scan: &Gray,
    transform: &Homography,
    block: &Block,
    options: MeasureOptions,
) -> Vec<CellMeasure> {
    block
        .cells
        .par_iter()
        .map(|cell| measure_solo(&sample_native(scan, transform, cell.rect), options))
        .collect()
}

/// Ranking score of a measured cell.
///
/// Only the eroded coverage counts. The uneroded residual also includes the
/// printed bracket, which is present in every option, so weighting it in pulls
/// the scores of blank and marked cells towards each other and can make a light
/// mark lose to a dark letter.
fn score(cell: &CellMeasure) -> f64 {
    cell.solid
}

/// Chooses the strongest cell and reports how clear that choice is.
///
/// The thresholds interlock: `min_score` is the absolute floor for any mark,
/// `confident_score` the level above which a read is trusted (baseline
/// subtraction leaves a few percent of speckle, real marks measure ten times
/// that), `rival_ratio` decides which runners-up count as also-marked, and
/// `min_separation` guards against near-ties that cannot be ranked.
pub fn decide(cells: &[CellMeasure], options: DecideOptions) -> Decision {
    if cells.is_empty() {
        return Decision {
            best: None,
            alternates: Vec::new(),
            confidence: 0.0,
            needs_review: true,
            reason: Some(ReviewReason::NearBlank),
        };
    }

    let mut order: Vec<usize> = (0..cells.len()).collect();
    order.sort_by(|a, b| {
        score(&cells[*b])
            .partial_cmp(&score(&cells[*a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = score(&cells[order[0]]);
    let second = order.get(1).map_or(0.0, |i| score(&cells[*i]));

    if best < options.min_score {
        let near = best > options.min_score * 0.5;
        return Decision {
            best: None,
            alternates: Vec::new(),
            confidence: (1.0 - best / options.min_score).clamp(0.0, 1.0),
            needs_review: near,
            reason: near.then_some(ReviewReason::NearBlank),
        };
    }
    // Below the confident mark level the reading is not a mark: baseline
    // subtraction leaves a few percent of residual on thick printed letters,
    // and reading the densest letter of a blank row would be wrong. Anything
    // under `confident_score` is blank unless ink is also present in rivals.
    if best < options.confident_score {
        return Decision {
            best: None,
            alternates: Vec::new(),
            confidence: best / options.confident_score,
            needs_review: true,
            reason: Some(ReviewReason::Faint),
        };
    }

    // Baseline subtraction leaves a few percent of speckle on bracket edges;
    // a real pencil mark measures ten times that. A winner this weak is the
    // paper speaking, not a mark: report nothing and let review decide.
    if best < options.confident_score * 2.5 {
        return Decision {
            best: None,
            alternates: Vec::new(),
            confidence: (best / (options.confident_score * 2.5)).clamp(0.0, 1.0),
            needs_review: true,
            reason: Some(ReviewReason::Faint),
        };
    }

    let rival_floor = (best * options.rival_ratio).max(options.min_score);
    let alternates: Vec<usize> = order
        .iter()
        .skip(1)
        .copied()
        .filter(|i| score(&cells[*i]) >= rival_floor)
        .collect();

    let separation = if best > 0.0 {
        ((best - second) / best).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let strength = (best / options.confident_score).clamp(0.0, 1.0);

    let reason = if !alternates.is_empty() || best - second < 1e-6 {
        // Two cells with the same score cannot be told apart; report both as
        // present rather than silently keeping the first.
        Some(ReviewReason::Multiple)
    } else if separation < options.min_separation {
        // The top cells are too close to rank. Two causes: a double mark
        // (rivals already reported) or printed letters of nearly equal
        // density, which is what a blank row of a thick-printed template
        // looks like after measurement. A pencil mark pushes its cell far
        // above the printed-letter level, so an unconvincing winner this
        // close to `confident_score` is paper, not pencil.
        if best < options.confident_score * 6.0 {
            return Decision {
                best: None,
                alternates: Vec::new(),
                confidence: strength,
                needs_review: true,
                reason: Some(ReviewReason::NearBlank),
            };
        }
        Some(ReviewReason::Faint)
    } else {
        None
    };

    // A read with no clear separation is either a double mark (rivals above
    // report both) or the printed paper: letters differ in ink density by
    // exactly this much on thick-printed templates. A pencil mark adds solid
    // coverage that pushes the winner far above `confident_score`, so a
    // low-separation winner near that level is the paper speaking, not a
    // pencil. Report no selection and let review catch it.
    let unconvincing = separation < options.min_separation
        && best < options.confident_score * 6.0;
    Decision {
        best: if unconvincing { None } else { Some(order[0]) },
        alternates,
        confidence: (0.55 * separation + 0.45 * strength).clamp(0.0, 1.0),
        needs_review: reason.is_some(),
        reason,
    }
}

/// Binary erosion with a square structuring element of `size` px applied
/// from every side. Thin printed strokes do not survive it; a pencil mark
/// does. Erosion runs on the eroded map only (no iterative passes).
pub fn erode(mask: &[bool], width: usize, height: usize, size: usize) -> Vec<bool> {
    if size <= 1 {
        return mask.to_vec();
    }
    let half = size / 2;
    if width <= size || height <= size {
        return vec![false; mask.len()];
    }
    let mut out = vec![false; mask.len()];
    for y in half..height - half {
        for x in half..width - half {
            let mut keep = true;
            'kernel: for dy in 0..size {
                for dx in 0..size {
                    if !mask[(y + dy - half) * width + (x + dx - half)] {
                        keep = false;
                        break 'kernel;
                    }
                }
            }
            out[y * width + x] = keep;
        }
    }
    out
}
