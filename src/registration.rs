//! Registration of a scan against the template coordinate space.
//!
//! Each template page declares four corner marks. The template was generated
//! from a scan of the same sheet, so a corner mark always sits at approximately
//! the position the template records; registration therefore searches for the
//! darkest solid mark near each expected position instead of grouping every
//! dark blob on the page. That keeps working when a mark touches handwriting
//! or a printed rule, which merge it into a larger component.
//!
//! Two paths exist, tried in order by [`register`]:
//!
//! * **Corner marks** (`register_by_marks`). Pass 1 locates each mark with
//!   escalating darkness sweeps, falling back to stamp correlation for shapes
//!   darkness cannot rank (triangles, rings) and to a second first-guess
//!   (template spread over the whole scan) when the identity guess fails.
//!   Pass 2 fits a homography, re-locates from the predicted positions and
//!   fits again. The fit reports its residuals, so an implausible layout
//!   (wrong sheet side, wrong scale) can be rejected outright.
//! * **Content alignment** (`register_by_content`). When the marks are
//!   missing or cropped off, the printed brackets and letters of the answer
//!   blocks align the scan by translation: a scored comparison of "ink inside
//!   the declared cells vs ink between them" over all blocks, searched by
//!   coarse-to-fine grid. Confidence on such a page is scaled by 0.7 by the
//!   recognizer because rotation and scale are uncompensated.

use crate::error::{Error, Result};
use crate::geometry::{Homography, Point, Rect};
use crate::gray::Gray;
use crate::template::{Block, BlockKind, Gray2, Page};
use rayon::prelude::*;

/// How the scan was aligned to the template.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlignmentMode {
    /// All four corner marks were found and fitted.
    CornerRegistered,
    /// No (or too few) corner marks; the readings were aligned on the printed
    /// content of the answer blocks themselves.
    ContentAligned,
}

impl AlignmentMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CornerRegistered => "corner_registered",
            Self::ContentAligned => "content_aligned",
        }
    }
}

/// Registration outcome for one page.
#[derive(Clone, Debug)]
pub struct Registration {
    pub transform: Homography,
    /// How this alignment was produced.
    pub mode: AlignmentMode,
    /// Template corner positions used for the fit.
    pub template_points: Vec<Point>,
    /// Corresponding positions measured in the scan.
    pub image_points: Vec<Point>,
    /// Per-corner reprojection error in pixels.
    pub residuals: Vec<f64>,
    /// Slowest corner error, the main quality indicator.
    pub max_residual: f64,
    /// Rotation of the fitted x axis in degrees.
    pub rotation_deg: f64,
    /// Scale along the x axis, relative to the template.
    pub scale_x: f64,
    /// Scale along the y axis, relative to the template.
    pub scale_y: f64,
}

impl Registration {
    /// `tolerance` is in scan pixels; the template's pixel size is only nominal.
    pub fn is_usable(&self, tolerance: f64) -> bool {
        self.max_residual <= tolerance
    }

    /// True when the marks were found but do not describe a page of this
    /// template: the corner errors are far beyond what a scan or a photograph
    /// produces, or the sheet is at the wrong scale.
    ///
    /// This is what tells "the wrong side of the sheet was scanned" apart from a
    /// slightly skew page, and it is the reason a reading of such a scan must be
    /// discarded rather than reported.
    pub fn is_implausible(&self, mark_size: f64) -> bool {
        // A corner mark is a fixed size on paper, so the fit that matches it best
        // has an error of at most about one mark.
        let allowed = (mark_size * 1.5).max(12.0);
        self.max_residual > allowed
            || !(0.5..=2.0).contains(&(self.scale_x / self.scale_y))
            || self.scale_x < 0.6
            || self.scale_x > 1.8
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DetectOptions {
    /// Mean darkness a corner window must reach, on a 0..255 scale. A filled mark
    /// is near 200, blank paper near 0.
    pub min_mark_darkness: f64,
    /// Correlation a template stamp must reach to be accepted, 0..1.
    pub min_patch_score: f64,
    /// Search radius around the expected position, as a multiple of the mark
    /// size.
    pub search_radius: f64,
    /// Fraction of the mark size used as the sampled window.
    pub window_ratio: f64,
}

impl Default for DetectOptions {
    fn default() -> Self {
        Self {
            min_mark_darkness: 90.0,
            min_patch_score: 0.55,
            search_radius: 3.0,
            window_ratio: 0.8,
        }
    }
}

/// Aligns the scan on the printed content of the answer blocks (the fallback
/// when the corner marks cannot be found).
///
/// Every block with cells (question rows, page numbers) contributes a score:
/// mean ink inside the declared cells minus mean ink in the gaps between them,
/// on a binarized, box-blurred copy of the scan addressed through an integral
/// image. The true offset lines cells up with ink and gaps with paper, so it
/// is the maximum of the score over the translation. Scoring all blocks
/// together breaks the row-pitch alias of single question rows; a rightward
/// extension of each block area penalizes one-cell x aliases; blocks are
/// skipped near the page edge only while most of the layout stays visible; and
/// a final score threshold rejects a scan of a different template outright.
fn register_by_content(page: &Page, scan: &Gray, _options: DetectOptions) -> Result<Registration> {
    let blocks: Vec<&Block> = page
        .blocks
        .iter()
        .filter(|b| !b.cells.is_empty() && b.kind != BlockKind::Absent)
        .collect();
    if blocks.is_empty() {
        return Err(Error::Registration(
            "the page declares neither corner marks nor answer blocks".into(),
        ));
    }

    // Content alignment: score = mean ink inside the declared cells minus mean
    // ink in the gaps between them, over every block with cells (question rows,
    // page numbers, barcodes). Blank paper scores 0; the true offset lines
    // cells up with ink and gaps with paper. Scoring all blocks together breaks
    // the row-pitch alias of single question rows, and box-blurring the ink map
    // keeps the score smooth in dx and dy so a coarse grid search lands in the
    // right basin before a fine pass refines it. An integral image over the
    // blurred map makes each grid point cost one lookup per cell instead of a
    // pass over the pixels.
    let (w, h) = (scan.width(), scan.height());
    let mut all: Vec<u8> = scan.pixels().iter().step_by(7).copied().collect();
    all.sort();
    let paper = f64::from(all[all.len() * 9 / 10].max(1));
    let ink_level = (paper - 45.0).max(1.0) as u8;

    // Blur with sliding sums, binarizing on the fly: the vertical pass sums 9
    // rows (replicated at the edges) of the thresholded scan into a u16 (max
    // 9), the horizontal pass sums 5 of those into a u16 (max 45). No divisions
    // and no intermediate byte image: the integral image carries raw sums and
    // the score divides once per rect. The vertical pass walks columns in
    // stripes of 64 so the working set stays inside L1.
    let pixels = scan.pixels();
    let binarize = |index: usize| -> u16 { u16::from(pixels[index] < ink_level) };
    let mut vblur = vec![0u16; w * h];
    // Row bands with one running window per column: the window at a band's
    // first row is summed once, then slides down the band, so each pass
    // touches every pixel of the scan about three times in row order and the
    // bands run on separate threads.
    vblur
        .par_chunks_mut(w * 64)
        .enumerate()
        .for_each(|(band, rows)| {
            let y0 = band * 64;
            let mut acc = vec![0u16; w];
            let lo = y0.saturating_sub(4);
            let hi = (y0 + 4).min(h - 1);
            for (x, slot) in acc.iter_mut().enumerate() {
                let mut sum = 0u16;
                for yy in lo..=hi {
                    sum += binarize(yy * w + x);
                }
                *slot = sum;
            }
            for (local, row) in rows.chunks_mut(w).enumerate() {
                let y = y0 + local;
                for (slot, value) in row.iter_mut().zip(acc.iter()) {
                    *slot = *value;
                }
                let add = y + 5;
                if add < h {
                    let row_add = add * w;
                    for (value, slot) in pixels[row_add..row_add + w].iter().zip(acc.iter_mut()) {
                        *slot += u16::from(*value < ink_level);
                    }
                }
                if y >= 4 {
                    let row_sub = (y - 4) * w;
                    for (value, slot) in pixels[row_sub..row_sub + w].iter().zip(acc.iter_mut()) {
                        *slot -= u16::from(*value < ink_level);
                    }
                }
            }
        });
    let mut blur = vec![0u16; w * h];
    blur.par_chunks_mut(w)
        .enumerate()
        .for_each(|(y, row)| {
            let src = y * w;
            let mut acc = 0u16;
            for dx in 0..2 {
                acc += vblur[src + dx];
            }
            for (x, slot) in row.iter_mut().enumerate() {
                let add = (x + 2).min(w - 1);
                acc += vblur[src + add];
                if x >= 3 {
                    acc -= vblur[src + x - 3];
                }
                *slot = acc;
            }
        });
    // integral[i * (w + 1) + j] = sum of blur over rows 0..i, columns 0..j.
    // Two passes: row sums in parallel (independent), then the serial column
    // prefix over h rows — each step is one u32 add per cell over a contiguous
    // (w+1)-stride.
    let mut integral = vec![0u32; (w + 1) * (h + 1)];
    integral
        .par_chunks_mut(w + 1)
        .enumerate()
        .for_each(|(y, row)| {
            if y == 0 {
                return;
            }
            let src = (y - 1) * w;
            let mut row_sum = 0u32;
            for (x, slot) in row[1..].iter_mut().enumerate() {
                row_sum += u32::from(blur[src + x]);
                *slot = row_sum;
            }
        });
    for y in 1..h {
        let stride = w + 1;
        let (prev_row, cur_row) = integral[y * stride..].split_at_mut(stride);
        for (dst, src) in cur_row.iter_mut().zip(&prev_row[..stride]) {
            *dst += *src;
        }
    }
    let rect_sum = |x0: i32, y0: i32, x1: i32, y1: i32| -> u32 {
        // The score path only calls this with rects proven to lie inside the
        // scan, so no clamping: four loads and three adds per rect.
        debug_assert!(x0 >= 0 && y0 >= 0 && x1 <= w as i32 && y1 <= h as i32);
        let (xa, ya) = (x0 as usize, y0 as usize);
        let (xb, yb) = (x1 as usize, y1 as usize);
        integral[yb * (w + 1) + xb] + integral[ya * (w + 1) + xa]
            - integral[ya * (w + 1) + xb]
            - integral[yb * (w + 1) + xa]
    };

    // Cell rects relative to each block area, so a grid point only adds an
    // offset to the area origin. The area is extended to the right by one cell
    // pitch: the bracket that a one-pitch x alias would leave uncovered then
    // falls into the extension and counts as gap ink, which is exactly the
    // penalty the periodic bracket pattern cannot provide on its own. The
    // extension stops at the next block so multi-column layouts keep the
    // inter-column gap as expected blank paper.
    let mut layout: Vec<(Rect, Vec<Rect>)> = blocks
        .iter()
        .map(|b| {
            let origin = Point::new(b.area.x0, b.area.y0);
            let cells = b
                .cells
                .iter()
                .map(|c| c.rect.translate(Point::new(-origin.x, -origin.y)))
                .collect();
            (b.area, cells)
        })
        .collect();
    for i in 0..layout.len() {
        let (area, cells) = &layout[i];
        if cells.is_empty() {
            continue;
        }
        let pitch = (area.width() / cells.len() as f64).max(1.0);
        let mut ext = pitch;
        for (other, _) in layout.iter().skip(i + 1) {
            let overlaps_y = other.y0 < area.y1 && area.y0 < other.y1;
            if overlaps_y && other.x0 > area.x1 {
                ext = ext.min(other.x0 - area.x1);
            }
        }
        ext = ext.min(scan.width() as f64 - area.x1).max(0.0);
        layout[i].0 = Rect::new(area.x0, area.y0, area.x1 + ext, area.y1);
    }

    // Precompute the scoring layout in integers: the closure runs tens of
    // thousands of times, so the per-point work is four loads per rect plus
    // nothing else. Areas are constants, not per-point multiplications.
    struct ScoreBlock {
        area: [i32; 4],
        cells: Vec<[i32; 4]>,
        area_area: f64,
        cell_area: f64,
    }
    let score_blocks: Vec<ScoreBlock> = layout
        .iter()
        .map(|(area, cells)| ScoreBlock {
            area: [
                area.x0 as i32,
                area.y0 as i32,
                area.x1 as i32,
                area.y1 as i32,
            ],
            cells: cells
                .iter()
                .map(|c| {
                    [c.x0 as i32, c.y0 as i32, c.x1 as i32, c.y1 as i32]
                })
                .collect(),
            area_area: area.width() * area.height(),
            cell_area: cells.iter().map(|c| c.width() * c.height()).sum(),
        })
        .collect();
    let total_cell_area: f64 = score_blocks.iter().map(|b| b.cell_area).sum();
    let score_at = |dx: i32, dy: i32| -> Option<f64> {
        let (mut cell_sum, mut cell_area, mut gap_sum, mut gap_area, mut visible) =
            (0u64, 0.0f64, 0u64, 0.0f64, 0.0f64);
        for block in &score_blocks {
            let (ax, ay) = (block.area[0] + dx, block.area[1] + dy);
            let (ax1, ay1) = (block.area[2] + dx, block.area[3] + dy);
            if ax < 0 || ay < 0 || ax1 > w as i32 || ay1 > h as i32 {
                // The scan may be cropped inside the template page, so blocks
                // falling off its edge are skipped rather than rejected.
                continue;
            }
            visible += block.cell_area;
            let area_sum = u64::from(rect_sum(ax, ay, ax1, ay1));
            let mut cells = 0u64;
            for cell in &block.cells {
                // Cells are relative to the block area, whose origin already
                // carries the offset.
                cells += u64::from(rect_sum(
                    ax + cell[0],
                    ay + cell[1],
                    ax + cell[2],
                    ay + cell[3],
                ));
            }
            cell_sum += cells;
            gap_sum += area_sum - cells;
            cell_area += block.cell_area;
            gap_area += block.area_area - block.cell_area;
        }
        if cell_area < 400.0 || gap_area < 400.0 {
            return None;
        }
        // A tiny visible fraction makes the score incomparable with full-page
        // offsets, so far-off candidates cannot win by scoring a convenient
        // subset of blocks.
        if visible < 0.6 * total_cell_area {
            return None;
        }
        // ink = 2 * blur - 1 inside cells and gaps, so the difference of the
        // two means is twice the difference of the blur means; blur stores the
        // raw sum over 45 taps, so the means divide by 45 once here.
        Some(
            2.0 * (cell_sum as f64 / cell_area - gap_sum as f64 / gap_area)
                / 45.0,
        )
    };

    let span = 600i32;
    let coarse = 8i32;
    let rows: Vec<i32> = (-span..=span).step_by(coarse as usize).collect();
    let best: Option<(f64, i32, i32)> = rows
        .par_iter()
        .map(|&dy| {
            let mut best: Option<(f64, i32, i32)> = None;
            for dx in (-span..=span).step_by(coarse as usize) {
                if let Some(s) = score_at(dx, dy) {
                    if best.as_ref().map_or(true, |b| s > b.0) {
                        best = Some((s, dx, dy));
                    }
                }
            }
            best
        })
        .reduce(
            || None,
            |a, b| match (a, b) {
                (Some(a), Some(b)) => Some(if b.0 > a.0 || (b.0 == a.0 && (b.2, b.1) < (a.2, a.1)) {
                    b
                } else {
                    a
                }),
                (a, b) => a.or(b),
            },
        );
    let Some((_, cx, cy)) = best else {
        return Err(Error::Registration(
            "no printed content found to align on".into(),
        ));
    };

    let mut best: Option<(f64, i32, i32)> = None;
    for dy in (cy - coarse)..=(cy + coarse) {
        for dx in (cx - coarse)..=(cx + coarse) {
            if let Some(s) = score_at(dx, dy) {
                if best.map_or(true, |b| s > b.0) {
                    best = Some((s, dx, dy));
                }
            }
        }
    }
    let Some((score, ox, oy)) = best else {
        return Err(Error::Registration(
            "no printed content found to align on".into(),
        ));
    };
    if score < 0.15 {
        return Err(Error::Registration(
            "the printed content does not match this template".into(),
        ));
    }
    let (offset_x, offset_y) = (ox as f64, oy as f64);

    Ok(Registration {
        transform: Homography::scale_translate(1.0, 1.0, offset_x, offset_y),
        mode: AlignmentMode::ContentAligned,
        template_points: Vec::new(),
        image_points: Vec::new(),
        residuals: Vec::new(),
        max_residual: offset_x.hypot(offset_y),
        rotation_deg: 0.0,
        scale_x: 1.0,
        scale_y: 1.0,
    })
}

/// Fits the scan to the template page: corner marks first, printed content as
/// the fallback. The error of whichever path was tried first is reported when
/// both refuse, so a wrong-page scan still says "no corner mark near …".
pub fn register(page: &Page, scan: &Gray, options: DetectOptions) -> Result<Registration> {
    match register_by_marks(page, scan, options) {
        Ok(registration) => Ok(registration),
        Err(mark_error) => register_by_content(page, scan, options).map_err(|_| mark_error),
    }
}

/// Registers using the four corner marks.
///
/// Pass 1 locates each mark independently: darkness sweeps with an escalating
/// radius, a stamp correlation for marks the darkness window cannot rank, and
/// a second first-guess (template spread over the whole scan) when the
/// identity guess fails. Pass 2 fits a homography from pass 1, re-locates the
/// marks from the predicted positions, and fits again; the transform between
/// the recorded and fitted positions is what the caller applies.
fn register_by_marks(page: &Page, scan: &Gray, options: DetectOptions) -> Result<Registration> {
    let corners: Vec<(Point, f64, Option<&Gray2>, Option<Rect>)> = page
        .marks
        .iter()
        .filter_map(|mark| {
            mark.found.map(|found| {
                let window = mark.window.then_some(found);
                (
                    mark_point(mark, found),
                    mark_size(mark),
                    mark.patch.as_ref(),
                    window,
                )
            })
        })
        .collect();
    if corners.len() < 4 {
        return Err(Error::Registration(format!(
            "page {} declares {} usable corner marks, four are required",
            page.index + 1,
            corners.len()
        )));
    }

    // First pass: the template records the four corners of the sheet and the scan
    // is that sheet, so mapping the template's bounding rectangle onto the scan's
    // gives a first guess for every corner.
    let template_box = bounding_box(corners.iter().map(|(p, ..)| *p));
    // Two first guesses are worth trying: the marks mapped onto the whole scan,
    // and the template at its own scale in place. The former suits a scan that
    // fills the frame, the latter one where the sheet sits on a larger platen;
    // whichever finds all four corners first wins.
    let guess = Homography::identity();
    let guess_scale = coarse_scale(&guess);

    let mut image_points = Vec::with_capacity(4);
    let mut search = DarknessSearch::new();
    for (center, size, patch, window) in &corners {
        let expected = guess.apply(*center);
        let mut found = None;
        let mut last_span: Option<isize> = None;
        for radius in [3.0f64, 6.0, 12.0, 24.0] {
            let attempt = DetectOptions {
                search_radius: radius,
                ..options
            };
            // The stamp search spans a window clamped at 40 px, so once the
            // escalation widens past that, repeating it would find exactly what
            // the previous attempt found; only the darkness window grows.
            let span = stamp_span(*size * guess_scale, &attempt);
            let attempt_patch = if last_span == Some(span) { None } else { *patch };
            last_span = Some(span);
            if let Some(point) =
                locate_mark(scan, expected, *size * guess_scale, attempt_patch, *window, attempt, &mut search)
            {
                found = Some(point);
                break;
            }
        }
        if found.is_none() {
            let spread = Homography::scale_translate(
                scan.width() as f64 / template_box.width().max(1.0),
                scan.height() as f64 / template_box.height().max(1.0),
                -template_box.x0 * scan.width() as f64 / template_box.width().max(1.0),
                -template_box.y0 * scan.height() as f64 / template_box.height().max(1.0),
            );
            let expected = spread.apply(*center);
            let mut last_span: Option<isize> = None;
            for radius in [3.0f64, 6.0, 12.0, 24.0] {
                let attempt = DetectOptions {
                    search_radius: radius,
                    ..options
                };
                let span = stamp_span(*size * guess_scale, &attempt);
                let attempt_patch = if last_span == Some(span) { None } else { *patch };
                last_span = Some(span);
                if let Some(point) =
                    locate_mark(scan, expected, *size * guess_scale, attempt_patch, *window, attempt, &mut search)
                {
                    found = Some(point);
                    break;
                }
            }
        }
        let point = found.ok_or_else(|| {
            Error::Registration(format!(
                "no corner mark near ({:.0}, {:.0})",
                expected.x, expected.y
            ))
        })?;
        image_points.push(point);
    }

    // Second pass: with four marks the scale and offset are known, so the search
    // is repeated from the fitted positions and the fit is refined.
    let template_points: Vec<Point> = corners.iter().map(|(p, ..)| *p).collect();
    let coarse = Homography::fit(&template_points, &image_points)?;
    let mut refined = Vec::with_capacity(4);
    for (index, (_, size, patch, window)) in corners.iter().enumerate() {
        let predicted = coarse.apply(template_points[index]);
        let point = locate_mark(
            scan,
            predicted,
            *size * coarse_scale(&coarse),
            *patch,
            *window,
            options,
            &mut search,
        )
        .unwrap_or(image_points[index]);
        refined.push(point);
    }

    // `x, y` means the mark centre in some templates and the origin of the search
    // window in others. The box tells the two apart: when its centre coincides
    // with `x, y`, the recorded position is the mark itself and the fitted
    // translation is real; when it does not, `x, y` is only where to look and the
    // translation it implies is an artefact of the search window.
    let fitted = Homography::fit(&template_points, &refined)?;
    let transform = if recorded_position_is_mark_centre(page) {
        fitted
    } else {
        anchor_origin(&fitted, &refined)
    };
    let mark = corners
        .iter()
        .map(|(_, size, ..)| *size)
        .fold(0.0f64, f64::max);
    let registration = finish(transform, template_points, refined)?;
    if registration.is_implausible(mark) {
        return Err(Error::Registration(format!(
            "the four marks do not describe a page of this template: {} px error at scale {:.2}",
            registration.max_residual, registration.scale_x
        )));
    }
    Ok(registration)
}

/// True when the template's corner positions are the marks themselves.
///
/// Only a stated bounding box proves that: without one, the recorded position is
/// where the generator searched for the mark, which sits up and to the left of it.
/// True when every mark of the page declares a bounding box, which is how the
/// generator says "x, y is the mark centre" rather than "x, y is where to
/// look".
fn recorded_position_is_mark_centre(page: &Page) -> bool {
    !page.marks.is_empty() && page.marks.iter().all(|mark| mark.box_declared)
}

/// Removes the translation from a fitted transform.
///
/// The answer blocks are declared in page coordinates that already describe the
/// scan, so only the linear part of the fit may be applied to them; keeping the
/// fitted translation would shift every block by the distance between a corner
/// mark's stamped centre and the origin recorded for it.
/// Removes the translation a search-window origin would inject into the fit.
/// Only needed when the template's `x, y` is a search origin rather than the
/// mark centre (see [`recorded_position_is_mark_centre`]).
fn anchor_origin(fitted: &Homography, _image_points: &[Point]) -> Homography {
    let a = fitted.apply(Point::new(0.0, 0.0));
    let bx = fitted.apply(Point::new(1.0, 0.0));
    let by = fitted.apply(Point::new(0.0, 1.0));
    let dx = Point::new(bx.x - a.x, bx.y - a.y);
    let dy = Point::new(by.x - a.x, by.y - a.y);
    let c = fitted.coefficients();
    Homography::from_coefficients([dx.x, dy.x, 0.0, dx.y, dy.y, 0.0, c[6], c[7], 1.0])
}

/// Assembles the final `Registration`, reporting the rotation, scales and
/// per-corner residuals of the fitted transform.
fn finish(
    transform: Homography,
    template_points: Vec<Point>,
    image_points: Vec<Point>,
) -> Result<Registration> {
    let mut residuals = Vec::with_capacity(template_points.len());
    for (src, dst) in template_points.iter().zip(&image_points) {
        residuals.push(transform.apply(*src).distance(*dst));
    }
    let max_residual = residuals.iter().copied().fold(0.0, f64::max);

    let origin = transform.apply(Point::new(0.0, 0.0));
    let unit_x = transform.apply(Point::new(1.0, 0.0));
    let unit_y = transform.apply(Point::new(0.0, 1.0));
    let dx = Point::new(unit_x.x - origin.x, unit_x.y - origin.y);
    let dy = Point::new(unit_y.x - origin.x, unit_y.y - origin.y);
    let scale_x = (dx.x * dx.x + dx.y * dx.y).sqrt();
    let scale_y = (dy.x * dy.x + dy.y * dy.y).sqrt();
    let rotation_deg = dx.y.atan2(dx.x).to_degrees();

    Ok(Registration {
        transform,
        mode: AlignmentMode::CornerRegistered,
        template_points,
        image_points,
        residuals,
        max_residual,
        rotation_deg,
        scale_x,
        scale_y,
    })
}

/// Mean scale of a fitted transform, used to size the next search window.
/// Geometric mean of the transform's axis scales, which the mark searches use
/// to size their windows in scan pixels.
fn coarse_scale(transform: &Homography) -> f64 {
    let origin = transform.apply(Point::new(0.0, 0.0));
    let unit_x = transform.apply(Point::new(1.0, 0.0));
    let unit_y = transform.apply(Point::new(0.0, 1.0));
    let dx = (unit_x.x - origin.x).hypot(unit_x.y - origin.y);
    let dy = (unit_y.x - origin.x).hypot(unit_y.y - origin.y);
    ((dx + dy) / 2.0).max(1e-6)
}

/// The template point a mark is registered against.
///
/// When the file states a bounding box, its centre is the mark. When it only
/// states a stamp, `x, y` is the stamp's top-left corner and the mark is the
/// stamp's ink centroid relative to that corner; the same offset is then applied
/// to the scan, which keeps the two coordinate systems consistent.
/// The template-space point a mark's detection must be compared against:
/// the box centre when the file states a box, otherwise the recorded origin
/// shifted by the stamp's ink centroid.
fn mark_point(mark: &crate::template::Mark, found: Rect) -> Point {
    if mark.box_declared {
        // A stated box is the mark itself; its centre is the point to register.
        return found.center();
    }
    let Some(patch) = &mark.patch else {
        // No stamp: the recorded position and the detected one carry the same
        // constant offset (the search origin to the ink), so registering the
        // recorded position against the detected centre keeps the fit consistent.
        return mark.origin;
    };
    // A stamp shifts the recorded point by the ink centroid inside it, and the
    // detection reports the stamp's top-left plus the same centroid.
    let mut sum = 0.0f64;
    let (mut cx, mut cy) = (0.0f64, 0.0f64);
    for y in 0..patch.height {
        for x in 0..patch.width {
            let weight = 255.0 - f64::from(patch.get(x, y));
            sum += weight;
            cx += x as f64 * weight;
            cy += y as f64 * weight;
        }
    }
    if sum > 0.0 {
        Point::new(mark.origin.x + cx / sum, mark.origin.y + cy / sum)
    } else {
        found.center()
    }
}

/// Test hook around the private `mark_size`.
pub fn mark_size_of(mark: &crate::template::Mark) -> f64 {
    mark_size(mark)
}

/// The size of a corner mark in template units, preferring the recorded box
/// and falling back to the stamp raster.
fn mark_size(mark: &crate::template::Mark) -> f64 {
    if let Some(found) = mark.found {
        let size = found.width().min(found.height());
        if size > 1.0 {
            return size;
        }
    }
    if let Some(patch) = &mark.patch {
        return patch.width.min(patch.height) as f64;
    }
    24.0
}

/// Finds the darkest solid square near `expected`.
///
/// A corner mark is a filled block of ink, so the window whose neighbourhood is
/// darkest is the mark. Sampling on a coarse grid keeps the cost low; the best
/// window is then nudged by matching the centroid of its ink.
/// Searches for the template stamp around `expected` by normalised correlation.
///
/// The stamp is compared at a few integer scales, because the scan may not be at
/// the scale the template was cut at; the best matching offset wins.
/// Correlates the stamp over a window around the expected position, coarse
/// grid first (quarter steps, parallel over rows) and a fine pass around the
/// winner. The stamp's ink centroid converts the best top-left corner into
/// the mark point.
fn locate_by_patch(
    scan: &Gray,
    expected: Point,
    size: f64,
    patch: &Gray2,
    options: DetectOptions,
) -> Option<Point> {
    // Correlating a stamp over a large window is expensive, so the search stays
    // local and falls back to darkness when nothing matches well enough.
    let radius = (size * options.search_radius.max(1.0)).ceil() as isize;
    // The window is clamped to the image, not to the radius: a stamp at the page
    // corner extends beyond the search circle, and clipping it to the circle would
    // leave no candidate at all.
    let x0 = 0;
    let y0 = 0;
    let x1 = scan.width() as isize - 1;
    let y1 = scan.height() as isize - 1;

    // The template states where the stamp is, so the search is a fine grid around
    // that position, at the stamp's own scale only: scanning several scales over a
    // half-page window costs hundreds of milliseconds per corner and finds nothing
    // that the declared position does not.
    let (pw, ph) = (patch.width, patch.height);
    if pw >= (x1 - x0) as usize || ph >= (y1 - y0) as usize {
        return None;
    }
    let centre_x = expected.x as isize - pw as isize / 2;
    let centre_y = expected.y as isize - ph as isize / 2;
    // A fine grid over a huge window is the whole cost of a failed darkness
    // search; a mark sits within about half a stamp of where the template says.
    let span = (radius / 2).clamp(12, 40);
    // Coarse pass on a quarter grid, then a fine pass around the winner: a full
    // one-pixel grid over this window costs tens of millions of multiply-adds per
    // corner, and the response surface is smooth enough that two stages find it.
    let coarse = 6;
    let stats = StampStats::build(patch);
    let points = (-span..=span)
        .step_by(coarse as usize)
        .flat_map(|dy| (-span..=span).step_by(coarse as usize).map(move |dx| (centre_x + dx, centre_y + dy)));
    let best = correlation_grid(scan, patch, &stats, points);
    let best = if let Some((_, bx, by)) = best {
        let fine = coarse.min(4);
        let points = (-fine..=fine)
            .flat_map(|dy| (-fine..=fine).map(move |dx| (bx + dx, by + dy)));
        correlation_grid(scan, patch, &stats, points).map_or(best, |fine_best| {
            if fine_best.0 > best.unwrap_or((0.0, 0, 0)).0 {
                Some(fine_best)
            } else {
                best
            }
        })
    } else {
        best
    };

    let (score, tx, ty) = best?;
    if score < options.min_patch_score {
        return None;
    }
    // The template point of a stamped mark is the stamp's top-left plus the ink
    // centroid, so the detected point is built the same way.
    let mut stamp_sum = 0.0f64;
    let (mut stamp_x, mut stamp_y) = (0.0f64, 0.0f64);
    for sy in 0..patch.height {
        for sx in 0..patch.width {
            let weight = 255.0 - f64::from(patch.get(sx, sy));
            stamp_sum += weight;
            stamp_x += sx as f64 * weight;
            stamp_y += sy as f64 * weight;
        }
    }
    let (offset_x, offset_y) = if stamp_sum > 0.0 {
        (stamp_x / stamp_sum, stamp_y / stamp_sum)
    } else {
        (patch.width as f64 / 2.0, patch.height as f64 / 2.0)
    };
    Some(Point::new(tx as f64 + offset_x, ty as f64 + offset_y))
}

/// Normalised correlation of the stamp against the scan at one offset.
///
/// The score is 1.0 for a perfect match and can go negative; it is invariant to
/// the overall brightness of the scan, which varies between scanners.
/// Test hook around the private `correlation`.
pub fn correlation_at_pub(scan: &Gray, x: isize, y: isize, patch: &Gray2) -> f64 {
    correlation(scan, x, y, patch)
}

/// The stamp's own sums: `sum_p`, `sum_pp` and the per-row pixel pointer
/// arithmetic never change across a correlation sweep, so they are computed
/// once instead of once per grid point.
/// The stamp's own sums (`sum_p`, `sum_pp`): fixed for a given stamp, so they
/// are computed once for a whole correlation sweep instead of once per point.
struct StampStats {
    sum_p: f64,
    sum_pp: f64,
    n: f64,
}

impl StampStats {
    /// Sums the stamp once; called at the start of a sweep.
    fn build(patch: &Gray2) -> Self {
        let (mut sum_p, mut sum_pp) = (0u64, 0u64);
        for &value in &patch.pixels {
            let v = u64::from(value);
            sum_p += v;
            sum_pp += v * v;
        }
        Self {
            sum_p: sum_p as f64,
            sum_pp: sum_pp as f64,
            n: patch.width as f64 * patch.height as f64,
        }
    }
}

/// Normalised correlation of `patch` at every point of the grid, sharing the
/// stamp sums. Returns the best point and score.
/// Normalised correlation of `patch` at every point of the grid, sharing the
/// stamp sums and scanning rows in order (vectorisable, no clamping). Returns
/// the best point and score.
fn correlation_grid(
    scan: &Gray,
    patch: &Gray2,
    stats: &StampStats,
    points: impl IntoIterator<Item = (isize, isize)>,
) -> Option<(f64, isize, isize)> {
    let (pw, ph) = (patch.width, patch.height);
    let pixels = scan.pixels();
    let sw = scan.width() as isize;
    let sh = scan.height() as isize;
    let stamp = &patch.pixels;
    let mut best: Option<(f64, isize, isize)> = None;
    for (tx, ty) in points {
        if tx < 0 || ty < 0 || tx + pw as isize > sw || ty + ph as isize > sh {
            continue;
        }
        // u32 accumulators: max per-pixel product 65025 and n < 2^16 fit
        // comfortably, and 32-bit lanes vectorize twice as wide as 64-bit.
        let (mut sum_s, mut sum_ss, mut sum_ps) = (0u32, 0u32, 0u32);
        for py in 0..ph {
            let stamp_row = py * pw;
            let scan_row = ((ty + py as isize) * sw) as usize + tx as usize;
            for px in 0..pw {
                let stamp = u32::from(stamp[stamp_row + px]);
                let scan_value = u32::from(pixels[scan_row + px]);
                sum_s += scan_value;
                sum_ss += scan_value * scan_value;
                sum_ps += stamp * scan_value;
            }
        }
        let n = stats.n;
        let (sum_s, sum_ss, sum_ps) = (f64::from(sum_s), f64::from(sum_ss), f64::from(sum_ps));
        let cov = sum_ps - stats.sum_p * sum_s / n;
        let var_p = stats.sum_pp - stats.sum_p * stats.sum_p / n;
        let var_s = sum_ss - sum_s * sum_s / n;
        let denom = (var_p * var_s).sqrt();
        let score = if denom < 1e-6 { 0.0 } else { cov / denom };
        if best.as_ref().map_or(true, |b| score > b.0) {
            best = Some((score, tx, ty));
        }
    }
    best
}

/// Normalised correlation of the stamp against the scan at one offset.
///
/// The score is 1.0 for a perfect match and can go negative; it is invariant
/// to the overall brightness of the scan, which varies between scanners.
fn correlation(scan: &Gray, x: isize, y: isize, patch: &Gray2) -> f64 {
    let n = patch.width as u64 * patch.height as u64;
    if n < 4 {
        return 0.0;
    }
    // Integer accumulators: the stamp is fixed and small, so the products fit
    // easily in u64 and the loop vectorizes without float dependencies.
    let (mut sum_p, mut sum_s, mut sum_pp, mut sum_ss, mut sum_ps) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let patch_pixels = &patch.pixels;
    let pw = patch.width;
    let pixels = scan.pixels();
    let sw = scan.width() as isize;
    let sh = scan.height() as isize;
    if x >= 0 && y >= 0 && x + pw as isize <= sw && y + patch.height as isize <= sh {
        // Fast path: the caller's coarse grid keeps the stamp inside the scan,
        // so no per-pixel clamping is needed.
        for py in 0..patch.height {
            let stamp_row = py * pw;
            let scan_row = ((y + py as isize) * sw) as usize + x as usize;
            for px in 0..pw {
                let stamp = u64::from(patch_pixels[stamp_row + px]);
                let scan_value = u64::from(pixels[scan_row + px]);
                sum_p += stamp;
                sum_s += scan_value;
                sum_pp += stamp * stamp;
                sum_ss += scan_value * scan_value;
                sum_ps += stamp * scan_value;
            }
        }
    } else {
        // Partially outside: keep the clamped-edge sampling.
        for py in 0..patch.height {
            for px in 0..pw {
                let stamp = u64::from(patch_pixels[py * pw + px]);
                let scan_value = u64::from(scan.get_signed(x + px as isize, y + py as isize));
                sum_p += stamp;
                sum_s += scan_value;
                sum_pp += stamp * stamp;
                sum_ss += scan_value * scan_value;
                sum_ps += stamp * scan_value;
            }
        }
    }
    let (sum_p, sum_s, sum_pp, sum_ss, sum_ps, n) =
        (sum_p as f64, sum_s as f64, sum_pp as f64, sum_ss as f64, sum_ps as f64, n as f64);
    let cov = sum_ps - sum_p * sum_s / n;
    let var_p = sum_pp - sum_p * sum_p / n;
    let var_s = sum_ss - sum_s * sum_s / n;
    let denom = (var_p * var_s).sqrt();
    if denom < 1e-6 {
        return 0.0;
    }
    cov / denom
}

/// Width of the square window the stamp search sweeps around the expected
/// position; must match `locate_by_patch` exactly.
/// Width of the square window the stamp search sweeps around the expected
/// position; must match `locate_by_patch` exactly, which is what lets the
/// escalation loop skip the correlation once the span stops growing.
fn stamp_span(size: f64, options: &DetectOptions) -> isize {
    let radius = (size * options.search_radius.max(1.0)).ceil() as isize;
    (radius / 2).clamp(12, 40)
}

/// Locates one mark: darkness sweep first (an order of magnitude cheaper),
/// stamp correlation only when the darkest window is not the mark, both
/// bounded by the plausibility tolerance.
fn locate_mark(
    scan: &Gray,
    expected: Point,
    size: f64,
    patch: Option<&Gray2>,
    window: Option<Rect>,
    options: DetectOptions,
    search: &mut DarknessSearch,
) -> Option<Point> {
    let size = size.max(4.0);

    // A template stamp turns the search into correlation: the mark may be a
    // triangle, a ring or any other shape, and only its own pixels know that.
    // Scoring darkness alone would rank any dark blob, which is why shapes other
    // than squares used to be missed.
    // The darkness search is an order of magnitude cheaper than correlation and
    // settles a square mark immediately; the stamp is only consulted when it does
    // not produce a plausible mark, which is what happens for the triangular and
    // ring-shaped ones.
    // A window-style mark (the JSON `lx=0` marks) may sit anywhere inside its
    // declared box, so the sweep is clipped to the box and a hit is accepted
    // when it lands inside — ink elsewhere (page headers, handwriting) loses
    // no matter how dark.
    let hit = match window {
        Some(bounds) => {
            let dark = darkness_mark_in_bounds(scan, bounds, size, options, search)?;
            if bounds.contains(dark) { Some(dark) } else { None }
        }
        None => darkness_mark(scan, expected, size, options, search),
    };
    if let Some(dark) = hit {
        let tolerance = (size * 1.5).max(20.0);
        if dark.distance(expected) <= tolerance {
            return Some(dark);
        }
        // The darkest window is a different blob of ink; keep going.
    }
    let patch = patch?;
    let hit = locate_by_patch(scan, expected, size, patch, options)?;
    // A stamp that matches best far from where the template says has matched
    // handwriting rather than the mark.
    let tolerance = (size * 1.5).max(20.0);
    if hit.distance(expected) > tolerance * 2.0 {
        return None;
    }
    Some(hit)
}

/// The darkest square window near `expected`, without any shape knowledge.
/// The darkest square window near `expected`, without any shape knowledge.
/// Sweeps a grid at `size/3` steps over the search radius; beyond about one
/// scan worth of cumulative windows an integral image turns each window into
/// four lookups (built once and shared by the sweeps of all corners).
fn darkness_mark(
    scan: &Gray,
    expected: Point,
    size: f64,
    options: DetectOptions,
    search: &mut DarknessSearch,
) -> Option<Point> {
    let size = size.max(4.0);
    let radius = size * options.search_radius.max(1.0);
    let x0 = (expected.x - radius).floor().max(0.0) as usize;
    let y0 = (expected.y - radius).floor().max(0.0) as usize;
    let x1 = ((expected.x + radius).ceil() as usize).min(scan.width().saturating_sub(1));
    let y1 = ((expected.y + radius).ceil() as usize).min(scan.height().saturating_sub(1));
    darkest_window(scan, x0, y0, x1, y1, size, options, search)
        .filter(|(darkness, _, _)| *darkness >= options.min_mark_darkness)
        .map(|(_, cx, cy)| refine_center(scan, cx as isize, cy as isize,
            (size * options.window_ratio.max(0.2) / 2.0).round().max(1.0) as isize, options))
}

/// The darkest sampling window inside the box `(x0,y0)..(x1,y1)`, returning
/// `(darkness, x, y)` of the winner.
fn darkest_window(
    scan: &Gray,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    size: f64,
    options: DetectOptions,
    search: &mut DarknessSearch,
) -> Option<(f64, usize, usize)> {
    let size = size.max(4.0);
    let window = (size * options.window_ratio.max(0.2) / 2.0)
        .round()
        .max(1.0) as isize;

    // A window that leaves the image would be padded with the clamped edge pixel,
    // and on a cropped scan that edge is often the dark page border, which would
    // win over the real mark. Centres are therefore kept fully inside.
    let margin = window as usize + 1;
    let step = (size / 3.0).round().max(2.0) as usize;
    let grid_cols = (x1 - x0).max(1) / step + 1;
    let grid_rows = (y1 - y0).max(1) / step + 1;
    let grid_cost = grid_cols as u64 * grid_rows as u64 * (2 * window + 1).pow(2) as u64;
    let mut best: Option<(f64, usize, usize)> = None;
    let mut y = y0.max(margin);
    while y + margin < scan.height() && y <= y1 {
        let mut x = x0.max(margin);
        while x + margin < scan.width() && x <= x1 {
            let darkness = search.window_mean(scan, x as isize, y as isize, window, grid_cost);
            if best.map_or(true, |b| darkness > b.0) {
                best = Some((darkness, x, y));
            }
            x += step;
        }
        y += step;
    }

    let (darkness, cx, cy) = best?;
    Some((darkness, cx, cy))
}

/// The darkest window inside the declared box of a window-style mark.
fn darkness_mark_in_bounds(
    scan: &Gray,
    bounds: Rect,
    size: f64,
    options: DetectOptions,
    search: &mut DarknessSearch,
) -> Option<Point> {
    let x0 = bounds.x0.floor().max(0.0) as usize;
    let y0 = bounds.y0.floor().max(0.0) as usize;
    let x1 = (bounds.x1.ceil() as usize).min(scan.width().saturating_sub(1));
    let y1 = (bounds.y1.ceil() as usize).min(scan.height().saturating_sub(1));
    let (darkness, cx, cy) = darkest_window(scan, x0, y0, x1, y1, size, options, search)?;
    if darkness < options.min_mark_darkness {
        return None;
    }
    Some(refine_center(
        scan,
        cx as isize,
        cy as isize,
        (size * options.window_ratio.max(0.2) / 2.0).round().max(1.0) as isize,
        options,
    ))
}

/// Centroid of the ink inside the winning window, which lands on the middle of
/// the mark rather than on the grid point.
/// Centroid of the ink inside the winning window, which lands on the middle
/// of the mark rather than on the grid point.
fn refine_center(
    scan: &Gray,
    cx: isize,
    cy: isize,
    window: isize,
    options: DetectOptions,
) -> Point {
    let half = window.max(1);
    let (mut sum, mut count) = (0.0, 0.0);
    let (mut sx, mut sy) = (0.0, 0.0);
    for y in (cy - half)..=(cy + half) {
        for x in (cx - half)..=(cx + half) {
            let weight = 255.0 - f64::from(scan.get_signed(x, y));
            if weight < options.min_mark_darkness * 2.0 {
                continue;
            }
            sum += weight;
            sx += x as f64 * weight;
            sy += y as f64 * weight;
            count += 1.0;
        }
    }
    if sum > 0.0 && count > 0.0 {
        Point::new(sx / sum, sy / sum)
    } else {
        Point::new(cx as f64, cy as f64)
    }
}

/// Test hook: one corner search, so the harness can time it alone.
pub fn locate_mark_for_test(
    scan: &Gray,
    expected: Point,
    size: f64,
    patch: Option<&Gray2>,
    options: DetectOptions,
) -> Option<Point> {
    locate_mark(scan, expected, size, patch, None, options, &mut DarknessSearch::new())
}

/// Integral image of the inverted scan, so the darkness search can price a
/// window with four lookups instead of summing it. Built lazily: small search
/// windows are cheaper summed directly, so the build waits until the sweeps
/// have touched about one scan's worth of pixels in total.
struct DarknessIntegral {
    width: usize,
    data: Vec<u32>,
}

/// Shared state of the corner searches of one `register_by_marks` call.
struct DarknessSearch {
    integral: Option<DarknessIntegral>,
    swept: u64,
}

impl DarknessSearch {
    fn new() -> Self {
        Self {
            integral: None,
            swept: 0,
        }
    }

    /// Mean darkness of one window of the grid, building the integral once the
    /// accumulated sweep cost has paid for it.
    fn window_mean(
        &mut self,
        scan: &Gray,
        x: isize,
        y: isize,
        window: isize,
        grid_cost: u64,
    ) -> f64 {
        if self.integral.is_none() {
            self.swept += grid_cost;
            if self.swept > (scan.width() * scan.height()) as u64 {
                self.integral = Some(DarknessIntegral::build(scan));
            }
        }
        match &self.integral {
            Some(built) => built.window_mean(x, y, window),
            None => mean_darkness(scan, x, y, window),
        }
    }
}

impl DarknessIntegral {
    fn build(scan: &Gray) -> Self {
        let (width, height) = (scan.width(), scan.height());
        let pixels = scan.pixels();
        let mut data = vec![0u32; (width + 1) * (height + 1)];
        for y in 0..height {
            let mut row = 0u32;
            for x in 0..width {
                row += u32::from(255 - pixels[y * width + x]);
                data[(y + 1) * (width + 1) + x + 1] = data[y * (width + 1) + x + 1] + row;
            }
        }
        Self { width, data }
    }

    /// Mean darkness of the square window around the centre; the caller keeps
    /// windows fully inside the scan, so no clamping here either.
    fn window_mean(&self, cx: isize, cy: isize, half: isize) -> f64 {
        let stride = self.width + 1;
        let (xa, ya) = ((cx - half) as usize, (cy - half) as usize);
        let (xb, yb) = ((cx + half + 1) as usize, (cy + half + 1) as usize);
        let sum = self.data[yb * stride + xb] + self.data[ya * stride + xa]
            - self.data[ya * stride + xb]
            - self.data[yb * stride + xa];
        f64::from(sum) / f64::from(((xb - xa) * (yb - ya)) as u32)
    }
}

/// Mean ink of a square window, sampled on a coarse grid.
///
/// A corner mark is a solid block, so a subsample of the window is enough to rank
/// candidates; the winner is then measured exactly by `refine_center`. Summing
/// every pixel here instead of subsampling costs more than the rest of the
/// pipeline together.
fn mean_darkness(scan: &Gray, cx: isize, cy: isize, half: isize) -> f64 {
    let mut sum = 0u32;
    let mut count = 0u32;
    for y in (cy - half)..=(cy + half) {
        for x in (cx - half)..=(cx + half) {
            sum += 255 - u32::from(scan.get_signed(x, y));
            count += 1;
        }
    }
    f64::from(sum) / f64::from(count.max(1))
}

/// Axis aligned bounding box of the points.
fn bounding_box(points: impl Iterator<Item = Point>) -> Rect {
    let mut rect = Rect::new(f64::MAX, f64::MAX, f64::MIN, f64::MIN);
    for p in points {
        rect = Rect::new(
            rect.x0.min(p.x),
            rect.y0.min(p.y),
            rect.x1.max(p.x),
            rect.y1.max(p.y),
        );
    }
    rect
}
