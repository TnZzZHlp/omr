//! End-to-end recognition: register the scan, then read every declared block.

use crate::barcode::{self, BarcodeResult};
use crate::error::{Error, Result};
use crate::fill::{self, CellMeasure, DecideOptions, MeasureOptions, ReviewReason};
use crate::geometry::{Homography, Rect};
use crate::gray::Gray;
use crate::registration::{self, AlignmentMode, DetectOptions, Registration};
use crate::template::{Block, BlockKind, Page, Template};
use serde::Serialize;

/// Tuning knobs for a reading. The defaults work for every template in the
/// test corpus; the sub-structs expose the registration, measurement and
/// decision thresholds separately.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options {
    pub registration: DetectOptions,
    pub measure: MeasureOptions,
    pub decide: DecideOptions,
}

/// One answered or blank question.
///
/// `selected` is `None` for a blank row or an unconvincing read; both cases
/// carry `needs_review`/`reason` so a human can look. `scores` holds the raw
/// per-label coverage for diagnostics and for the annotated output.
#[derive(Clone, Debug, Serialize)]
pub struct Selection {
    pub question: Option<u32>,
    pub name: String,
    /// Label of the strongest cell, `None` when nothing is marked.
    pub selected: Option<String>,
    /// Labels of other cells that also carry ink.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub also_marked: Vec<String>,
    /// Score per label, in declaration order.
    pub scores: Vec<(String, f64)>,
    pub confidence: f64,
    pub needs_review: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ReviewReason>,
}

/// Reading of the page-number column (`PNO`), one box per candidate page.
#[derive(Clone, Debug, Serialize)]
pub struct PageNumber {
    pub selected: Option<String>,
    pub confidence: f64,
    pub needs_review: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ReviewReason>,
}

/// Reading of a multiple-select question, any subset of options.
#[derive(Clone, Debug, Serialize)]
pub struct MultipleSelection {
    pub question: Option<u32>,
    pub name: String,
    /// Options marked, in declaration order. Empty when nothing is marked.
    pub selected: Vec<String>,
    /// Score per label, in declaration order.
    pub scores: Vec<(String, f64)>,
    /// True when at least one option is marked.
    pub answered: bool,
    pub needs_review: bool,
}

/// Reading of the candidate number grid, one digit per column.
#[derive(Clone, Debug, Serialize)]
pub struct CandidateNumber {
    /// Digits read from the columns, in template order.
    pub value: String,
    pub confidence: f64,
    pub needs_review: bool,
    /// Per-column readings, aligned with the digits in `value`.
    pub columns: Vec<Selection>,
}

/// Reading of the absence bracket (`QK`): set when filled in.
#[derive(Clone, Debug, Serialize)]
pub struct AbsentMark {
    pub marked: bool,
    pub confidence: f64,
    pub needs_review: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<ReviewReason>,
}

/// How well the scan aligned to the template, reported with every reading.
#[derive(Clone, Debug, Serialize)]
pub struct RegistrationInfo {
    /// How the scan was aligned to the template.
    pub mode: AlignmentMode,
    /// Largest corner reprojection error in scan pixels; the content offset in
    /// pixels for a content-aligned page.
    pub max_residual: f64,
    pub rotation_deg: f64,
    /// Scan pixels per template unit along each axis.
    pub scale_x: f64,
    pub scale_y: f64,
    /// Corner positions measured in the scan.
    pub corners: Vec<(f64, f64)>,
}

impl From<&Registration> for RegistrationInfo {
    fn from(value: &Registration) -> Self {
        Self {
            mode: value.mode,
            max_residual: value.max_residual,
            rotation_deg: value.rotation_deg,
            scale_x: value.scale_x,
            scale_y: value.scale_y,
            corners: value.image_points.iter().map(|p| (p.x, p.y)).collect(),
        }
    }
}

/// The complete reading of one scanned page.
#[derive(Clone, Debug, Serialize)]
pub struct Recognition {
    /// Template page index, zero based.
    pub page: usize,
    pub template_id: String,
    pub registration: RegistrationInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_number: Option<PageNumber>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub absent: Option<AbsentMark>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub barcode: Option<BarcodeResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidate_number: Option<CandidateNumber>,
    pub selections: Vec<Selection>,
    /// Multiple-select questions, answered by any subset of options.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub multiple: Vec<MultipleSelection>,
    /// Readings that need a human look.
    pub review_count: usize,
    /// Mapping from template coordinates to scan coordinates, used to draw the
    /// readings back onto the scan.
    #[serde(skip)]
    pub transform: Homography,
}

/// Registers `scan` against template page `page_index` and reads it.
///
/// `crop` restricts the work to a sub-rectangle of the scan, which is what a
/// spread holding two pages needs; measurements are relative to the crop.
pub fn register_scan(
    template: &Template,
    page_index: usize,
    scan: &Gray,
    crop: Option<Rect>,
    options: Options,
) -> Result<Recognition> {
    let page = template
        .page(page_index)
        .ok_or_else(|| Error::Template(format!("template has no page {page_index}")))?;
    let work = match crop {
        Some(rect) => scan.sub_image(rect),
        None => scan.clone(),
    };
    let registration = registration::register(page, &work, options.registration)?;
    let mut recognition = read_page(page, &work, &registration.transform, options);
    recognition.registration = RegistrationInfo::from(&registration);
    recognition.registration.mode = registration.mode;
    recognition.template_id = template.id.clone();
    Ok(recognition)
}

/// Reads the page using an already fitted transform, skipping detection.
///
/// The companion of [`register_scan`]: the annotation pipeline and the
/// tests use it to re-measure with a known-good transform, and callers that
/// align sheets themselves (e.g. an external printer pipeline) can inject
/// their own transform.
pub fn read_with_transform(
    template: &Template,
    page_index: usize,
    scan: &Gray,
    transform: Homography,
    options: Options,
) -> Result<Recognition> {
    let page = template
        .page(page_index)
        .ok_or_else(|| Error::Template(format!("template has no page {page_index}")))?;
    let mut recognition = read_page(page, scan, &transform, options);
    recognition.template_id = template.id.clone();
    Ok(recognition)
}

/// Reads with corner-mark evidence assumed.
fn read_page(page: &Page, scan: &Gray, transform: &Homography, options: Options) -> Recognition {
    read_page_with_mode(
        page,
        scan,
        transform,
        options,
        AlignmentMode::CornerRegistered,
    )
}

/// Reads the page, scaling confidences by the alignment mode: content
/// alignment is weaker evidence than four corner marks (a slight rotation or
/// scale is uncompensated), so every confidence carries a 0.7 factor.
fn read_page_with_mode(
    page: &Page,
    scan: &Gray,
    transform: &Homography,
    options: Options,
    mode: AlignmentMode,
) -> Recognition {
    // Content alignment is weaker evidence than four corner marks: the offset
    // comes from printed brackets, which are unambiguous, but a slight rotation or
    // scale is uncompensated. Reflect that in every confidence.
    let weight = match mode {
        AlignmentMode::CornerRegistered => 1.0,
        AlignmentMode::ContentAligned => 0.7,
    };
    // Question blocks that share a template column are measured against that
    // column's baseline: the median of the clean rows removes the printed
    // letter and bracket, which otherwise dominate thick-printed templates and
    // make a blank sheet read as letters being darker than each other. Blocks
    // without enough siblings keep the absolute measurement.
    let question_blocks: Vec<&Block> = page
        .blocks
        .iter()
        .filter(|b| b.kind == BlockKind::Question)
        .collect();
    let measured = measure_questions(scan, transform, &question_blocks, options.measure);
    let mut selections = Vec::new();
    for (block, cells) in question_blocks.iter().zip(&measured) {
        let mut s = selection(block, cells, options.decide);
        s.confidence *= weight;
        selections.push(s);
    }
    selections.sort_by_key(|s| s.question.unwrap_or(u32::MAX));

    let page_number = page
        .blocks
        .iter()
        .find(|b| b.kind == BlockKind::PageNumber)
        .map(|block| {
            let cells = fill::measure_solo_block(scan, transform, block, options.measure);
            let decision = fill::decide(&cells, options.decide);
            PageNumber {
                selected: decision.best.map(|i| block.cells[i].label.clone()),
                confidence: decision.confidence,
                needs_review: decision.needs_review,
                reason: decision.reason,
            }
        });

    let absent = page
        .blocks
        .iter()
        .find(|b| b.kind == BlockKind::Absent)
        .map(|block| read_absent(scan, transform, block, options));

    let barcode = page
        .blocks
        .iter()
        .find(|b| b.kind == BlockKind::Barcode)
        .and_then(|block| barcode::decode_block(scan, transform, block.area).ok());

    let candidate_number = read_candidate_number(page, scan, transform, options);

    let multiple: Vec<MultipleSelection> = page
        .blocks
        .iter()
        .filter(|b| b.kind == BlockKind::Multiple)
        .map(|block| read_multiple(scan, transform, block, options))
        .collect();

    let review_count = selections.iter().filter(|s| s.needs_review).count()
        + multiple.iter().filter(|m| m.needs_review).count()
        + usize::from(page_number.as_ref().is_some_and(|p| p.needs_review))
        + usize::from(absent.as_ref().is_some_and(|a| a.needs_review));

    Recognition {
        page: page.index,
        template_id: String::new(),
        registration: RegistrationInfo {
            mode,
            max_residual: 0.0,
            rotation_deg: 0.0,
            scale_x: 1.0,
            scale_y: 1.0,
            corners: Vec::new(),
        },
        page_number,
        absent,
        barcode,
        candidate_number,
        selections,
        multiple,
        review_count,
        transform: *transform,
    }
}

/// Reads the candidate number.
///
/// The number is printed as one column of ten digit cells per digit, so each
/// column is decided on its own and the digits are joined in template order.
/// Reads the candidate number columns left to right; a column contributes a
/// digit when exactly one cell is marked, otherwise a `?` placeholder keeps
/// the column count stable. The value is only reported when every column
/// resolves.
fn read_candidate_number(
    page: &Page,
    scan: &Gray,
    transform: &Homography,
    options: Options,
) -> Option<CandidateNumber> {
    let mut blocks: Vec<&Block> = page
        .blocks
        .iter()
        .filter(|b| b.kind == BlockKind::CandidateNumber)
        .collect();
    if blocks.is_empty() {
        return None;
    }
    blocks.sort_by(|a, b| {
        a.area
            .x0
            .partial_cmp(&b.area.x0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut value = String::new();
    let mut columns = Vec::with_capacity(blocks.len());
    let mut confidence = 1.0f64;
    for block in blocks {
        let cells = fill::measure_solo_block(scan, transform, block, options.measure);
        let decision = fill::decide(&cells, options.decide);
        let digit = decision
            .best
            .map(|i| block.cells[i].label.clone())
            .unwrap_or_else(|| "-".to_owned());
        value.push_str(&digit);
        confidence = confidence.min(decision.confidence);
        columns.push(selection(block, &cells, options.decide));
    }

    let needs_review = columns.iter().any(|c| c.needs_review);
    Some(CandidateNumber {
        value,
        confidence,
        needs_review,
        columns,
    })
}

/// Reads a multiple-select question.
///
/// Every option is an independent yes/no decision, so each cell is compared with
/// the others in the same row to find what stands out, and all options above the
/// threshold are reported rather than the single strongest one.
/// Reads a multiple-select block: every cell above the marked floor counts,
/// in declaration order.
fn read_multiple(
    scan: &Gray,
    transform: &Homography,
    block: &Block,
    options: Options,
) -> MultipleSelection {
    let cells = fill::measure_solo_block(scan, transform, block, options.measure);
    let best = cells.iter().map(|c| c.solid).fold(0.0f64, f64::max);
    let floor = (best * options.decide.rival_ratio).max(options.decide.min_score);

    let selected: Vec<String> = block
        .cells
        .iter()
        .zip(&cells)
        .filter(|(_, measure)| measure.solid >= floor)
        .map(|(cell, _)| cell.label.clone())
        .collect();

    let readable = !selected.is_empty() && best >= options.decide.confident_score;
    MultipleSelection {
        question: block.question,
        name: block.name.clone(),
        scores: block
            .cells
            .iter()
            .zip(&cells)
            .map(|(cell, measure)| (cell.label.clone(), measure.solid))
            .collect(),
        answered: !selected.is_empty(),
        needs_review: !readable,
        selected,
    }
}

/// Reads the absence bracket.
///
/// The bracket is printed on every sheet, so its ink is always present; only a
/// filled bracket darkens the whole cell. The test is therefore on the level of
/// the cell as a whole, not on the presence of any ink.
/// Reads the absence bracket: a printed shape with no siblings, so the cell
/// is judged by how dark its interior is.
fn read_absent(scan: &Gray, transform: &Homography, block: &Block, options: Options) -> AbsentMark {
    let cells = fill::measure_solo_block(scan, transform, block, options.measure);
    let level = cells.iter().map(|c| c.mean_level).fold(255.0f64, f64::min);
    // A filled box drops well below paper; the printed bracket alone does not.
    let marked = level < options.measure.ink_margin * 3.0;
    let confidence = ((options.measure.ink_margin * 3.0 - level).abs()
        / (options.measure.ink_margin * 3.0))
        .clamp(0.0, 1.0);
    AbsentMark {
        marked,
        confidence,
        needs_review: (level - options.measure.ink_margin * 3.0).abs() < 25.0,
        reason: None,
    }
}

/// Measures question blocks, subtracting a column baseline when several
/// blocks share the same template column (cells at nearly the same x).
///
/// The baseline is the median of the clean rows in that column: it captures
/// exactly the printed letter and bracket, which are identical on every row,
/// and leaves only the pencil mark. Thick-printed templates score ~0.2 solid
/// on a blank sheet without this subtraction, which makes an empty cell beat
/// `min_score` and reads letters instead of marks. Groups smaller than three
/// siblings cannot build a baseline and fall back to the absolute measure.
/// Measures question blocks: blocks sharing a template column are measured
/// against that column's baseline (the printed letter and bracket are
/// subtracted), the rest are measured absolutely.
fn measure_questions(
    scan: &Gray,
    transform: &Homography,
    blocks: &[&Block],
    options: crate::fill::MeasureOptions,
) -> Vec<Vec<crate::fill::CellMeasure>> {
    // Group by column: blocks whose first-row cell edges all sit within
    // `TOLERANCE` px of the group's representative form one group. Template
    // generators jitter edges by a pixel between rows, so exact signatures
    // would split one printed column into fragments.
    const TOLERANCE: f64 = 3.0;

    let mut order: Vec<usize> = (0..blocks.len()).collect();
    order.sort_by_key(|&i| (blocks[i].cells.first().map_or(0.0, |c| c.rect.x0) * 10.0) as i64);

    let mut groups: Vec<Vec<usize>> = Vec::new();
    for index in order {
        let block = blocks[index];
        let edges: Vec<f64> = block.cells.iter().map(|c| c.rect.x0).collect();
        let fits = |group: &Vec<usize>| -> bool {
            let representative = blocks[group[0]];
            let rep: Vec<f64> = representative.cells.iter().map(|c| c.rect.x0).collect();
            rep.len() == edges.len()
                && rep
                    .iter()
                    .zip(&edges)
                    .all(|(a, b)| (a - b).abs() <= TOLERANCE)
        };
        match groups.iter_mut().rev().find(|group| fits(group)) {
            Some(group) => group.push(index),
            None => groups.push(vec![index]),
        }
    }

    let mut measured: Vec<Option<Vec<crate::fill::CellMeasure>>> =
        vec![None; blocks.len()];
    for members in &groups {
        let siblings: Vec<&Block> = members.iter().map(|&i| blocks[i]).collect();
        if siblings.len() >= 3 {
            for (&index, cells) in members
                .iter()
                .zip(fill::measure_group(scan, transform, &siblings, options))
            {
                measured[index] = Some(cells);
            }
        } else {
            for (&index, cells) in members.iter().zip(
                siblings
                    .iter()
                    .map(|block| fill::measure_solo_block(scan, transform, block, options)),
            ) {
                measured[index] = Some(cells);
            }
        }
    }
    measured.into_iter().map(|cells| cells.unwrap_or_default()).collect()
}

/// Turns measured cells into a `Selection` for one question block.
fn selection(block: &Block, cells: &[CellMeasure], options: DecideOptions) -> Selection {
    let decision = fill::decide(cells, options);
    Selection {
        question: block.question,
        name: block.name.clone(),
        selected: decision.best.map(|i| block.cells[i].label.clone()),
        also_marked: decision
            .alternates
            .iter()
            .filter_map(|i| block.cells.get(*i).map(|c| c.label.clone()))
            .collect(),
        scores: block
            .cells
            .iter()
            .zip(cells)
            .map(|(cell, measure)| (cell.label.clone(), measure.solid))
            .collect(),
        confidence: decision.confidence,
        needs_review: decision.needs_review,
        reason: decision.reason,
    }
}
