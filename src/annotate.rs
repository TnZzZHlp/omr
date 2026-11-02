//! Draws a recognition result back onto the scan it came from.
//!
//! The template declares every block in its own coordinate space, so a reading
//! is turned into a picture by mapping the template rectangles through the
//! registration transform and overlaying them: accepted cells in green, cells
//! that need review in amber, rival marks in red, and the barcode and page
//! number as labels.

use crate::draw::{self, badge};
use crate::geometry::{Homography, Point, Rect};
use crate::gray::Gray;
use crate::recognize::{Recognition, Selection};
use crate::template::{BlockKind, Page, Template};
use image::RgbImage;

/// How the annotation should look.
#[derive(Clone, Copy, Debug)]
pub struct Style {
    /// Scale of the labels and the thickness of the outlines.
    pub scale: u32,
    /// Draw a light outline around every measured cell.
    pub show_all_cells: bool,
    /// Draw the fitted page corners and the quadrilateral joining them.
    pub show_registration: bool,
}

impl Default for Style {
    fn default() -> Self {
        Self {
            scale: 2,
            show_all_cells: true,
            show_registration: true,
        }
    }
}

/// Renders `scan` with the readings of `recognition` drawn over it.
///
/// The image must be the scan that was recognised, cropped the same way, because
/// the transform maps template coordinates into that image.
pub fn annotate(
    template: &Template,
    page_index: usize,
    recognition: &Recognition,
    scan: &RgbImage,
    style: Style,
) -> RgbImage {
    let mut image = scan.clone();
    let Some(page) = template.page(page_index) else {
        return image;
    };
    let transform = &recognition.transform;
    let scale = style.scale.max(1);
    let thickness = (scale / 2).max(1);

    if style.show_registration {
        draw_registration(&mut image, recognition, transform, thickness, scale);
    }

    if style.show_all_cells {
        for block in page.blocks.iter().filter(|b| b.kind == BlockKind::Question) {
            for cell in &block.cells {
                draw::quad(
                    &mut image,
                    quad_of(transform, cell.rect),
                    thickness,
                    draw::OUTLINE,
                );
            }
        }
    }

    for block in &page.blocks {
        match block.kind {
            BlockKind::Question => {
                if let Some(selection) =
                    recognition.selections.iter().find(|s| s.name == block.name)
                {
                    draw_question(
                        &mut image, transform, page, block, selection, thickness, scale,
                    );
                }
            }
            BlockKind::PageNumber => {
                if let Some(page_number) = &recognition.page_number {
                    let accent = if page_number.needs_review {
                        draw::REVIEW
                    } else {
                        draw::OK
                    };
                    if let Some(index) = block
                        .cells
                        .iter()
                        .position(|c| Some(&c.label) == page_number.selected.as_ref())
                    {
                        draw::quad(
                            &mut image,
                            quad_of(transform, block.cells[index].rect),
                            thickness + 1,
                            accent,
                        );
                    }
                    let label = match &page_number.selected {
                        Some(value) => format!("PAGE {value}"),
                        None => "PAGE ?".to_owned(),
                    };
                    // Above the boxes, so the printed page numbers stay visible.
                    place_above(&mut image, transform, block.area, &label, scale, accent);
                }
            }
            BlockKind::Absent => {
                if let Some(absent) = &recognition.absent {
                    let accent = if absent.needs_review {
                        draw::REVIEW
                    } else if absent.marked {
                        draw::CONFLICT
                    } else {
                        draw::OK
                    };
                    for cell in &block.cells {
                        draw::quad(
                            &mut image,
                            quad_of(transform, cell.rect),
                            thickness + 1,
                            if absent.marked { accent } else { draw::OUTLINE },
                        );
                    }
                    let label = if absent.marked { "ABSENT" } else { "PRESENT" };
                    place_above(&mut image, transform, block.area, label, scale, accent);
                }
            }
            BlockKind::Barcode => {
                if let Some(barcode) = &recognition.barcode {
                    draw::quad(
                        &mut image,
                        quad_of(transform, block.area),
                        thickness,
                        draw::BARCODE,
                    );
                    let label = if barcode.check_ok {
                        format!("CAND {}", barcode.text)
                    } else {
                        "CAND ?".to_owned()
                    };
                    place_above(
                        &mut image,
                        transform,
                        block.area,
                        &label,
                        scale,
                        draw::BARCODE,
                    );
                }
            }
            BlockKind::Multiple => {
                if let Some(reading) = recognition.multiple.iter().find(|m| m.name == block.name) {
                    let accent = if reading.needs_review {
                        draw::REVIEW
                    } else {
                        draw::OK
                    };
                    for label in &reading.selected {
                        if let Some(cell) = block.cells.iter().find(|c| &c.label == label) {
                            draw::quad(
                                &mut image,
                                quad_of(transform, cell.rect),
                                thickness + 1,
                                accent,
                            );
                        }
                    }
                    let text = if reading.selected.is_empty() {
                        "-".to_owned()
                    } else {
                        reading.selected.join("")
                    };
                    place_in_gap(&mut image, transform, page, block, &text, scale, accent);
                }
            }
            BlockKind::CandidateNumber => {
                if let Some(candidate) = &recognition.candidate_number {
                    if let Some(column) = candidate.columns.iter().find(|c| c.name == block.name) {
                        let accent = if column.needs_review {
                            draw::REVIEW
                        } else {
                            draw::OK
                        };
                        if let Some(selected) = &column.selected {
                            if let Some(cell) = block.cells.iter().find(|c| &c.label == selected) {
                                draw::quad(
                                    &mut image,
                                    quad_of(transform, cell.rect),
                                    thickness + 1,
                                    accent,
                                );
                            }
                        }
                        let label = column.selected.clone().unwrap_or_else(|| "-".to_owned());
                        place_above(&mut image, transform, block.area, &label, scale, accent);
                    }
                }
            }
            BlockKind::Other(_) => {}
        }
    }

    image
}

fn draw_registration(
    image: &mut RgbImage,
    recognition: &Recognition,
    _transform: &Homography,
    thickness: u32,
    scale: u32,
) {
    let corners = &recognition.registration.corners;
    if corners.len() < 4 {
        return;
    }
    let points: Vec<(f64, f64)> = corners.iter().map(|(x, y)| (*x, *y)).collect();
    for point in &points {
        draw::cross(image, *point, 9 * scale, thickness, draw::REGISTRATION);
    }
    for i in 0..4 {
        draw::line(
            image,
            points[i],
            points[(i + 1) % 4],
            thickness,
            draw::REGISTRATION,
        );
    }

    // The summary goes in the top right corner of the image, away from the
    // candidate details printed near the registration marks.
    let mut summary = format!(
        "{:.1} PX  {:.2} DEG",
        recognition.registration.max_residual, recognition.registration.rotation_deg
    );
    if let Some(candidate) = &recognition.candidate_number {
        summary = format!("NO {}  {summary}", candidate.value);
    }
    if let Some(barcode) = &recognition.barcode {
        summary = format!("NO {}  {summary}", barcode.text);
    }
    if let Some(number) = &recognition.page_number {
        if let Some(value) = &number.selected {
            summary = format!("PAGE {value}  {summary}");
        }
    }
    let width = draw::text_width(&summary, scale) + 8 * scale;
    let x = i64::from(image.width()) - i64::from(width) - i64::from(4 * scale);
    let y = i64::from(4 * scale);
    badge(
        image,
        x.max(0),
        y.max(0),
        &summary,
        scale,
        draw::REGISTRATION,
    );
}

fn draw_question(
    image: &mut RgbImage,
    transform: &Homography,
    page: &Page,
    block: &crate::template::Block,
    selection: &Selection,
    thickness: u32,
    scale: u32,
) {
    // A light reading is drawn in the review colour even when the score was
    // decisive, so a faint pencil mark is visible at a glance.
    let accent = if selection.needs_review || selection.confidence < 0.8 {
        draw::REVIEW
    } else {
        draw::OK
    };

    for rival in &selection.also_marked {
        if let Some(cell) = block.cells.iter().find(|c| &c.label == rival) {
            draw::quad(
                image,
                quad_of(transform, cell.rect),
                thickness + 1,
                draw::CONFLICT,
            );
        }
    }

    if let Some(selected) = &selection.selected {
        if let Some(cell) = block.cells.iter().find(|c| &c.label == selected) {
            draw::quad(image, quad_of(transform, cell.rect), thickness + 2, accent);
        }
    }

    // One letter keeps the label inside the gap before the next column; the
    // colour of the outline carries the review state.
    let label = selection.selected.clone().unwrap_or_else(|| "-".to_owned());
    place_in_gap(image, transform, page, block, &label, scale, accent);
}

/// Places a label above a block, centred on it.
fn place_above(
    image: &mut RgbImage,
    transform: &Homography,
    area: Rect,
    label: &str,
    scale: u32,
    accent: image::Rgb<u8>,
) {
    let top_left = transform.apply(Point::new(area.x0, area.y0));
    let top_right = transform.apply(Point::new(area.x1, area.y0));
    let width = draw::text_width(label, scale) as f64 + 12.0 * f64::from(scale);
    let centre = (top_left.x + top_right.x) / 2.0;
    let x = (centre - width / 2.0).max(0.0);
    let y =
        (top_left.y.min(top_right.y) - f64::from(draw::text_height(scale) + 4 * scale)).max(0.0);
    badge(image, x as i64, y as i64, label, scale, accent);
}

/// Places a label in the free space to the right of a block, or above it when
/// the next column leaves no room.
fn place_in_gap(
    image: &mut RgbImage,
    transform: &Homography,
    page: &Page,
    block: &crate::template::Block,
    label: &str,
    scale: u32,
    accent: image::Rgb<u8>,
) {
    let gap = right_gap(page, block);
    // Width of the badge: text, padding on both sides, and the accent bar.
    let badge_width = f64::from(draw::text_width(label, scale) + 12 * scale);
    let scale_x = transform
        .apply(Point::new(block.area.x1 + 1.0, block.area.y0))
        .x
        - transform.apply(Point::new(block.area.x1, block.area.y0)).x;

    if gap * scale_x >= badge_width {
        let right_top = transform.apply(Point::new(block.area.x1, block.area.y0));
        let x = right_top.x + 2.0 * f64::from(scale);
        let y = right_top.y - f64::from(draw::text_height(scale)) / 2.0;
        badge(
            image,
            x.max(0.0) as i64,
            y.max(0.0) as i64,
            label,
            scale,
            accent,
        );
    } else {
        place_above(image, transform, block.area, label, scale, accent);
    }
}

/// Free template width between a block and the next block to its right that
/// overlaps it vertically.
fn right_gap(page: &Page, block: &crate::template::Block) -> f64 {
    let mut gap = f64::MAX;
    for other in &page.blocks {
        if std::ptr::eq(other, block) {
            continue;
        }
        let vertical = other.area.y0 < block.area.y1 && other.area.y1 > block.area.y0;
        if vertical && other.area.x0 >= block.area.x1 {
            gap = gap.min(other.area.x0 - block.area.x1);
        }
    }
    if gap.is_finite() {
        gap
    } else {
        200.0
    }
}

fn quad_of(transform: &Homography, rect: Rect) -> [(f64, f64); 4] {
    let corners = [
        Point::new(rect.x0, rect.y0),
        Point::new(rect.x1, rect.y0),
        Point::new(rect.x1, rect.y1),
        Point::new(rect.x0, rect.y1),
    ];
    let mapped: Vec<(f64, f64)> = corners
        .iter()
        .map(|p| {
            let q = transform.apply(*p);
            (q.x, q.y)
        })
        .collect();
    [mapped[0], mapped[1], mapped[2], mapped[3]]
}

/// Saves an annotated scan as a lossless PNG at its original size.
///
/// The image exists to be inspected, so it keeps every pixel of the scan: no
/// scaling, no lossy codec, no palette reduction. The marker colours are flat and
/// survive any compression, and the scan itself is already compressed as JPEG in
/// the input, so the file stays reasonably small.
pub fn save_annotated(image: &RgbImage, path: &std::path::Path) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, image.width(), image.height());
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    // Compression level only changes size and encode time, never pixels: PNG
    // is lossless at every level. High spends ~170 ms per sheet on a 2 MP scan
    // to save ~45% of the file; Fast spends ~8 ms, which keeps the annotated
    // output from dominating the pipeline.
    encoder.set_compression(png::Compression::Fast);
    let mut writer = encoder.write_header()?;
    writer.write_image_data(image.as_raw())?;
    writer.finish()?;
    Ok(())
}

/// Fingerprint of what the annotation added to a scan.
///
/// The digest covers every pixel the drawing changed, so a regression in the
/// geometry or the labels fails a test without the test having to store a
/// megabyte image. It is exact: the palette colours are never equal to a
/// grayscale source pixel, and nothing is antialiased, so no threshold is needed.
pub fn overlay_digest(annotated: &RgbImage, scan: &Gray) -> String {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    let width = annotated.width() as usize;
    let mut hash = OFFSET;
    let mut changed = 0usize;
    for (index, pixel) in annotated.pixels().enumerate() {
        let x = index % width;
        let y = index / width;
        let source = scan.get(x, y);
        let [r, g, b] = pixel.0;
        if r == source && g == source && b == source {
            continue;
        }
        changed += 1;
        for value in (index as u64).to_le_bytes().into_iter().chain([r, g, b]) {
            hash ^= u64::from(value);
            hash = hash.wrapping_mul(PRIME);
        }
    }
    format!("{changed:08x}:{hash:016x}")
}
