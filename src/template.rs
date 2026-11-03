//! Parser for the `.mb` answer-sheet template.
//!
//! The document is XML declared as GB2312. Attribute values are therefore
//! transcoded through GBK before being handed to the XML reader; `img`
//! payloads are base64 rasters of the registration marks.

use crate::error::{Error, Result};
use crate::geometry::{Point, Rect};
use base64::Engine;
use std::path::Path;

/// Declared answer block types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    /// Page number column (`PNO`), one box per candidate page.
    PageNumber,
    /// Candidate number. Either a grid of digit cells, one column per digit and
    /// ten rows per column, or a barcode, told apart by `fillType`.
    CandidateNumber,
    /// Barcode region, `fillType` 7 or 5.
    Barcode,
    /// Absence mark, a single bracket filled when the candidate is missing (`QK`).
    Absent,
    /// Scored multiple-choice question (`S#n`), exactly one option.
    Question,
    /// Multiple-select question (`M#n`), any subset of options.
    Multiple,
    /// Any type this crate does not model, with the raw numeric tag.
    Other(u32),
}

impl BlockKind {
    /// Maps `omrType` to a kind.
    ///
    /// `omrType` 2 covers both shapes of candidate number block; only `fillType`
    /// tells a printed barcode from a grid of digit cells.
    fn from_tags(omr_type: &str, fill_type: &str) -> Self {
        match omr_type.trim() {
            "1" => BlockKind::PageNumber,
            // `fillType` 7 and 5 both mark a printed barcode rather than a grid.
            "2" if matches!(fill_type.trim(), "7" | "5") => BlockKind::Barcode,
            "2" => BlockKind::CandidateNumber,
            "5" => BlockKind::Question,
            "6" => BlockKind::Multiple,
            "7" => BlockKind::Absent,
            other => BlockKind::Other(other.parse().unwrap_or(0)),
        }
    }
}

/// One cell of an answer block in page coordinates.
#[derive(Clone, Debug)]
pub struct Cell {
    /// Column index inside the block, left to right.
    pub column: usize,
    /// Row index inside the block, top to bottom.
    pub row: usize,
    /// Human readable option label taken from `omrmap`.
    pub label: String,
    /// Rectangle in template page coordinates; `x1`/`y1` are exclusive.
    pub rect: Rect,
}

/// An answer block (`<omr>`).
#[derive(Clone, Debug)]
pub struct Block {
    /// Group name, for example `S#1` (scored question 1) or `KSH` (candidate
    /// number).
    pub name: String,
    /// The block family, derived from `omrType` and `fillType`.
    pub kind: BlockKind,
    /// Question number from `stno` when present.
    pub question: Option<u32>,
    /// Answer key declared by the template. Never treated as a result.
    pub answer_key: Option<String>,
    /// Points available for the question.
    pub score: Option<f64>,
    /// Block rectangle in template page coordinates.
    pub area: Rect,
    /// Candidate cells, ordered row-major.
    pub cells: Vec<Cell>,
    /// Raw numeric tag, kept for diagnostics.
    pub raw_type: String,
}

impl Block {
    /// The cell at `(row, column)`, if the block declares it.
    pub fn cell(&self, row: usize, column: usize) -> Option<&Cell> {
        self.cells
            .iter()
            .find(|c| c.row == row && c.column == column)
    }

    /// Number of distinct columns, from the highest declared column index.
    pub fn columns(&self) -> usize {
        self.cells
            .iter()
            .map(|c| c.column)
            .max()
            .map_or(0, |m| m + 1)
    }

    /// Number of distinct rows, from the highest declared row index.
    pub fn rows(&self) -> usize {
        self.cells.iter().map(|c| c.row).max().map_or(0, |m| m + 1)
    }
}

/// A registration mark (`<mli>`).
#[derive(Clone, Debug)]
pub struct Mark {
    /// Ordinal among the page's marks, top-left to bottom-right.
    pub index: u32,
    pub name: String,
    /// Position of the mark on the page. Some generators write the search window
    /// origin here, others the mark centre itself; both land on the mark.
    pub origin: Point,
    /// Bounding box reported by the template generator, if present. Older files
    /// leave it empty and only carry `origin`.
    pub found: Option<Rect>,
    /// True when the file itself states a bounding box, as opposed to `found`
    /// being inferred from the stamp size. Only a stated box means the recorded
    /// position is the mark centre.
    pub box_declared: bool,
    /// Template raster of the mark, row-major, one byte per pixel.
    pub patch: Option<Gray2>,
}

/// Small owned grayscale raster used for mark templates.
#[derive(Clone, Debug)]
pub struct Gray2 {
    pub width: usize,
    pub height: usize,
    /// Row-major luma bytes, 0 = black, 255 = paper.
    pub pixels: Vec<u8>,
}

impl Gray2 {
    /// Stamp pixel at `(x, y)`; reads outside the raster return 255 (paper),
    /// which lets the correlation sweep probe edge-adjacent positions without
    /// bounds checks.
    pub fn get(&self, x: usize, y: usize) -> u8 {
        if x >= self.width || y >= self.height {
            return 255;
        }
        self.pixels[y * self.width + x]
    }
}

/// One template page, corresponding to one scanned side.
#[derive(Clone, Debug)]
pub struct Page {
    /// Zero-based page ordinal within the template.
    pub index: usize,
    /// Registration marks (`mli`), four per side on every template seen.
    pub marks: Vec<Mark>,
    /// Answer blocks (`omr`), in declaration order.
    pub blocks: Vec<Block>,
    /// Row order for barcode decoding: `direct` of the barcode block.
    pub barcode_direct: u32,
}

impl Page {
    /// Union of the mark rectangles, i.e. the region the template was laid out on.
    pub fn extent(&self) -> Option<Rect> {
        self.marks
            .iter()
            .filter_map(|m| m.found)
            .reduce(|a, b| a.union(b))
            .or_else(|| self.blocks.iter().map(|b| b.area).reduce(|a, b| a.union(b)))
    }
}

/// A parsed `.mb` template: the identity of the sheet and, per scanned side,
/// the registration marks and answer blocks in page coordinates.
#[derive(Clone, Debug)]
pub struct Template {
    /// `ver` attribute of the root element.
    pub version: String,
    /// `mbid` attribute; the stable identity of this sheet layout.
    pub id: String,
    /// `dpi` attribute as declared by the generator.
    pub dpi: f64,
    /// One entry per scanned side, in declaration order.
    pub pages: Vec<Page>,
}

impl Template {
    /// Reads and parses a `.mb` file from disk.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::parse(&bytes)
    }

    /// Parses the GBK-encoded XML document. The reader tolerates the
    /// generator's quirks: multi-line tags, very long base64 attributes and
    /// non-standard trailing bits in the base64 payloads.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let text = decode_xml(bytes)?;
        let mut reader = quick_xml::Reader::from_str(&text);
        reader.config_mut().check_end_names = false;
        // The generator writes multi-line tags and very long base64 attributes;
        // without expansion they come through truncated.
        reader.config_mut().expand_empty_elements = true;

        let mut template = Template {
            version: String::new(),
            id: String::new(),
            dpi: 0.0,
            pages: Vec::new(),
        };

        loop {
            let event = reader
                .read_event()
                .map_err(|e| Error::Template(format!("xml: {e}")))?;
            match event {
                quick_xml::events::Event::Start(ref start)
                | quick_xml::events::Event::Empty(ref start) => {
                    let name = local_name(start.name().into_inner());
                    let attrs = attributes(start)?;
                    match name.as_str() {
                        "mb" => {
                            template.version = attrs.first("ver");
                            template.id = attrs.first("mbid");
                            template.dpi = attrs.number("dpi").unwrap_or(0.0);
                        }
                        "page" => {
                            let index = template.pages.len();
                            template.pages.push(Page {
                                index,
                                marks: Vec::new(),
                                blocks: Vec::new(),
                                barcode_direct: 1,
                            });
                        }
                        "mli" => {
                            if let Some(page) = template.pages.last_mut() {
                                page.marks.push(parse_mark(&attrs));
                            }
                        }
                        "omr" => {
                            if let Some(page) = template.pages.last_mut() {
                                let block = parse_block(&attrs)?;
                                if block.kind == BlockKind::Barcode {
                                    page.barcode_direct =
                                        attrs.number("direct").unwrap_or(1.0) as u32;
                                }
                                page.blocks.push(block);
                            }
                        }
                        _ => {}
                    }
                }
                quick_xml::events::Event::Eof => break,
                _ => {}
            }
        }

        if template.pages.is_empty() {
            return Err(Error::Template("no <page> element found".into()));
        }
        Ok(template)
    }

    /// The page with the given zero-based index, if declared.
    pub fn page(&self, index: usize) -> Option<&Page> {
        self.pages.get(index)
    }
}

/// Builds a `Mark` from the `mli` attributes. The declared box is either a
/// real bounding box (`x1..y2` with positive extent) or, on older files, a
/// synthetic box of the stamp's own size centred on the recorded position.
/// `box_declared` records which happened: only a real box means `x, y` is the
/// mark centre rather than a search-window origin.
fn parse_mark(attrs: &AttrMap) -> Mark {
    let patch_w = attrs.number("width").unwrap_or(0.0);
    let patch_h = attrs.number("height").unwrap_or(0.0);
    let found = match (
        attrs.number("x1"),
        attrs.number("y1"),
        attrs.number("x2"),
        attrs.number("y2"),
    ) {
        // A real bounding box has a positive extent; zeroes mean the generator
        // did not record one.
        (Some(x1), Some(y1), Some(x2), Some(y2))
            if (x2 - x1).abs() > 1.0 && (y2 - y1).abs() > 1.0 =>
        {
            Some(Rect::from_corners(Point::new(x1, y1), Point::new(x2, y2)))
        }
        // Fall back to a box of the mark's own size centred on `x, y`.
        _ if patch_w > 0.0 && patch_h > 0.0 => {
            let half_w = patch_w / 2.0;
            let half_h = patch_h / 2.0;
            Some(Rect::new(
                attrs.number("x").unwrap_or(0.0) - half_w,
                attrs.number("y").unwrap_or(0.0) - half_h,
                attrs.number("x").unwrap_or(0.0) + half_w,
                attrs.number("y").unwrap_or(0.0) + half_h,
            ))
        }
        _ => None,
    };

    let patch = attrs.opt("img").and_then(|raw| {
        let width = attrs.number("width")? as usize;
        let height = attrs.number("height")? as usize;
        let decoded = base64::engine::general_purpose::STANDARD
            // Generators leave non-canonical trailing bits in the stamp data; the
            // strict engine would reject the whole raster over one character.
            .decode(raw.trim().as_bytes())
            .or_else(|_| {
                base64::engine::general_purpose::GeneralPurpose::new(
                    &base64::alphabet::STANDARD,
                    base64::engine::GeneralPurposeConfig::new()
                        .with_decode_allow_trailing_bits(true),
                )
                .decode(raw.trim().as_bytes())
            })
            .ok()?;
        if decoded.len() != width * height {
            return None;
        }
        Some(Gray2 {
            width,
            height,
            pixels: decoded,
        })
    });

    let box_declared = match (
        attrs.number("x1"),
        attrs.number("y1"),
        attrs.number("x2"),
        attrs.number("y2"),
    ) {
        (Some(x1), Some(y1), Some(x2), Some(y2)) => (x2 - x1).abs() > 1.0 && (y2 - y1).abs() > 1.0,
        _ => false,
    };

    Mark {
        index: attrs.number("idx").unwrap_or(0.0) as u32,
        name: attrs.first("name"),
        origin: Point::new(
            attrs.number("x").unwrap_or(0.0),
            attrs.number("y").unwrap_or(0.0),
        ),
        found,
        box_declared,
        patch,
    }
}

/// Builds a `Block` from the `omr` attributes: the declared area, the kind
/// derived from `omrType`/`fillType`, the question metadata, and the cells
/// laid out from the `ax`/`ay` edge lists (`onum` columns, `inum` rows).
fn parse_block(attrs: &AttrMap) -> Result<Block> {
    let raw_type = attrs.first("omrType");
    let kind = BlockKind::from_tags(&raw_type, &attrs.first("fillType"));
    let origin_x = attrs.number("x").unwrap_or(0.0);
    let origin_y = attrs.number("y").unwrap_or(0.0);
    let width = attrs.number("width").unwrap_or(0.0);
    let height = attrs.number("height").unwrap_or(0.0);
    let area = Rect::new(origin_x, origin_y, origin_x + width, origin_y + height);

    let ax = attrs.numbers("ax");
    let ay = attrs.numbers("ay");
    let mut onum = attrs.number("onum").unwrap_or(0.0) as usize;
    let mut inum = attrs.number("inum").unwrap_or(1.0).max(1.0) as usize;
    if onum == 0 {
        onum = ax.len() / 2;
    }
    if inum == 0 {
        inum = ay.len() / 2;
    }

    let labels = attrs.labels("omrmap");
    let mut cells = Vec::new();
    for row in 0..inum {
        for column in 0..onum {
            let (Some(&x0), Some(&x1)) = (ax.get(2 * column), ax.get(2 * column + 1)) else {
                continue;
            };
            let (Some(&y0), Some(&y1)) = (ay.get(2 * row), ay.get(2 * row + 1)) else {
                continue;
            };
            let label = labels
                .get(row * onum + column)
                .cloned()
                .unwrap_or_else(|| format!("{column}"));
            cells.push(Cell {
                column,
                row,
                label,
                rect: Rect::new(origin_x + x0, origin_y + y0, origin_x + x1, origin_y + y1),
            });
        }
    }

    Ok(Block {
        name: attrs.first("ogname"),
        kind,
        question: attrs.number("stno").map(|v| v as u32),
        answer_key: attrs.opt("answer"),
        score: attrs.number("score"),
        area,
        cells,
        raw_type,
    })
}

/// Strips a namespace prefix, if any.
fn local_name(raw: &str) -> String {
    raw.rsplit(':').next().unwrap_or(raw).to_owned()
}

/// Collects the element's attributes into a lookup map, GBK-decoded.
fn attributes(start: &quick_xml::events::BytesStart<'_>) -> Result<AttrMap> {
    // A generator lays long attributes out over several lines, so the tag content
    // is normalised to single spaces first: `quick-xml` stops at the first line
    // break inside an attribute value otherwise.

    let mut map = AttrMap::default();
    for attr in start.attributes().with_checks(false) {
        let attr = attr.map_err(|e| Error::Template(format!("attribute: {e}")))?;
        let key = attr.key.as_ref().to_owned();
        let value = attr
            .normalized_value(quick_xml::XmlVersion::Implicit1_0)
            .map_err(|e| Error::Template(format!("value of {key}: {e}")))?
            .into_owned();
        map.entries.push((key, value));
    }
    Ok(map)
}

#[derive(Default)]
/// Attribute lookup with the accessors the parser needs.
struct AttrMap {
    entries: Vec<(String, String)>,
}

impl AttrMap {
    fn opt(&self, key: &str) -> Option<String> {
        self.entries
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.clone())
    }

    /// Raw string value, or an empty string when absent.
    fn first(&self, key: &str) -> String {
        self.opt(key).unwrap_or_default()
    }

    /// Value parsed as f64, or `None` when absent or unparsable.
    fn number(&self, key: &str) -> Option<f64> {
        self.opt(key).and_then(|v| v.trim().parse().ok())
    }

    /// Comma-separated f64 list (`ax`, `ay`, `omrmap` use this shape).
    fn numbers(&self, key: &str) -> Vec<f64> {
        self.opt(key)
            .map(|v| {
                v.split(',')
                    .filter_map(|part| part.trim().parse().ok())
                    .collect()
            })
            .unwrap_or_default()
    }

    fn labels(&self, key: &str) -> Vec<String> {
        self.opt(key)
            .map(|v| {
                v.split(',')
                    .map(|part| part.trim().to_owned())
                    .filter(|part| !part.is_empty())
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Transcodes a GBK/GB2312 document to UTF-8 so the XML reader sees valid text.
/// Decodes the document to UTF-8. The declaration says GB2312, which is a
/// subset of GBK, so the bytes are transcoded through GBK; a UTF-8 document
/// decodes through `from_utf8` first.
fn decode_xml(bytes: &[u8]) -> Result<String> {
    if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) || std::str::from_utf8(bytes).is_ok() {
        return String::from_utf8(bytes.to_vec())
            .map_err(|e| Error::Template(format!("utf-8: {e}")));
    }
    let (text, _, had_errors) = encoding_rs::GBK.decode(bytes);
    if had_errors {
        return Err(Error::Template(
            "template is neither valid UTF-8 nor valid GBK".into(),
        ));
    }
    Ok(text.into_owned())
}
