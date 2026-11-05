//! Parses the scan-centre JSON template (the `/smmb/get.jsmeb` payload) into
//! the same [`Template`] model the `.mb` XML parser produces.
//!
//! The JSON organises every item into per-page dictionaries keyed by the
//! one-based page number: `dwds` (registration marks), `yms` (page number
//! blocks), `xxds` (information items: candidate number grid, absence mark)
//! and `sts` (scored questions). Cell geometry is spelled out as absolute
//! `x,y,w,h` rectangles in the `zbs` string, so no `ax`/`ay` layout math is
//! needed. `dtcqks` (subjective answer regions) carries no fill cells and is
//! skipped.

use crate::error::{Error, Result};
use crate::geometry::{Point, Rect};
use crate::template::{Block, BlockKind, Cell, Mark, Page, Template};
use serde::Deserialize;

#[derive(Deserialize)]
struct Doc {
    result: Root,
}

#[derive(Deserialize)]
struct Root {
    mbid: String,
    #[serde(default)]
    dwds: std::collections::BTreeMap<String, Vec<MarkDef>>,
    #[serde(default)]
    yms: std::collections::BTreeMap<String, ItemDef>,
    #[serde(default)]
    xxds: std::collections::BTreeMap<String, Vec<ItemDef>>,
    #[serde(default)]
    sts: std::collections::BTreeMap<String, Vec<ItemDef>>,
}

#[derive(Deserialize)]
struct MarkDef {
    xh: i64,
    #[serde(default)]
    #[allow(dead_code)]
    lx: i64,
    #[serde(default)]
    x1: String,
    #[serde(default)]
    y1: String,
    #[serde(default)]
    x2: String,
    #[serde(default)]
    y2: String,
    /// Minimum filled fraction of the mark, percent 0..100. 0 means the
    /// generator did not measure the mark; the box may hold several separate
    /// shapes, so the detector should not demand a solid square.
    #[serde(default)]
    minrate: String,
}

#[derive(Deserialize)]
struct ItemDef {
    sbx: String,
    /// Type: 0 page number, 2 absence, 4 candidate number grid, 10 scored
    /// question.
    lx: i64,
    /// Answers expected per question: 1 for a single-choice question, 0 for an
    /// unconstrained one (multi-select, and the page number's bit sum).
    #[serde(default = "one")]
    dx: i64,
    #[serde(default)]
    stfs: f64,
    #[serde(default)]
    stda: String,
    #[serde(default)]
    zxs: Vec<CellDef>,
}

fn one() -> i64 {
    1
}

#[derive(Deserialize)]
struct CellDef {
    #[serde(default)]
    #[allow(dead_code)]
    xh: i64,
    hs: i64,
    ls: i64,
    #[serde(default)]
    hlys: String,
    /// `x,y,w,h;x,y,w,h;...` absolute rectangles, one per option cell.
    #[serde(default)]
    zbs: String,
}

impl Template {
    /// Parses the scan-centre JSON template document.
    pub fn parse_json(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes)
            .map_err(|e| Error::Template(format!("utf-8: {e}")))?;
        let doc: Doc = serde_json::from_str(text)
            .map_err(|e| Error::Template(format!("json: {e}")))?;
        Self::from_root(doc.result)
    }

    /// Reads and parses a JSON template from disk.
    pub fn parse_json_file(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::parse_json(&bytes)
    }

    fn from_root(root: Root) -> Result<Self> {
        let mut page_numbers: Vec<u64> = root
            .dwds
            .keys()
            .chain(root.yms.keys())
            .chain(root.xxds.keys())
            .chain(root.sts.keys())
            .filter_map(|k| k.parse().ok())
            .collect();
        page_numbers.sort_unstable();
        page_numbers.dedup();
        if page_numbers.is_empty() {
            return Err(Error::Template("json template declares no pages".into()));
        }

        let pages = page_numbers
            .iter()
            .enumerate()
            .map(|(index, number)| {
                let key = number.to_string();
                Ok(Page {
                    index,
                    marks: marks_of(&root.dwds, &key)?,
                    blocks: blocks_of(&root, &key)?,
                    barcode_direct: 1,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let minrate = root
            .dwds
            .values()
            .flat_map(|marks| marks.iter())
            .filter_map(|m| m.minrate.trim().parse::<f64>().ok())
            .map(|v| v / 100.0)
            .next();

        Ok(Template {
            version: "json".into(),
            id: root.mbid,
            dpi: 0.0,
            pages,
            mark_minrate: minrate,
        })
    }
}

/// Builds the registration marks of one page. The JSON states a bounding box
/// (`x1..y2`) and `minrate`, the same two facts the XML `mli` records; there
/// is no stamp raster, so the darkness sweep locates these marks.
fn marks_of(
    dwds: &std::collections::BTreeMap<String, Vec<MarkDef>>,
    key: &str,
) -> Result<Vec<Mark>> {
    let empty = Vec::new();
    let defs = dwds.get(key).unwrap_or(&empty);
    let mut marks = Vec::new();
    for def in defs {
        let (Some(x1), Some(y1), Some(x2), Some(y2)) = (
            def.x1.trim().parse::<f64>().ok(),
            def.y1.trim().parse::<f64>().ok(),
            def.x2.trim().parse::<f64>().ok(),
            def.y2.trim().parse::<f64>().ok(),
        ) else {
            continue;
        };
        if (x2 - x1).abs() <= 1.0 || (y2 - y1).abs() <= 1.0 {
            continue;
        }
        // `lx=0` marks declare a search window the mark sits inside (possibly
        // a wide strip of several shapes); `lx=1` marks state the mark itself.
        let window = def.lx == 0;
        marks.push(Mark {
            index: def.xh.max(0) as u32,
            name: format!("mli{}", def.xh),
            origin: Point::new((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            found: Some(Rect::from_corners(Point::new(x1, y1), Point::new(x2, y2))),
            box_declared: true,
            window,
            patch: None,
        });
    }
    if marks.is_empty() {
        return Err(Error::Template(format!(
            "page {key} declares no usable registration marks"
        )));
    }
    Ok(marks)
}

/// Builds the answer blocks of one page from the three item families. Items
/// keep their declaration order within each family; the families are appended
/// page number, information items, then questions, matching the `.mb`
/// declaration order.
fn blocks_of(root: &Root, key: &str) -> Result<Vec<Block>> {
    let mut blocks = Vec::new();
    if let Some(item) = root.yms.get(key) {
        blocks.extend(item_blocks(item, BlockKind::PageNumber, "1")?);
    }
    let empty = Vec::new();
    for item in root.xxds.get(key).unwrap_or(&empty) {
        let kind = match item.lx {
            2 => BlockKind::Absent,
            // The scanner prints the number as a Code 128 symbol above the
            // digit grid and declares the whole sticker as one cell, so the
            // digit grid cannot be read from this item; the barcode is the
            // reading. JSON templates that do declare a real grid get the grid
            // reading instead.
            4 if is_barcode_area(item) => BlockKind::Barcode,
            4 => BlockKind::CandidateNumber,
            other => {
                return Err(Error::Template(format!(
                    "unsupported information item {} (lx {other})",
                    item.sbx
                )))
            }
        };
        blocks.extend(item_blocks(item, kind, "2")?);
    }
    for item in root.sts.get(key).unwrap_or(&empty) {
        // `dx` is the JSON spelling of the XML `multi` flag: 1 declares exactly
        // one answer, anything else leaves the question open to several.
        let kind = if item.dx == 1 {
            BlockKind::Question
        } else {
            BlockKind::Multiple
        };
        blocks.extend(item_blocks(item, kind, "5")?);
    }
    Ok(blocks)
}

/// True when a candidate-number item is the printed sticker rather than a
/// digit grid.
///
/// A grid declares one cell per digit (`ls` columns of ten cells), while the
/// sticker declares a single cell covering the barcode, the human readable
/// number and the name. Its height then spans much more than one line of print,
/// which is what this measures: a digit row is a tenth of a grid column, a
/// sticker cell is tens of pixels tall.
fn is_barcode_area(item: &ItemDef) -> bool {
    let cells: usize = item.zxs.iter().map(|zx| zx.ls.max(1)).sum::<i64>() as usize;
    let tallest = item
        .zxs
        .iter()
        .flat_map(|zx| zx.zbs.split(';'))
        .filter_map(|spec| {
            let parts: Vec<f64> = spec.split(',').map(|v| v.trim().parse().unwrap_or(0.0)).collect();
            parts.get(3).copied()
        })
        .fold(0.0f64, f64::max);
    cells <= 1 || tallest > 40.0
}

/// Converts one item into its blocks.
///
/// An item declares its geometry as one or more `zx` groups. A question keeps
/// every option of one row in a single group, so it stays one block. A digit
/// grid declares one group per digit column, and those columns are read
/// independently, so each group becomes a block of its own exactly the way the
/// `.mb` format writes `KSH.1`, `KSH.2`, … — collapsing them into one block
/// would rank all hundred cells of the grid against each other and report a
/// single digit instead of the number.
fn item_blocks(item: &ItemDef, kind: BlockKind, raw_type: &str) -> Result<Vec<Block>> {
    let digit_grid = kind == BlockKind::CandidateNumber
        && item.zxs.len() > 1
        && item.zxs.iter().all(|zx| zx.hs > 1);
    if !digit_grid {
        return Ok(vec![item_block(item, &item.zxs, item.sbx.clone(), kind, raw_type)?]);
    }
    item.zxs
        .iter()
        .enumerate()
        .map(|(index, zx)| {
            let name = format!("{}.{}", item.sbx, index + 1);
            item_block(item, std::slice::from_ref(zx), name, kind, raw_type)
        })
        .collect()
}

/// Converts one item, or one `zx` group of it, into a block by expanding every
/// `zbs` rectangle into a cell; `hs` rows by `ls` columns of cells live inside
/// each group, laid out down the rows of each column.
fn item_block(
    item: &ItemDef,
    zxs: &[CellDef],
    name: String,
    kind: BlockKind,
    raw_type: &str,
) -> Result<Block> {
    let mut cells = Vec::new();
    for zx in zxs {
        let specs: Vec<&str> = zx.zbs.split(';').filter(|s| !s.trim().is_empty()).collect();
        let numbers: Vec<Vec<f64>> = specs
            .iter()
            .map(|spec| {
                spec.split(',')
                    .map(|v| v.trim().parse::<f64>().unwrap_or(0.0))
                    .collect()
            })
            .collect();
        // The zx declares `hs` rows and `ls` columns; the flat `zbs` list runs
        // down the rows of each column group. `ls` columns of cells map to
        // consecutive rectangles when the item is a digit/letter column set.
        let option_labels: Vec<String> = zx
            .hlys
            .split(',')
            .map(|v| v.trim().to_owned())
            .filter(|v| !v.is_empty())
            .collect();
        let rows = zx.hs.max(1) as usize;
        let columns = zx.ls.max(1) as usize;
        let mut index = 0usize;
        for row in 0..rows {
            for column in 0..columns {
                let Some(n) = numbers.get(index) else { break };
                index += 1;
                if n.len() < 4 {
                    continue;
                }
                // A single-row item (a question with `ls` options) maps the
                // labels across the columns; a column-shaped item (a digit
                // grid with `hs` rows) maps them down the rows.
                let label = if rows == 1 {
                    option_labels
                        .get(column)
                        .cloned()
                        .unwrap_or_else(|| column.to_string())
                } else {
                    option_labels
                        .get(row)
                        .cloned()
                        .unwrap_or_else(|| row.to_string())
                };
                cells.push(Cell {
                    column,
                    row,
                    label,
                    rect: Rect::new(n[0], n[1], n[0] + n[2], n[1] + n[3]),
                });
            }
        }
    }
    if cells.is_empty() {
        return Err(Error::Template(format!(
            "item {name} declares no cells"
        )));
    }

    let area = cells
        .iter()
        .map(|c| c.rect)
        .reduce(|a, b| a.union(b))
        .expect("non-empty cells");

    let question = item.sbx.trim().parse::<u32>().ok();
    let answer_key = (!item.stda.trim().is_empty()).then(|| item.stda.trim().to_owned());
    let score = (item.stfs > 0.0).then_some(item.stfs);

    Ok(Block {
        name,
        kind,
        question,
        answer_key,
        score,
        area,
        cells,
        raw_type: raw_type.to_owned(),
    })
}
