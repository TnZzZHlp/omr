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
    #[serde(default)]
    #[allow(dead_code)]
    minrate: String,
}

#[derive(Deserialize)]
struct ItemDef {
    sbx: String,
    /// Type: 0 page number, 2 absence, 4 candidate number grid, 10 scored
    /// question.
    lx: i64,
    #[serde(default)]
    stfs: f64,
    #[serde(default)]
    stda: String,
    #[serde(default)]
    zxs: Vec<CellDef>,
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

        Ok(Template {
            version: "json".into(),
            id: root.mbid,
            dpi: 0.0,
            pages,
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
        marks.push(Mark {
            index: def.xh.max(0) as u32,
            name: format!("mli{}", def.xh),
            origin: Point::new((x1 + x2) / 2.0, (y1 + y2) / 2.0),
            found: Some(Rect::from_corners(Point::new(x1, y1), Point::new(x2, y2))),
            box_declared: true,
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
        blocks.push(item_block(item, BlockKind::PageNumber, "1")?);
    }
    let empty = Vec::new();
    for item in root.xxds.get(key).unwrap_or(&empty) {
        let kind = match item.lx {
            2 => BlockKind::Absent,
            4 => BlockKind::CandidateNumber,
            other => {
                return Err(Error::Template(format!(
                    "unsupported information item {} (lx {other})",
                    item.sbx
                )))
            }
        };
        blocks.push(item_block(item, kind, "2")?);
    }
    for item in root.sts.get(key).unwrap_or(&empty) {
        blocks.push(item_block(item, BlockKind::Question, "5")?);
    }
    Ok(blocks)
}

/// Converts one item into a block by expanding every `zbs` rectangle into a
/// cell. Each `zx` is one option column of the item; `hs` rows by `ls`
/// columns of cells live inside it, laid out vertically.
fn item_block(item: &ItemDef, kind: BlockKind, raw_type: &str) -> Result<Block> {
    let mut cells = Vec::new();
    for zx in &item.zxs {
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
                let label = option_labels
                    .get(row)
                    .cloned()
                    .unwrap_or_else(|| row.to_string());
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
            "item {} declares no cells",
            item.sbx
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
        name: item.sbx.clone(),
        kind,
        question,
        answer_key,
        score,
        area,
        cells,
        raw_type: raw_type.to_owned(),
    })
}
