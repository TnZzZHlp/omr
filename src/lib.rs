//! Reader for `.mb` answer-sheet templates.
//!
//! The template is a GBK encoded XML document that declares, per scanned page,
//! four registration marks (`mli`) and the answer blocks (`omr`) laid out in the
//! page coordinate space. This crate parses that document, registers a scan to
//! the template coordinate space, and reports the marks that a candidate filled.
//!
//! # Pipeline
//!
//! 1. [`template::Template::parse_file`] decodes the GBK XML into pages of
//!    marks and blocks in template coordinates.
//! 2. [`registration::register`] aligns the scan: corner marks first (blobs
//!    located by darkness, verified by stamp correlation, fitted with a
//!    projective transform and checked against the printed grid); three marks
//!    are enough, so a folded corner does not stop the page. When too few marks
//!    can be found, the printed content of the blocks aligns the scan by
//!    translation.
//! 3. [`fill`] measures every declared cell — against its column baseline when
//!    sibling rows share the printed letter, absolutely when they do not — and
//!    [`recognize::register_scan`] decides the reading with per-question
//!    confidence and review flags.
//! 4. [`barcode`] decodes the candidate-number Code 128 symbol where present.
//! 5. [`annotate`] draws the readings back onto the scan for human review.
//!
//! # Entry points
//!
//! * [`recognize::register_scan`] — the one-call API: template, page, scan in,
//!   [`recognize::Recognition`] out.
//! * [`annotate::annotate`] / [`annotate::save_annotated`] — the annotated
//!   image the CLI binaries write.

pub mod annotate;
pub mod barcode;
pub mod draw;
pub mod error;
pub mod fill;
pub mod geometry;
pub mod gray;
pub mod recognize;
pub mod registration;
pub mod template;
pub mod template_json;

pub use annotate::Style;
pub use barcode::BarcodeResult;
pub use error::{Error, Result};
pub use geometry::{Point, Rect};
pub use gray::Gray;
pub use recognize::{
    read_with_transform, register_scan, AbsentMark, Options, PageNumber, Recognition,
    RegistrationInfo, Selection,
};
pub use registration::AlignmentMode;
pub use template::{Block, BlockKind, Mark, Page, Template};
