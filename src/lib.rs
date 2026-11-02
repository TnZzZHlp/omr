//! Reader for `.mb` answer-sheet templates.
//!
//! The template is a GBK encoded XML document that declares, per scanned page,
//! four registration marks (`mli`) and the answer blocks (`omr`) laid out in the
//! page coordinate space. This crate parses that document, registers a scan to
//! the template coordinate space, and reports the marks that a candidate filled.

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
