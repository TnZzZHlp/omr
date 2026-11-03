//! Crate error type.
//!
//! Every fallible operation in the crate funnels through [`Error`]: image and
//! file access (`Io`, `Image`), malformed templates and unreadable marks
//! (`Template`), alignment refusals (`Registration`) and symbol decoding
//! failures (`Decode`). The variants carry a human readable message rather
//! than structured data — the callers that need details (the snapshot tests,
//! the JSON output) embed the message verbatim.

use std::fmt;

/// The error type of the crate.
#[derive(Debug)]
pub enum Error {
    /// A file could not be read or written.
    Io(std::io::Error),
    /// An image could not be loaded, or a raster was malformed.
    Image(String),
    /// A barcode or candidate number could not be decoded.
    Decode(String),
    /// The template document is malformed, or declares no such page.
    Template(String),
    /// The scan could not be aligned to the template. This is a defined
    /// outcome for a wrong page or a missing corner layout, not a crash:
    /// callers record it and move on.
    Registration(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Image(m) => write!(f, "image error: {m}"),
            Error::Decode(m) => write!(f, "decode error: {m}"),
            Error::Template(m) => write!(f, "template error: {m}"),
            Error::Registration(m) => write!(f, "registration error: {m}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(value: std::io::Error) -> Self {
        Error::Io(value)
    }
}

impl From<image::ImageError> for Error {
    fn from(value: image::ImageError) -> Self {
        Error::Image(value.to_string())
    }
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
