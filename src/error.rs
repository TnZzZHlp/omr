use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Image(String),
    Decode(String),
    Template(String),
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

pub type Result<T> = std::result::Result<T, Error>;
