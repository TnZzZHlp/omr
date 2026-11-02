//! Minimal 8-bit grayscale raster with bilinear sampling.

use crate::error::{Error, Result};
use crate::geometry::{Point, Rect};

#[derive(Clone, Debug)]
pub struct Gray {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Gray {
    pub fn new(width: usize, height: usize, pixels: Vec<u8>) -> Result<Self> {
        if pixels.len() != width * height {
            return Err(Error::Image(format!(
                "expected {} pixels, found {}",
                width * height,
                pixels.len()
            )));
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let image = image::open(path)?;
        Ok(Self::from_dynamic(image))
    }

    pub fn from_dynamic(image: image::DynamicImage) -> Self {
        let gray = image.into_luma8();
        let (width, height) = gray.dimensions();
        Self {
            width: width as usize,
            height: height as usize,
            pixels: gray.into_raw(),
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[u8] {
        &self.pixels
    }

    pub fn get(&self, x: usize, y: usize) -> u8 {
        let x = x.min(self.width.saturating_sub(1));
        let y = y.min(self.height.saturating_sub(1));
        self.pixels[y * self.width + x]
    }

    pub fn get_signed(&self, x: isize, y: isize) -> u8 {
        let x = x.clamp(0, self.width.saturating_sub(1) as isize) as usize;
        let y = y.clamp(0, self.height.saturating_sub(1) as isize) as usize;
        self.pixels[y * self.width + x]
    }

    pub fn sample(&self, p: Point) -> f64 {
        let x = p.x - 0.5;
        let y = p.y - 0.5;
        let x0 = x.floor();
        let y0 = y.floor();
        let fx = x - x0;
        let fy = y - y0;
        let x0 = x0 as isize;
        let y0 = y0 as isize;
        let p00 = f64::from(self.get_signed(x0, y0));
        let p10 = f64::from(self.get_signed(x0 + 1, y0));
        let p01 = f64::from(self.get_signed(x0, y0 + 1));
        let p11 = f64::from(self.get_signed(x0 + 1, y0 + 1));
        let top = p00 * (1.0 - fx) + p10 * fx;
        let bottom = p01 * (1.0 - fx) + p11 * fx;
        top * (1.0 - fy) + bottom * fy
    }

    pub fn sub_image(&self, rect: Rect) -> Gray {
        let (x0, y0, x1, y1) = rect.pixel_bounds();
        let x1 = x1.min(self.width.saturating_sub(1));
        let y1 = y1.min(self.height.saturating_sub(1));
        if x1 < x0 || y1 < y0 {
            return Gray::new(1, 1, vec![255]).expect("1x1 raster");
        }
        let w = x1 - x0 + 1;
        let h = y1 - y0 + 1;
        let mut pixels = Vec::with_capacity(w * h);
        for y in y0..=y1 {
            let start = y * self.width + x0;
            pixels.extend_from_slice(&self.pixels[start..start + w]);
        }
        Gray::new(w, h, pixels).expect("consistent raster")
    }

    /// Writes `P2` PGM, which keeps debug dumps dependency free.
    pub fn save_pgm(&self, path: impl AsRef<std::path::Path>) -> Result<()> {
        use std::io::Write;
        let mut out = std::fs::File::create(path)?;
        write!(out, "P5\n{} {}\n255\n", self.width, self.height)?;
        out.write_all(&self.pixels)?;
        Ok(())
    }
}
