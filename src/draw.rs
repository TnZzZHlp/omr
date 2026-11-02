//! Raster primitives used to draw over a scan.
//!
//! Everything here works on an RGB image in place: lines, quadrilateral
//! outlines, filled rectangles and a small bitmap font. The font is built in
//! rather than loaded from the system, so an annotated sheet looks the same on
//! every machine.

use image::{Rgb, RgbImage};

/// Colour of a confidently read mark.
pub const OK: Rgb<u8> = Rgb([0, 150, 60]);
/// Colour of a reading that a human should confirm.
pub const REVIEW: Rgb<u8> = Rgb([215, 135, 0]);
/// Colour of a second cell that also carries ink.
pub const CONFLICT: Rgb<u8> = Rgb([200, 40, 40]);
/// Colour of the registration marks.
pub const REGISTRATION: Rgb<u8> = Rgb([30, 100, 210]);
/// Colour of the barcode reading.
pub const BARCODE: Rgb<u8> = Rgb([120, 60, 190]);
/// Colour of a cell that was measured but not selected.
pub const OUTLINE: Rgb<u8> = Rgb([150, 170, 200]);
/// Text colour.
pub const INK: Rgb<u8> = Rgb([15, 15, 15]);
const PAPER: Rgb<u8> = Rgb([255, 255, 255]);

/// Copies a grayscale scan into an RGB image.
pub fn to_rgb(width: usize, height: usize, pixels: &[u8]) -> RgbImage {
    let mut out = RgbImage::new(width as u32, height as u32);
    for (x, y, pixel) in out.enumerate_pixels_mut() {
        let value = pixels[y as usize * width + x as usize];
        *pixel = Rgb([value, value, value]);
    }
    out
}

/// Draws a line of the given thickness.
pub fn line(img: &mut RgbImage, from: (f64, f64), to: (f64, f64), thickness: u32, color: Rgb<u8>) {
    let (x0, y0) = (from.0.round() as i64, from.1.round() as i64);
    let (x1, y1) = (to.0.round() as i64, to.1.round() as i64);
    let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
    let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
    let (mut x, mut y, mut error) = (x0, y0, dx + dy);
    loop {
        dot(img, x, y, thickness, color);
        if x == x1 && y == y1 {
            break;
        }
        let twice = 2 * error;
        if twice >= dy {
            error += dy;
            x += sx;
        }
        if twice <= dx {
            error += dx;
            y += sy;
        }
    }
}

/// Draws a closed four sided outline, e.g. a template rectangle seen through a
/// projective transform.
pub fn quad(img: &mut RgbImage, corners: [(f64, f64); 4], thickness: u32, color: Rgb<u8>) {
    for i in 0..4 {
        line(img, corners[i], corners[(i + 1) % 4], thickness, color);
    }
}

/// Draws a rectangle given by two opposite corners.
pub fn rect(img: &mut RgbImage, from: (f64, f64), to: (f64, f64), thickness: u32, color: Rgb<u8>) {
    quad(
        img,
        [
            (from.0, from.1),
            (to.0, from.1),
            (to.0, to.1),
            (from.0, to.1),
        ],
        thickness,
        color,
    );
}

/// Paints a solid rectangle, clipped to the image.
pub fn fill_rect(img: &mut RgbImage, x: i64, y: i64, width: u32, height: u32, color: Rgb<u8>) {
    for py in y..y + height as i64 {
        for px in x..x + width as i64 {
            put(img, px, py, color);
        }
    }
}

/// Draws a small cross centred on a point, used for detected corners.
pub fn cross(img: &mut RgbImage, at: (f64, f64), radius: u32, thickness: u32, color: Rgb<u8>) {
    let (x, y) = (at.0.round(), at.1.round());
    let r = f64::from(radius);
    line(img, (x - r, y), (x + r, y), thickness, color);
    line(img, (x, y - r), (x, y + r), thickness, color);
}

fn dot(img: &mut RgbImage, x: i64, y: i64, thickness: u32, color: Rgb<u8>) {
    let half = (thickness.max(1) / 2) as i64;
    for py in y - half..=y + half {
        for px in x - half..=x + half {
            put(img, px, py, color);
        }
    }
}

fn put(img: &mut RgbImage, x: i64, y: i64, color: Rgb<u8>) {
    if x < 0 || y < 0 {
        return;
    }
    if x < i64::from(img.width()) && y < i64::from(img.height()) {
        img.put_pixel(x as u32, y as u32, color);
    }
}

/// Width of a rendered string in pixels.
pub fn text_width(text: &str, scale: u32) -> u32 {
    text.chars().count() as u32 * 6 * scale.max(1)
}

/// Height of a rendered string in pixels.
pub const fn text_height(scale: u32) -> u32 {
    7 * if scale == 0 { 1 } else { scale }
}

/// Draws a string in the built in 5x7 font.
pub fn text(img: &mut RgbImage, x: i64, y: i64, text: &str, scale: u32, color: Rgb<u8>) {
    let scale = scale.max(1);
    let mut cursor = x;
    for ch in text.chars() {
        if let Some(rows) = glyph(ch) {
            for (row, pattern) in rows.iter().enumerate() {
                for (column, bit) in pattern.bytes().enumerate() {
                    if bit != b'1' {
                        continue;
                    }
                    fill_rect(
                        img,
                        cursor + column as i64 * i64::from(scale),
                        y + row as i64 * i64::from(scale),
                        scale,
                        scale,
                        color,
                    );
                }
            }
        }
        cursor += i64::from(6 * scale);
    }
}

/// Draws a label on a solid plate so it stays legible over any background.
///
/// Returns the plate rectangle as `(x, y, width, height)`.
pub fn badge(
    img: &mut RgbImage,
    x: i64,
    y: i64,
    label: &str,
    scale: u32,
    accent: Rgb<u8>,
) -> (i64, i64, u32, u32) {
    let padding = 2 * scale.max(1);
    let bar = 2 * scale.max(1);
    let width = text_width(label, scale) + padding * 2 + bar;
    let height = text_height(scale) + padding;
    fill_rect(img, x, y, width, height, PAPER);
    fill_rect(img, x, y, bar, height, accent);
    text(
        img,
        x + bar as i64 + padding as i64,
        y + (padding / 2) as i64,
        label,
        scale,
        INK,
    );
    (x, y, width, height)
}

/// 5x7 glyph for `ch`, as seven rows of five bits.
///
/// Only the characters the annotations need are defined; anything else is drawn
/// as a blank, which keeps an unexpected label from corrupting the image.
fn glyph(ch: char) -> Option<[&'static str; 7]> {
    let rows = match ch.to_ascii_uppercase() {
        '0' => [
            "01110", "10001", "10011", "10101", "11001", "10001", "01110",
        ],
        '1' => [
            "00100", "01100", "00100", "00100", "00100", "00100", "01110",
        ],
        '2' => [
            "01110", "10001", "00001", "00010", "00100", "01000", "11111",
        ],
        '3' => [
            "11111", "00010", "00100", "00010", "00001", "10001", "01110",
        ],
        '4' => [
            "00010", "00110", "01010", "10010", "11111", "00010", "00010",
        ],
        '5' => [
            "11111", "10000", "11110", "00001", "00001", "10001", "01110",
        ],
        '6' => [
            "00110", "01000", "10000", "11110", "10001", "10001", "01110",
        ],
        '7' => [
            "11111", "00001", "00010", "00100", "01000", "01000", "01000",
        ],
        '8' => [
            "01110", "10001", "10001", "01110", "10001", "10001", "01110",
        ],
        '9' => [
            "01110", "10001", "10001", "01111", "00001", "00010", "01100",
        ],
        'A' => [
            "01110", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'B' => [
            "11110", "10001", "10001", "11110", "10001", "10001", "11110",
        ],
        'C' => [
            "01110", "10001", "10000", "10000", "10000", "10001", "01110",
        ],
        'D' => [
            "11100", "10010", "10001", "10001", "10001", "10010", "11100",
        ],
        'E' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "11111",
        ],
        'F' => [
            "11111", "10000", "10000", "11110", "10000", "10000", "10000",
        ],
        'G' => [
            "01110", "10001", "10000", "10111", "10001", "10001", "01111",
        ],
        'H' => [
            "10001", "10001", "10001", "11111", "10001", "10001", "10001",
        ],
        'I' => [
            "01110", "00100", "00100", "00100", "00100", "00100", "01110",
        ],
        'J' => [
            "00111", "00010", "00010", "00010", "00010", "10010", "01100",
        ],
        'K' => [
            "10001", "10010", "10100", "11000", "10100", "10010", "10001",
        ],
        'L' => [
            "10000", "10000", "10000", "10000", "10000", "10000", "11111",
        ],
        'M' => [
            "10001", "11011", "10101", "10101", "10001", "10001", "10001",
        ],
        'N' => [
            "10001", "11001", "10101", "10101", "10011", "10001", "10001",
        ],
        'O' => [
            "01110", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'P' => [
            "11110", "10001", "10001", "11110", "10000", "10000", "10000",
        ],
        'Q' => [
            "01110", "10001", "10001", "10001", "10101", "10010", "01101",
        ],
        'R' => [
            "11110", "10001", "10001", "11110", "10100", "10010", "10001",
        ],
        'S' => [
            "01111", "10000", "10000", "01110", "00001", "00001", "11110",
        ],
        'T' => [
            "11111", "00100", "00100", "00100", "00100", "00100", "00100",
        ],
        'U' => [
            "10001", "10001", "10001", "10001", "10001", "10001", "01110",
        ],
        'V' => [
            "10001", "10001", "10001", "10001", "10001", "01010", "00100",
        ],
        'W' => [
            "10001", "10001", "10001", "10101", "10101", "11011", "10001",
        ],
        'X' => [
            "10001", "10001", "01010", "00100", "01010", "10001", "10001",
        ],
        'Y' => [
            "10001", "10001", "01010", "00100", "00100", "00100", "00100",
        ],
        'Z' => [
            "11111", "00001", "00010", "00100", "01000", "10000", "11111",
        ],
        '.' => [
            "00000", "00000", "00000", "00000", "00000", "01100", "01100",
        ],
        ',' => [
            "00000", "00000", "00000", "00000", "01100", "01100", "01000",
        ],
        '-' => [
            "00000", "00000", "00000", "11111", "00000", "00000", "00000",
        ],
        '+' => [
            "00000", "00100", "00100", "11111", "00100", "00100", "00000",
        ],
        '=' => [
            "00000", "00000", "11111", "00000", "11111", "00000", "00000",
        ],
        ':' => [
            "00000", "01100", "01100", "00000", "01100", "01100", "00000",
        ],
        '?' => [
            "01110", "10001", "00001", "00010", "00100", "00000", "00100",
        ],
        '!' => [
            "00100", "00100", "00100", "00100", "00100", "00000", "00100",
        ],
        '/' => [
            "00001", "00010", "00010", "00100", "01000", "01000", "10000",
        ],
        '%' => [
            "10001", "00010", "00100", "00100", "01000", "10001", "00001",
        ],
        '#' => [
            "01010", "01010", "11111", "01010", "11111", "01010", "01010",
        ],
        '(' => [
            "00010", "00100", "01000", "01000", "01000", "00100", "00010",
        ],
        ')' => [
            "01000", "00100", "00010", "00010", "00010", "00100", "01000",
        ],
        ' ' => [
            "00000", "00000", "00000", "00000", "00000", "00000", "00000",
        ],
        _ => return None,
    };
    Some(rows)
}
