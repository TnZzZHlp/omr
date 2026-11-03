//! Coordinates, rectangles and projective transforms.

use crate::error::{Error, Result};

/// A point in template or scan coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl Point {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Euclidean distance to `other`.
    pub fn distance(self, other: Point) -> f64 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }
}

/// An axis-aligned rectangle in half-open template coordinates:
/// `x0 <= x < x1`, `y0 <= y < y1`. Cell rects, block areas and mark boxes all
/// use this form.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    pub const fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// The smallest rectangle covering both points.
    pub fn from_corners(a: Point, b: Point) -> Self {
        Self::new(a.x.min(b.x), a.y.min(b.y), a.x.max(b.x), a.y.max(b.y))
    }

    /// Extent along x: `x1 - x0`.
    pub fn width(self) -> f64 {
        self.x1 - self.x0
    }

    /// Extent along y: `y1 - y0`.
    pub fn height(self) -> f64 {
        self.y1 - self.y0
    }

    /// The midpoint of the two corners.
    pub fn center(self) -> Point {
        Point::new((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)
    }

    /// The smallest rectangle covering both.
    pub fn union(self, other: Rect) -> Rect {
        Rect::new(
            self.x0.min(other.x0),
            self.y0.min(other.y0),
            self.x1.max(other.x1),
            self.y1.max(other.y1),
        )
    }

    /// Moves the rectangle by `by`.
    pub fn translate(self, by: Point) -> Rect {
        Rect::new(
            self.x0 + by.x,
            self.y0 + by.y,
            self.x1 + by.x,
            self.y1 + by.y,
        )
    }

    /// Grows the rectangle by `by` on every side.
    pub fn expand(self, by: f64) -> Rect {
        Rect::new(self.x0 - by, self.y0 - by, self.x1 + by, self.y1 + by)
    }

    /// Shrinks the rectangle by fractions of its width (`fx`) and height
    /// (`fy`) on every side; 0.3 insets 30% from the left and right. Used to
    /// ignore the printed bracket border when judging a cell interior.
    pub fn inset(self, fx: f64, fy: f64) -> Rect {
        Rect::new(
            self.x0 + self.width() * fx,
            self.y0 + self.height() * fy,
            self.x1 - self.width() * fx,
            self.y1 - self.height() * fy,
        )
    }

    /// True when `p` lies inside the rectangle, edges included.
    pub fn contains(self, p: Point) -> bool {
        p.x >= self.x0 && p.x <= self.x1 && p.y >= self.y0 && p.y <= self.y1
    }

    /// The four corners, clockwise from the top-left. The annotation draws
    /// the quads through them.
    pub fn as_points(self) -> [Point; 4] {
        [
            Point::new(self.x0, self.y0),
            Point::new(self.x1, self.y0),
            Point::new(self.x0, self.y1),
            Point::new(self.x1, self.y1),
        ]
    }

    /// Every pixel whose centre lies inside the rectangle.
    /// The corners as inclusive integer pixel indices, negative values
    /// clamped to zero. The rasterizers use these to stay on the grid.
    pub fn pixel_bounds(self) -> (usize, usize, usize, usize) {
        let x0 = (self.x0 - 0.5).ceil().max(0.0) as usize;
        let y0 = (self.y0 - 0.5).ceil().max(0.0) as usize;
        let x1 = (self.x1 - 0.5).floor().max(-1.0) as isize;
        let y1 = (self.y1 - 0.5).floor().max(-1.0) as isize;
        (x0, y0, x1.max(0) as usize, y1.max(0) as usize)
    }
}

/// Projective mapping from template coordinates to image coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Homography {
    pub(crate) m: [f64; 9],
}

impl Homography {
    pub const fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Scale by `(sx, sy)` and then translate by `(tx, ty)`.
    pub fn scale_translate(sx: f64, sy: f64, tx: f64, ty: f64) -> Self {
        Self {
            m: [sx, 0.0, tx, 0.0, sy, ty, 0.0, 0.0, 1.0],
        }
    }

    /// Builds a transform from raw coefficients, row major.
    pub const fn from_coefficients(m: [f64; 9]) -> Self {
        Self { m }
    }

    pub fn coefficients(&self) -> [f64; 9] {
        self.m
    }

    /// True when the transform maps the unit w-row to 1 everywhere, so points
    /// can be advanced by adding the column step instead of re-applying the
    /// full projection.
    pub fn is_affine(&self) -> bool {
        self.m[6].abs() < 1e-12 && self.m[7].abs() < 1e-12
    }

    /// Maps a template point into the scan. The projective denominator is
    /// clamped away from zero; for the affine transforms the registration
    /// produces it is always 1.
    pub fn apply(&self, p: Point) -> Point {
        let m = &self.m;
        let w = m[6] * p.x + m[7] * p.y + m[8];
        let w = if w.abs() < 1e-12 { 1e-12 } else { w };
        Point::new(
            (m[0] * p.x + m[1] * p.y + m[2]) / w,
            (m[3] * p.x + m[4] * p.y + m[5]) / w,
        )
    }

    /// Least squares fit of `src -> dst`; four or more correspondences required.
    /// Least-squares projective fit of `src -> dst` via the normal equations;
    /// four or more correspondences required. Degenerate configurations (all
    /// points collinear, duplicate points) return an error. Used to fit the
    /// four corner marks and to refine the fit on the second pass.
    pub fn fit(src: &[Point], dst: &[Point]) -> Result<Self> {
        if src.len() != dst.len() || src.len() < 4 {
            return Err(Error::Registration(
                "homography needs at least four correspondences".into(),
            ));
        }

        let mut a = vec![vec![0.0f64; 8]; 8];
        let mut b = vec![0.0f64; 8];
        for (s, d) in src.iter().zip(dst) {
            let rows = [
                ([s.x, s.y, 1.0, 0.0, 0.0, 0.0, -s.x * d.x, -s.y * d.x], d.x),
                ([0.0, 0.0, 0.0, s.x, s.y, 1.0, -s.x * d.y, -s.y * d.y], d.y),
            ];
            for (row, rhs) in rows {
                for i in 0..8 {
                    for j in 0..8 {
                        a[i][j] += row[i] * row[j];
                    }
                    b[i] += row[i] * rhs;
                }
            }
        }

        let h = solve(&mut a, &mut b)
            .ok_or_else(|| Error::Registration("registration points are degenerate".into()))?;
        let mut m = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], 1.0];
        let norm = (m[8]).abs().max(1e-12);
        for v in m.iter_mut() {
            *v /= norm;
        }
        Ok(Self { m })
    }
}

/// Gaussian elimination with partial pivoting; `None` when singular.
fn solve(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<[f64; 8]> {
    let n = b.len();
    for col in 0..n {
        let mut pivot = col;
        for row in col + 1..n {
            if a[row][col].abs() > a[pivot][col].abs() {
                pivot = row;
            }
        }
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);

        let d = a[col][col];
        for value in a[col].iter_mut().skip(col) {
            *value /= d;
        }
        b[col] /= d;

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col];
            if factor == 0.0 {
                continue;
            }
            let source = a[col][col..].to_vec();
            for (target, value) in a[row][col..].iter_mut().zip(source.iter()) {
                *target -= factor * value;
            }
            b[row] -= factor * b[col];
        }
    }
    let mut out = [0.0; 8];
    out.copy_from_slice(&b[..8]);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homography_roundtrip() {
        let src = [
            Point::new(100.0, 50.0),
            Point::new(2100.0, 60.0),
            Point::new(110.0, 1440.0),
            Point::new(2120.0, 1450.0),
        ];
        let dst = [
            Point::new(102.0, 52.0),
            Point::new(2098.0, 58.0),
            Point::new(112.0, 1438.0),
            Point::new(2118.0, 1448.0),
        ];
        let h = Homography::fit(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let p = h.apply(*s);
            assert!(p.distance(*d) < 1e-6, "{p:?} vs {d:?}");
        }
    }
}
