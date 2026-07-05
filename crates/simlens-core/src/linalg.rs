//! Small numeric helpers. f64 accumulation for numerical stability.

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum()
}

#[inline]
pub fn dot64(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm(a: &[f32]) -> f64 {
    dot(a, a).sqrt()
}

/// Unit-normalized copy (returns the zero vector unchanged to avoid NaN).
pub fn normalize(a: &[f32]) -> Vec<f64> {
    let n = norm(a);
    if n == 0.0 {
        return a.iter().map(|x| *x as f64).collect();
    }
    a.iter().map(|x| *x as f64 / n).collect()
}

/// Squared Euclidean distance.
#[inline]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum()
}
