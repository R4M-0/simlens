//! Concept Activation Vectors for Level-3 (named-concept) attribution and steering.

use crate::linalg;

/// A registry of unit-norm concept directions, each with a human name.
pub struct CavSet {
    pub dim: usize,
    pub names: Vec<String>,
    /// Row-major `dirs[k*dim .. (k+1)*dim]` = concept `k`'s unit direction.
    pub dirs: Vec<f32>,
    pub conf: Vec<f64>,
}

impl CavSet {
    pub fn new(dim: usize, names: Vec<String>, dirs: Vec<f32>, conf: Vec<f64>) -> Self {
        let k = names.len();
        assert_eq!(dirs.len(), dim * k, "dirs shape");
        assert_eq!(conf.len(), k, "conf shape");
        CavSet {
            dim,
            names,
            dirs,
            conf,
        }
    }

    pub fn k(&self) -> usize {
        self.names.len()
    }

    pub fn dir(&self, k: usize) -> &[f32] {
        &self.dirs[k * self.dim..(k + 1) * self.dim]
    }

    /// Projection of a vector onto concept `k`'s direction.
    pub fn project(&self, x: &[f32], k: usize) -> f64 {
        linalg::dot(x, self.dir(k))
    }
}
