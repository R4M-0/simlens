//! Sparse-autoencoder forward pass for Level-2 (feature) attribution.
//!
//! We only need the *encoder* (to get sparse feature activations) and the squared
//! decoder-column norms `‖d_f‖²` (to weight each shared feature's contribution to the
//! dot product of reconstructions). The full decoder is not needed at attribution time.

/// Encoder weights in row-major layout: `w_enc[f * dim .. (f+1)*dim]` is feature `f`'s row.
pub struct Sae {
    pub dim: usize,
    pub n_features: usize,
    pub w_enc: Vec<f32>,
    pub b_enc: Vec<f32>,
    /// Precomputed ‖d_f‖² for each feature (from the decoder columns).
    pub dec_norm2: Vec<f64>,
    /// Optional human names / confidences per feature.
    pub names: Vec<Option<String>>,
    pub conf: Vec<Option<f64>>,
}

impl Sae {
    pub fn new(
        dim: usize,
        n_features: usize,
        w_enc: Vec<f32>,
        b_enc: Vec<f32>,
        dec_norm2: Vec<f64>,
    ) -> Self {
        assert_eq!(w_enc.len(), dim * n_features, "w_enc shape");
        assert_eq!(b_enc.len(), n_features, "b_enc shape");
        assert_eq!(dec_norm2.len(), n_features, "dec_norm2 shape");
        Sae {
            dim,
            n_features,
            w_enc,
            b_enc,
            dec_norm2,
            names: vec![None; n_features],
            conf: vec![None; n_features],
        }
    }

    /// `a_f(x) = ReLU(W_enc · x + b_enc)`. Returns a length-`n_features` activation vector.
    pub fn encode(&self, x: &[f32]) -> Vec<f64> {
        debug_assert_eq!(x.len(), self.dim);
        let mut out = vec![0.0f64; self.n_features];
        for f in 0..self.n_features {
            let row = &self.w_enc[f * self.dim..(f + 1) * self.dim];
            let mut acc = self.b_enc[f] as f64;
            for i in 0..self.dim {
                acc += row[i] as f64 * x[i] as f64;
            }
            out[f] = if acc > 0.0 { acc } else { 0.0 };
        }
        out
    }
}
