//! Sparse-autoencoder forward pass + reconstruction for Level-2 (feature) attribution.
//!
//! We keep the full encoder and decoder so that feature attribution can decompose the
//! *actual* reconstruction dot product `dot(recon_q, recon_c)` exactly — capturing decoder
//! cross-terms and the decoder bias — rather than a diagonal approximation.

/// Weights in feature-major row layout: `w_enc[f*dim..]` is feature f's encoder row and
/// `w_dec[f*dim..]` is its decoder atom `d_f`.
pub struct Sae {
    pub dim: usize,
    pub n_features: usize,
    pub w_enc: Vec<f32>,
    pub b_enc: Vec<f32>,
    pub w_dec: Vec<f32>, // [n_features * dim], row f = decoder atom d_f
    pub b_dec: Vec<f32>, // [dim]
    pub dec_norm2: Vec<f64>, // ‖d_f‖² (derived), used by dissimilarity weighting
    pub names: Vec<Option<String>>,
    pub conf: Vec<Option<f64>>,
}

impl Sae {
    pub fn new(
        dim: usize,
        n_features: usize,
        w_enc: Vec<f32>,
        b_enc: Vec<f32>,
        w_dec: Vec<f32>,
        b_dec: Vec<f32>,
    ) -> Self {
        assert_eq!(w_enc.len(), dim * n_features, "w_enc shape");
        assert_eq!(b_enc.len(), n_features, "b_enc shape");
        assert_eq!(w_dec.len(), dim * n_features, "w_dec shape");
        assert_eq!(b_dec.len(), dim, "b_dec shape");
        let dec_norm2 = (0..n_features)
            .map(|f| {
                w_dec[f * dim..(f + 1) * dim]
                    .iter()
                    .map(|x| *x as f64 * *x as f64)
                    .sum()
            })
            .collect();
        Sae {
            dim,
            n_features,
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            dec_norm2,
            names: vec![None; n_features],
            conf: vec![None; n_features],
        }
    }

    #[inline]
    pub fn dec_row(&self, f: usize) -> &[f32] {
        &self.w_dec[f * self.dim..(f + 1) * self.dim]
    }

    /// `a_f(x) = ReLU(W_enc · x + b_enc)`.
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

    /// Reconstruct the embedding: `x̂ = Σ_f a_f · d_f + b_dec`.
    pub fn recon(&self, a: &[f64]) -> Vec<f64> {
        let mut r: Vec<f64> = self.b_dec.iter().map(|x| *x as f64).collect();
        for f in 0..self.n_features {
            let af = a[f];
            if af == 0.0 {
                continue;
            }
            let d = self.dec_row(f);
            for i in 0..self.dim {
                r[i] += af * d[i] as f64;
            }
        }
        r
    }
}
