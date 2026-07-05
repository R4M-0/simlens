//! # simlens-core
//!
//! Faithful, vector-only similarity/ranking attribution. Given two embedding vectors and
//! a metric, decompose the similarity score into additive contributions at three zoom
//! levels — dimensions (exact), SAE features, and named concepts — each carrying a
//! completeness residual so consumers can see exactly how faithful the explanation is.

pub mod attribute;
pub mod cav;
pub mod linalg;
pub mod sae;
pub mod types;

pub use attribute::{
    ablate, explain_l1, explain_l2, explain_l3, explain_margin_l1, raw_score, steer, Ablation,
};
pub use cav::CavSet;
pub use sae::Sae;
pub use types::{Attribution, Contribution, ExplainConfig, Level, Metric, Polarity};

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> ExplainConfig {
        ExplainConfig {
            top_k: 1000,
            ..Default::default()
        }
    }

    #[test]
    fn dot_l1_is_exact() {
        let q = vec![1.0f32, 2.0, 3.0];
        let c = vec![0.5f32, -1.0, 2.0];
        let a = explain_l1(&q, &c, Metric::Dot, &cfg());
        assert!((a.score - (0.5 - 2.0 + 6.0)).abs() < 1e-6);
        // completeness: Σφ == score
        assert!(a.completeness_residual < 1e-9, "residual {}", a.completeness_residual);
        let sum: f64 = a.contributions.iter().map(|x| x.value).sum();
        assert!((sum - a.score).abs() < 1e-9);
    }

    #[test]
    fn cosine_l1_is_exact_and_bounded() {
        let q = vec![1.0f32, 0.0, 1.0];
        let c = vec![1.0f32, 1.0, 0.0];
        let a = explain_l1(&q, &c, Metric::Cosine, &cfg());
        assert!((a.score - 0.5).abs() < 1e-6, "cos {}", a.score);
        assert!(a.completeness_residual < 1e-9);
    }

    #[test]
    fn euclidean_l1_sums_to_sqdist() {
        let q = vec![0.0f32, 0.0];
        let c = vec![3.0f32, 4.0];
        let a = explain_l1(&q, &c, Metric::Euclidean, &cfg());
        assert!((a.score - 25.0).abs() < 1e-6);
        assert!(a.completeness_residual < 1e-9);
    }

    #[test]
    fn margin_sums_to_score_difference() {
        let q = vec![1.0f32, 1.0, 0.0];
        let better = vec![1.0f32, 1.0, 0.0];
        let worse = vec![0.0f32, 1.0, 1.0];
        let a = explain_margin_l1(&q, &better, &worse, Metric::Dot, &cfg());
        let expect = raw_score(&q, &better, Metric::Dot) - raw_score(&q, &worse, Metric::Dot);
        assert!((a.score - expect).abs() < 1e-9);
        assert!(a.completeness_residual < 1e-9);
    }

    #[test]
    fn sae_encode_relu_and_l2_shape() {
        // 2 features over dim 3. Feature 0 fires on dim0, feature 1 on dim2 (negative bias).
        let w_enc = vec![
            1.0, 0.0, 0.0, // f0
            0.0, 0.0, 1.0, // f1
        ];
        let b_enc = vec![0.0f32, -10.0]; // f1 suppressed
        let dec_norm2 = vec![1.0f64, 1.0];
        let sae = Sae::new(3, 2, w_enc, b_enc, dec_norm2);
        let a = sae.encode(&[2.0, 0.0, 5.0]);
        assert_eq!(a[0], 2.0);
        assert_eq!(a[1], 0.0); // 5 - 10 < 0 → relu 0

        let q = vec![2.0f32, 0.0, 1.0];
        let c = vec![3.0f32, 0.0, 1.0];
        let attr = explain_l2(&sae, &q, &c, Metric::Dot, &cfg());
        // only feature 0 is active in both → one contribution = 2*3*1 = 6
        assert_eq!(attr.contributions.len(), 1);
        assert!((attr.contributions[0].value - 6.0).abs() < 1e-9);
        assert_eq!(attr.contributions[0].polarity, Polarity::Shared);
    }

    #[test]
    fn ablation_drops_score() {
        let w_enc = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let b_enc = vec![0.0f32, 0.0];
        let dec_norm2 = vec![1.0f64, 1.0];
        let sae = Sae::new(3, 2, w_enc, b_enc, dec_norm2);
        let q = vec![1.0f32, 1.0, 0.0];
        let c = vec![1.0f32, 1.0, 0.0];
        let abl = ablate(&sae, &q, &c, Metric::Dot, 0.5);
        assert!(abl.dropped_below);
        assert!(abl.score_after < abl.score_before);
    }
}
