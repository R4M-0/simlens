//! Rust bundle loader (feature `bundle`) — reads a `.simlens` bundle exported for Rust
//! (`manifest.json` + `weights.safetensors`) and rebuilds the kernels. This is the piece
//! that was previously Python-only; it unblocks native serving (`explain_l1`/`explain_l2`
//! straight from a loaded bundle with no Python in the loop).
//!
//! Python writes the Rust-friendly form with `Bundle.save_rust(path)`.

use std::path::Path;

use safetensors::SafeTensors;
use serde::Deserialize;

use crate::sae::Sae;

#[derive(Debug, Deserialize)]
struct EmbedderMeta {
    id: String,
    dim: usize,
    #[serde(default = "default_modality")]
    modality: String,
}

fn default_modality() -> String {
    "text".to_string()
}

#[derive(Debug, Deserialize)]
struct SaeMeta {
    #[serde(default)]
    k: usize,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    embedder: EmbedderMeta,
    #[serde(default = "default_metric")]
    metric: String,
    #[serde(default)]
    sae: Option<SaeMeta>,
    #[serde(default)]
    content_hash: Option<String>,
}

fn default_metric() -> String {
    "cosine".to_string()
}

/// A loaded bundle: metadata plus the reconstructed SAE (when present).
pub struct BundleData {
    pub embedder: String,
    pub dim: usize,
    pub modality: String,
    pub metric: String,
    pub content_hash: Option<String>,
    pub sae: Option<Sae>,
}

fn to_f32(view: &safetensors::tensor::TensorView) -> Vec<f32> {
    view.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

impl BundleData {
    /// Load from a bundle directory containing `manifest.json` and `weights.safetensors`.
    pub fn load(dir: impl AsRef<Path>) -> Result<BundleData, String> {
        let dir = dir.as_ref();
        let manifest_s =
            std::fs::read_to_string(dir.join("manifest.json")).map_err(|e| e.to_string())?;
        let m: Manifest = serde_json::from_str(&manifest_s).map_err(|e| e.to_string())?;

        let mut sae = None;
        let wpath = dir.join("weights.safetensors");
        if wpath.exists() {
            let buf = std::fs::read(&wpath).map_err(|e| e.to_string())?;
            let st = SafeTensors::deserialize(&buf).map_err(|e| e.to_string())?;
            if let Ok(w_enc) = st.tensor("w_enc") {
                let dim = m.embedder.dim;
                let n_features = w_enc.shape()[0];
                let b_enc = to_f32(&st.tensor("b_enc").map_err(|e| e.to_string())?);
                let w_dec = to_f32(&st.tensor("w_dec").map_err(|e| e.to_string())?);
                let b_dec = to_f32(&st.tensor("b_dec").map_err(|e| e.to_string())?);
                let k = m.sae.as_ref().map(|s| s.k).unwrap_or(0);
                let threshold = st
                    .tensor("sae_threshold")
                    .map(|t| to_f32(&t))
                    .unwrap_or_default();
                sae = Some(
                    Sae::new(dim, n_features, to_f32(&w_enc), b_enc, w_dec, b_dec)
                        .with_gates(k, threshold),
                );
            }
        }

        Ok(BundleData {
            embedder: m.embedder.id,
            dim: m.embedder.dim,
            modality: m.embedder.modality,
            metric: m.metric,
            content_hash: m.content_hash,
            sae,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;
    use safetensors::Dtype;

    #[test]
    fn loads_manifest_and_sae() {
        let dir = std::env::temp_dir().join(format!("simlens_bundle_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("manifest.json"),
            r#"{"embedder":{"id":"t","dim":2,"modality":"text"},"metric":"dot",
                "sae":{"k":1},"content_hash":"sha256:abc"}"#,
        )
        .unwrap();

        // a tiny 2-feature SAE over dim 2, identity-ish
        let f = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let w_enc = f(&[1.0, 0.0, 0.0, 1.0]);
        let b_enc = f(&[0.0, 0.0]);
        let w_dec = f(&[1.0, 0.0, 0.0, 1.0]);
        let b_dec = f(&[0.0, 0.0]);
        let tensors = vec![
            (
                "w_enc",
                TensorView::new(Dtype::F32, vec![2, 2], &w_enc).unwrap(),
            ),
            (
                "b_enc",
                TensorView::new(Dtype::F32, vec![2], &b_enc).unwrap(),
            ),
            (
                "w_dec",
                TensorView::new(Dtype::F32, vec![2, 2], &w_dec).unwrap(),
            ),
            (
                "b_dec",
                TensorView::new(Dtype::F32, vec![2], &b_dec).unwrap(),
            ),
        ];
        safetensors::serialize_to_file(tensors, &None, &dir.join("weights.safetensors")).unwrap();

        let b = BundleData::load(&dir).unwrap();
        assert_eq!(b.dim, 2);
        assert_eq!(b.metric, "dot");
        assert_eq!(b.content_hash.as_deref(), Some("sha256:abc"));
        let sae = b.sae.expect("sae loaded");
        assert_eq!(sae.k, 1);
        // top-1 gate: encoding [3,1] keeps only the largest feature
        assert_eq!(sae.encode(&[3.0, 1.0]), vec![3.0, 0.0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
