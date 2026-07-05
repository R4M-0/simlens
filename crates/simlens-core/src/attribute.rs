//! Attribution kernels: the three zoom levels, contrastive margin, ablation, steering.

use crate::cav::CavSet;
use crate::linalg;
use crate::sae::Sae;
use crate::types::*;

/// Raw similarity/distance score for a metric.
pub fn raw_score(q: &[f32], c: &[f32], metric: Metric) -> f64 {
    match metric {
        Metric::Dot => linalg::dot(q, c),
        Metric::Cosine => {
            let a = linalg::normalize(q);
            let b = linalg::normalize(c);
            linalg::dot64(&a, &b)
        }
        Metric::Euclidean => linalg::sq_dist(q, c),
    }
}

fn to_f32(v: &[f64]) -> Vec<f32> {
    v.iter().map(|x| *x as f32).collect()
}

/// The vectors actually used for per-coordinate work under a metric
/// (cosine works on unit vectors; others on the raw vectors). f32 path for the
/// SAE/CAV kernels (which are approximate anyway).
fn effective(q: &[f32], c: &[f32], metric: Metric) -> (Vec<f32>, Vec<f32>) {
    match metric {
        Metric::Cosine => (to_f32(&linalg::normalize(q)), to_f32(&linalg::normalize(c))),
        _ => (q.to_vec(), c.to_vec()),
    }
}

/// f64 effective vectors — keeps Level-1 / margin exact to f64 precision.
fn effective64(q: &[f32], c: &[f32], metric: Metric) -> (Vec<f64>, Vec<f64>) {
    match metric {
        Metric::Cosine => (linalg::normalize(q), linalg::normalize(c)),
        _ => (
            q.iter().map(|x| *x as f64).collect(),
            c.iter().map(|x| *x as f64).collect(),
        ),
    }
}

struct Raw {
    id: String,
    name: Option<String>,
    value: f64,
    confidence: Option<f64>,
    polarity: Polarity,
}

/// Sort, truncate, compute residual + coverage, attach honesty warnings.
fn finalize(
    mut raws: Vec<Raw>,
    score: f64,
    metric: Metric,
    level: Level,
    cfg: &ExplainConfig,
) -> Attribution {
    let total_abs: f64 = raws.iter().map(|r| r.value.abs()).sum();
    let sum_val: f64 = raws.iter().map(|r| r.value).sum();
    let residual = (score - sum_val).abs();

    raws.sort_by(|a, b| b.value.abs().partial_cmp(&a.value.abs()).unwrap());
    let shown: Vec<Raw> = raws
        .into_iter()
        .filter(|r| r.value.abs() >= cfg.min_abs)
        .take(cfg.top_k)
        .collect();

    let shown_abs: f64 = shown.iter().map(|r| r.value.abs()).sum();
    let coverage = if total_abs > 0.0 {
        shown_abs / total_abs
    } else {
        1.0
    };

    let mut warnings = Vec::new();
    let denom = score.abs().max(1e-9);
    if level != Level::Dim && residual > 0.05 * denom {
        warnings.push(format!(
            "completeness_residual_high: Σφ deviates from score by {:.4} ({:.0}%); use level=\"dim\" for an exact decomposition",
            residual,
            100.0 * residual / denom
        ));
    }
    if level == Level::Concept {
        warnings.push(
            "partial_decomposition: concepts span a subspace, not a complete basis; a nonzero residual is expected".to_string(),
        );
    }
    if coverage < 0.5 && !shown.is_empty() {
        warnings.push(format!(
            "low_coverage: shown contributions cover {:.0}% of total magnitude; raise top_k",
            100.0 * coverage
        ));
    }

    let contributions = shown
        .into_iter()
        .map(|r| Contribution {
            id: r.id,
            name: r.name,
            value: r.value,
            confidence: r.confidence,
            polarity: r.polarity,
        })
        .collect();

    Attribution {
        score,
        metric,
        level,
        contributions,
        completeness_residual: residual,
        coverage,
        bundle_hash: None,
        warnings,
    }
}

/// Level 1 — exact per-dimension decomposition (no training required).
pub fn explain_l1(q: &[f32], c: &[f32], metric: Metric, cfg: &ExplainConfig) -> Attribution {
    let (qe, ce) = effective64(q, c, metric);
    let mut raws = Vec::with_capacity(q.len());
    for i in 0..q.len() {
        let value = match metric {
            Metric::Euclidean => {
                let d = qe[i] - ce[i];
                d * d
            }
            _ => qe[i] * ce[i],
        };
        let pol = Polarity::from_activity(q[i] != 0.0, c[i] != 0.0);
        raws.push(Raw {
            id: format!("dim:{i}"),
            name: None,
            value,
            confidence: None,
            polarity: pol,
        });
    }
    let score = raw_score(q, c, metric);
    finalize(raws, score, metric, Level::Dim, cfg)
}

/// Level 2 — shared SAE feature attribution.
pub fn explain_l2(
    sae: &Sae,
    q: &[f32],
    c: &[f32],
    metric: Metric,
    cfg: &ExplainConfig,
) -> Attribution {
    let (qe, ce) = effective(q, c, metric);
    let aq = sae.encode(&qe);
    let ac = sae.encode(&ce);
    let mut raws = Vec::new();
    for f in 0..sae.n_features {
        let value = aq[f] * ac[f] * sae.dec_norm2[f];
        if value == 0.0 {
            continue; // only features active in *both* drive the reconstruction dot
        }
        raws.push(Raw {
            id: format!("feat:{f}"),
            name: sae.names[f].clone(),
            value,
            confidence: sae.conf[f],
            polarity: Polarity::from_activity(aq[f] > 0.0, ac[f] > 0.0),
        });
    }
    let score = raw_score(q, c, metric);
    finalize(raws, score, metric, Level::Feature, cfg)
}

/// Level 3 — named-concept (CAV) attribution. Partial by construction.
pub fn explain_l3(
    cavs: &CavSet,
    q: &[f32],
    c: &[f32],
    metric: Metric,
    cfg: &ExplainConfig,
) -> Attribution {
    let (qe, ce) = effective(q, c, metric);
    let mut raws = Vec::new();
    for k in 0..cavs.k() {
        let pq = cavs.project(&qe, k);
        let pc = cavs.project(&ce, k);
        let value = pq * pc;
        raws.push(Raw {
            id: format!("concept:{}", cavs.names[k]),
            name: Some(cavs.names[k].clone()),
            value,
            confidence: Some(cavs.conf[k]),
            polarity: Polarity::from_activity(pq.abs() > 1e-9, pc.abs() > 1e-9),
        });
    }
    let score = raw_score(q, c, metric);
    finalize(raws, score, metric, Level::Concept, cfg)
}

/// Contrastive margin at the dimension level: why `better` outranks `worse`.
pub fn explain_margin_l1(
    q: &[f32],
    better: &[f32],
    worse: &[f32],
    metric: Metric,
    cfg: &ExplainConfig,
) -> Attribution {
    let (qb, bb) = effective64(q, better, metric);
    let (qw, ww) = effective64(q, worse, metric);
    let mut raws = Vec::with_capacity(q.len());
    for i in 0..q.len() {
        let (vb, vw) = match metric {
            Metric::Euclidean => {
                let db = qb[i] - bb[i];
                let dw = qw[i] - ww[i];
                (db * db, dw * dw)
            }
            _ => (qb[i] * bb[i], qw[i] * ww[i]),
        };
        raws.push(Raw {
            id: format!("dim:{i}"),
            name: None,
            value: vb - vw,
            confidence: None,
            polarity: Polarity::Shared,
        });
    }
    let score = raw_score(q, better, metric) - raw_score(q, worse, metric);
    finalize(raws, score, metric, Level::Dim, cfg)
}

pub struct Ablation {
    pub removed: Vec<Contribution>,
    pub score_before: f64,
    pub score_after: f64,
    pub dropped_below: bool,
}

/// Greedy minimal set of shared features whose removal drops the score below `threshold`.
pub fn ablate(
    sae: &Sae,
    q: &[f32],
    c: &[f32],
    metric: Metric,
    threshold: f64,
) -> Ablation {
    let (qe, ce) = effective(q, c, metric);
    let aq = sae.encode(&qe);
    let ac = sae.encode(&ce);
    let mut feats: Vec<(usize, f64)> = (0..sae.n_features)
        .map(|f| (f, aq[f] * ac[f] * sae.dec_norm2[f]))
        .filter(|(_, v)| *v > 0.0)
        .collect();
    feats.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let score_before = raw_score(q, c, metric);
    let mut running = score_before;
    let mut removed = Vec::new();
    for (f, v) in feats {
        if running < threshold {
            break;
        }
        running -= v;
        removed.push(Contribution {
            id: format!("feat:{f}"),
            name: sae.names[f].clone(),
            value: v,
            confidence: sae.conf[f],
            polarity: Polarity::Shared,
        });
    }
    Ablation {
        removed,
        score_before,
        score_after: running,
        dropped_below: running < threshold,
    }
}

/// Steer a query vector along named concept directions: `q' = q + Σ wₖ·dirₖ`.
pub fn steer(q: &[f32], cavs: &CavSet, weights: &[(usize, f64)]) -> Vec<f32> {
    let mut out: Vec<f32> = q.to_vec();
    for &(k, w) in weights {
        let dir = cavs.dir(k);
        for i in 0..out.len() {
            out[i] += (w * dir[i] as f64) as f32;
        }
    }
    out
}
