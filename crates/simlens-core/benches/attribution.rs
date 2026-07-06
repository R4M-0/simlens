//! Criterion micro-benchmarks for the attribution kernels (T1.3).
//! Run with `cargo bench -p simlens-core`.

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use simlens_core::{
    explain_l1, explain_l2, explain_l3, raw_score, CavSet, ExplainConfig, Level, Metric, Sae,
};

fn vecs(dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    // cheap deterministic pseudo-random vectors (no rng dependency)
    let mut s = seed
        .wrapping_mul(2862933555777941757)
        .wrapping_add(3037000493);
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 33) as f32 / u32::MAX as f32) - 0.5
    };
    let q = (0..dim).map(|_| next()).collect();
    let c = (0..dim).map(|_| next()).collect();
    (q, c)
}

fn make_sae(dim: usize, nf: usize, k: usize) -> Sae {
    let (mut w_enc, mut w_dec) = (Vec::with_capacity(dim * nf), Vec::with_capacity(dim * nf));
    let mut s = 12345u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 33) as f32 / u32::MAX as f32) - 0.5
    };
    for _ in 0..dim * nf {
        w_enc.push(next());
        w_dec.push(next());
    }
    Sae::new(dim, nf, w_enc, vec![0.0; nf], w_dec, vec![0.0; dim]).with_gates(k, Vec::new())
}

fn cfg() -> ExplainConfig {
    ExplainConfig {
        level: Level::Dim,
        top_k: 8,
        min_abs: 0.0,
    }
}

fn bench(c: &mut Criterion) {
    let cfg = cfg();
    for &dim in &[384usize, 768, 1536] {
        let (q, v) = vecs(dim, dim as u64);
        c.bench_function(&format!("raw_score/cosine/{dim}"), |b| {
            b.iter(|| raw_score(black_box(&q), black_box(&v), Metric::Cosine))
        });
        c.bench_function(&format!("explain_l1/cosine/{dim}"), |b| {
            b.iter(|| explain_l1(black_box(&q), black_box(&v), Metric::Cosine, &cfg))
        });

        let nf = dim * 8;
        let sae = make_sae(dim, nf, 32);
        c.bench_function(&format!("sae_encode/{dim}x{nf}/k32"), |b| {
            b.iter(|| sae.encode(black_box(&q)))
        });
        c.bench_function(&format!("explain_l2/{dim}x{nf}/k32"), |b| {
            b.iter(|| explain_l2(&sae, black_box(&q), black_box(&v), Metric::Cosine, &cfg))
        });

        // 64 concept directions
        let dirs: Vec<f32> = (0..dim * 64).map(|i| (i as f32).sin()).collect();
        let names = (0..64).map(|i| format!("c{i}")).collect();
        let cav = CavSet::new(dim, names, dirs, vec![0.9; 64]);
        c.bench_function(&format!("explain_l3/{dim}x64"), |b| {
            b.iter_batched(
                || (q.clone(), v.clone()),
                |(q, v)| explain_l3(&cav, &q, &v, Metric::Cosine, &cfg),
                BatchSize::SmallInput,
            )
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
