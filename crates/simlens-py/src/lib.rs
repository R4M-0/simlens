//! PyO3 bindings exposing simlens-core as `simlens._native`.
//!
//! Large weight matrices cross the boundary as little-endian f32 `bytes` (fast, no numpy
//! dependency); per-query vectors cross as plain lists. Results are returned as plain
//! dicts, which the Python layer reshapes into typed dataclasses.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use simlens_core as sc;
use simlens_core::{Attribution, CavSet, ExplainConfig, Metric, Sae};

/// Contiguous f32 slice from a numpy array (zero-copy view; errors if non-contiguous).
fn slice1<'a>(a: &'a PyReadonlyArray1<'a, f32>) -> PyResult<&'a [f32]> {
    a.as_slice()
        .map_err(|_| pyo3::exceptions::PyValueError::new_err("array must be C-contiguous"))
}

/// Guard a vector length against the SAE dimension (prevents out-of-bounds reads in `encode`).
fn check_dim(len: usize, dim: usize) -> PyResult<()> {
    if len != dim {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "vector length {len} != SAE dim {dim}"
        )));
    }
    Ok(())
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn metric(s: &str) -> PyResult<Metric> {
    Metric::parse(s)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown metric '{s}'")))
}

fn cfg(top_k: usize, min_abs: f64) -> ExplainConfig {
    ExplainConfig {
        level: sc::Level::Dim,
        top_k,
        min_abs,
    }
}

fn attr_to_py(py: Python<'_>, a: &Attribution) -> PyResult<PyObject> {
    let contribs = PyList::empty_bound(py);
    for con in &a.contributions {
        let d = PyDict::new_bound(py);
        d.set_item("id", &con.id)?;
        d.set_item("name", con.name.clone())?;
        d.set_item("value", con.value)?;
        d.set_item("confidence", con.confidence)?;
        d.set_item("polarity", con.polarity.as_str())?;
        contribs.append(d)?;
    }
    let out = PyDict::new_bound(py);
    out.set_item("score", a.score)?;
    out.set_item("metric", a.metric.as_str())?;
    out.set_item("level", a.level.as_str())?;
    out.set_item("contributions", contribs)?;
    out.set_item("completeness_residual", a.completeness_residual)?;
    out.set_item("coverage", a.coverage)?;
    out.set_item("bundle_hash", a.bundle_hash.clone())?;
    out.set_item("warnings", a.warnings.clone())?;
    Ok(out.into())
}

#[pyfunction]
fn score(q: Vec<f32>, c: Vec<f32>, metric_s: &str) -> PyResult<f64> {
    Ok(sc::raw_score(&q, &c, metric(metric_s)?))
}

// ---- zero-copy numpy fast paths (T2.4) ----------------------------------------
// These view the numpy buffers directly instead of round-tripping through Python lists.

#[pyfunction]
fn score_np(q: PyReadonlyArray1<f32>, c: PyReadonlyArray1<f32>, metric_s: &str) -> PyResult<f64> {
    Ok(sc::raw_score(slice1(&q)?, slice1(&c)?, metric(metric_s)?))
}

#[pyfunction]
#[pyo3(signature = (q, c, metric_s, top_k=8, min_abs=0.0))]
fn explain_l1_np(
    py: Python<'_>,
    q: PyReadonlyArray1<f32>,
    c: PyReadonlyArray1<f32>,
    metric_s: &str,
    top_k: usize,
    min_abs: f64,
) -> PyResult<PyObject> {
    let a = sc::explain_l1(
        slice1(&q)?,
        slice1(&c)?,
        metric(metric_s)?,
        &cfg(top_k, min_abs),
    );
    attr_to_py(py, &a)
}

#[pyfunction]
#[pyo3(signature = (q, better, worse, metric_s, top_k=8, min_abs=0.0))]
fn explain_margin_np(
    py: Python<'_>,
    q: PyReadonlyArray1<f32>,
    better: PyReadonlyArray1<f32>,
    worse: PyReadonlyArray1<f32>,
    metric_s: &str,
    top_k: usize,
    min_abs: f64,
) -> PyResult<PyObject> {
    let a = sc::explain_margin_l1(
        slice1(&q)?,
        slice1(&better)?,
        slice1(&worse)?,
        metric(metric_s)?,
        &cfg(top_k, min_abs),
    );
    attr_to_py(py, &a)
}

#[pyfunction]
#[pyo3(signature = (q, c, metric_s, top_k=8, min_abs=0.0))]
fn explain_l1(
    py: Python<'_>,
    q: Vec<f32>,
    c: Vec<f32>,
    metric_s: &str,
    top_k: usize,
    min_abs: f64,
) -> PyResult<PyObject> {
    let a = sc::explain_l1(&q, &c, metric(metric_s)?, &cfg(top_k, min_abs));
    attr_to_py(py, &a)
}

#[pyfunction]
#[pyo3(signature = (q, better, worse, metric_s, top_k=8, min_abs=0.0))]
fn explain_margin(
    py: Python<'_>,
    q: Vec<f32>,
    better: Vec<f32>,
    worse: Vec<f32>,
    metric_s: &str,
    top_k: usize,
    min_abs: f64,
) -> PyResult<PyObject> {
    let a = sc::explain_margin_l1(&q, &better, &worse, metric(metric_s)?, &cfg(top_k, min_abs));
    attr_to_py(py, &a)
}

#[pyclass]
struct PySae {
    inner: Sae,
}

#[pymethods]
impl PySae {
    #[new]
    #[pyo3(signature = (dim, n_features, w_enc, b_enc, w_dec, b_dec, k=0, threshold=None))]
    fn new(
        dim: usize,
        n_features: usize,
        w_enc: Vec<u8>,
        b_enc: Vec<u8>,
        w_dec: Vec<u8>,
        b_dec: Vec<u8>,
        k: usize,
        threshold: Option<Vec<u8>>,
    ) -> Self {
        let thr = threshold.map(|b| f32s(&b)).unwrap_or_default();
        PySae {
            inner: Sae::new(
                dim,
                n_features,
                f32s(&w_enc),
                f32s(&b_enc),
                f32s(&w_dec),
                f32s(&b_dec),
            )
            .with_gates(k, thr),
        }
    }

    /// Zero-copy construction from numpy arrays (skips the bytes round-trip).
    #[staticmethod]
    #[pyo3(signature = (dim, n_features, w_enc, b_enc, w_dec, b_dec, k=0, threshold=None))]
    fn from_numpy(
        dim: usize,
        n_features: usize,
        w_enc: PyReadonlyArray2<f32>,
        b_enc: PyReadonlyArray1<f32>,
        w_dec: PyReadonlyArray2<f32>,
        b_dec: PyReadonlyArray1<f32>,
        k: usize,
        threshold: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<Self> {
        let thr = match &threshold {
            Some(t) => slice1(t)?.to_vec(),
            None => Vec::new(),
        };
        let inner = Sae::new(
            dim,
            n_features,
            w_enc.as_slice()?.to_vec(),
            slice1(&b_enc)?.to_vec(),
            w_dec.as_slice()?.to_vec(),
            slice1(&b_dec)?.to_vec(),
        )
        .with_gates(k, thr);
        Ok(PySae { inner })
    }

    fn set_labels(&mut self, names: Vec<Option<String>>, conf: Vec<Option<f64>>) {
        self.inner.names = names;
        self.inner.conf = conf;
    }

    /// Sparse feature activations for a single vector: `relu(W_enc·x + b_enc)`.
    fn encode(&self, x: Vec<f32>) -> PyResult<Vec<f64>> {
        check_dim(x.len(), self.inner.dim)?;
        Ok(self.inner.encode(&x))
    }

    /// Zero-copy single-vector encode.
    fn encode_np(&self, x: PyReadonlyArray1<f32>) -> PyResult<Vec<f64>> {
        let s = slice1(&x)?;
        check_dim(s.len(), self.inner.dim)?;
        Ok(self.inner.encode(s))
    }

    /// Batch encode: `[n, dim]` f32 in → `[n, n_features]` f64 out (owned numpy array).
    fn encode_batch_np<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f32>,
    ) -> PyResult<Bound<'py, numpy::PyArray2<f64>>> {
        let view = x.as_array();
        let n = view.nrows();
        let dim = view.ncols();
        if dim != self.inner.dim {
            return Err(pyo3::exceptions::PyValueError::new_err("dim mismatch"));
        }
        let mut out = numpy::ndarray::Array2::<f64>::zeros((n, self.inner.n_features));
        let flat = x.as_slice()?;
        for i in 0..n {
            let row = &flat[i * dim..(i + 1) * dim];
            let a = self.inner.encode(row);
            for (f, v) in a.into_iter().enumerate() {
                out[[i, f]] = v;
            }
        }
        Ok(numpy::PyArray2::from_owned_array_bound(py, out))
    }

    #[pyo3(signature = (q, c, metric_s, top_k=8, min_abs=0.0))]
    fn explain_np(
        &self,
        py: Python<'_>,
        q: PyReadonlyArray1<f32>,
        c: PyReadonlyArray1<f32>,
        metric_s: &str,
        top_k: usize,
        min_abs: f64,
    ) -> PyResult<PyObject> {
        let (qs, cs) = (slice1(&q)?, slice1(&c)?);
        check_dim(qs.len(), self.inner.dim)?;
        check_dim(cs.len(), self.inner.dim)?;
        let a = sc::explain_l2(&self.inner, qs, cs, metric(metric_s)?, &cfg(top_k, min_abs));
        attr_to_py(py, &a)
    }

    #[getter]
    fn dec_norm2(&self) -> Vec<f64> {
        self.inner.dec_norm2.clone()
    }

    #[pyo3(signature = (q, c, metric_s, top_k=8, min_abs=0.0))]
    fn explain(
        &self,
        py: Python<'_>,
        q: Vec<f32>,
        c: Vec<f32>,
        metric_s: &str,
        top_k: usize,
        min_abs: f64,
    ) -> PyResult<PyObject> {
        check_dim(q.len(), self.inner.dim)?;
        check_dim(c.len(), self.inner.dim)?;
        let a = sc::explain_l2(&self.inner, &q, &c, metric(metric_s)?, &cfg(top_k, min_abs));
        attr_to_py(py, &a)
    }

    fn ablate(
        &self,
        py: Python<'_>,
        q: Vec<f32>,
        c: Vec<f32>,
        metric_s: &str,
        threshold: f64,
    ) -> PyResult<PyObject> {
        check_dim(q.len(), self.inner.dim)?;
        check_dim(c.len(), self.inner.dim)?;
        let abl = sc::ablate(&self.inner, &q, &c, metric(metric_s)?, threshold);
        let removed = PyList::empty_bound(py);
        for con in &abl.removed {
            let d = PyDict::new_bound(py);
            d.set_item("id", &con.id)?;
            d.set_item("name", con.name.clone())?;
            d.set_item("value", con.value)?;
            removed.append(d)?;
        }
        let out = PyDict::new_bound(py);
        out.set_item("removed", removed)?;
        out.set_item("score_before", abl.score_before)?;
        out.set_item("score_after", abl.score_after)?;
        out.set_item("dropped_below", abl.dropped_below)?;
        Ok(out.into())
    }
}

#[pyclass]
struct PyCavSet {
    inner: CavSet,
}

#[pymethods]
impl PyCavSet {
    #[new]
    fn new(dim: usize, names: Vec<String>, dirs: Vec<u8>, conf: Vec<f64>) -> Self {
        PyCavSet {
            inner: CavSet::new(dim, names, f32s(&dirs), conf),
        }
    }

    #[pyo3(signature = (q, c, metric_s, top_k=8, min_abs=0.0))]
    fn explain(
        &self,
        py: Python<'_>,
        q: Vec<f32>,
        c: Vec<f32>,
        metric_s: &str,
        top_k: usize,
        min_abs: f64,
    ) -> PyResult<PyObject> {
        let a = sc::explain_l3(&self.inner, &q, &c, metric(metric_s)?, &cfg(top_k, min_abs));
        attr_to_py(py, &a)
    }

    fn steer(&self, q: Vec<f32>, weights: Vec<(usize, f64)>) -> Vec<f32> {
        sc::steer(&q, &self.inner, &weights)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(score_np, m)?)?;
    m.add_function(wrap_pyfunction!(explain_l1, m)?)?;
    m.add_function(wrap_pyfunction!(explain_l1_np, m)?)?;
    m.add_function(wrap_pyfunction!(explain_margin, m)?)?;
    m.add_function(wrap_pyfunction!(explain_margin_np, m)?)?;
    m.add_class::<PySae>()?;
    m.add_class::<PyCavSet>()?;
    Ok(())
}
