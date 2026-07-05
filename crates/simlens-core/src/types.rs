//! Language-neutral core types (mirrored in python/simlens/types.py and proto).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    Dot,
    Cosine,
    Euclidean,
}

impl Metric {
    pub fn parse(s: &str) -> Option<Metric> {
        match s.to_ascii_lowercase().as_str() {
            "dot" | "dotproduct" | "ip" => Some(Metric::Dot),
            "cosine" | "cos" => Some(Metric::Cosine),
            "euclidean" | "l2" => Some(Metric::Euclidean),
            _ => None,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Metric::Dot => "dot",
            Metric::Cosine => "cosine",
            Metric::Euclidean => "euclidean",
        }
    }
    /// True when higher score means *more* similar (dot, cosine). Euclidean is a distance.
    pub fn higher_is_closer(&self) -> bool {
        !matches!(self, Metric::Euclidean)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Dim,
    Feature,
    Concept,
}

impl Level {
    pub fn as_str(&self) -> &'static str {
        match self {
            Level::Dim => "dim",
            Level::Feature => "feature",
            Level::Concept => "concept",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Polarity {
    Shared,
    QueryOnly,
    CandidateOnly,
    Neither,
}

impl Polarity {
    pub fn from_activity(q_active: bool, c_active: bool) -> Polarity {
        match (q_active, c_active) {
            (true, true) => Polarity::Shared,
            (true, false) => Polarity::QueryOnly,
            (false, true) => Polarity::CandidateOnly,
            (false, false) => Polarity::Neither,
        }
    }
    pub fn as_str(&self) -> &'static str {
        match self {
            Polarity::Shared => "shared",
            Polarity::QueryOnly => "query_only",
            Polarity::CandidateOnly => "candidate_only",
            Polarity::Neither => "neither",
        }
    }
}

#[derive(Clone, Debug)]
pub struct Contribution {
    pub id: String,
    pub name: Option<String>,
    pub value: f64,
    pub confidence: Option<f64>,
    pub polarity: Polarity,
}

#[derive(Clone, Debug)]
pub struct Attribution {
    pub score: f64,
    pub metric: Metric,
    pub level: Level,
    pub contributions: Vec<Contribution>,
    pub completeness_residual: f64,
    pub coverage: f64,
    pub bundle_hash: Option<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, Debug)]
pub struct ExplainConfig {
    pub level: Level,
    pub top_k: usize,
    pub min_abs: f64,
    pub include_polarity: bool,
}

impl Default for ExplainConfig {
    fn default() -> Self {
        ExplainConfig {
            level: Level::Dim,
            top_k: 8,
            min_abs: 0.0,
            include_polarity: true,
        }
    }
}
