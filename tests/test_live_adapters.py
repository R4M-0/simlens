"""Live vector-store adapter integration tests (§6 quality floor).

These exercise the real Qdrant / pgvector adapters against services from
``docker-compose.test.yml``. Connection details are read from the environment with **no
hardcoded fallbacks** (a baked-in URL/DSN is a credential leak); tests **skip** when the env
var, the client library, or the service is absent, so the default run stays hermetic:

    export SIMLENS_QDRANT_URL=http://localhost:6333
    export SIMLENS_PG_DSN=postgresql://USER:PASSWORD@localhost:5432/DB
    docker compose -f docker-compose.test.yml up -d
    pytest tests/test_live_adapters.py -v
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import simlens
from simlens.adapters import Pgvector, Qdrant

# Never default these — a hardcoded URL/DSN would leak credentials into the repo.
QDRANT_URL = os.environ.get("SIMLENS_QDRANT_URL")
PG_DSN = os.environ.get("SIMLENS_PG_DSN")


def _qdrant_client():
    qc = pytest.importorskip("qdrant_client")
    if not QDRANT_URL:
        pytest.skip("set SIMLENS_QDRANT_URL to run the live Qdrant test")
    try:
        client = qc.QdrantClient(url=QDRANT_URL, timeout=2.0)
        client.get_collections()  # forces a connection
        return client, qc
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Qdrant not reachable at {QDRANT_URL}: {e}")


def _pg_conn():
    psycopg = pytest.importorskip("psycopg")
    if not PG_DSN:
        pytest.skip("set SIMLENS_PG_DSN to run the live pgvector test")
    try:
        return psycopg.connect(PG_DSN, connect_timeout=2)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"Postgres not reachable: {e}")


@pytest.mark.integration
def test_qdrant_roundtrip_and_autofit():
    client, qc = _qdrant_client()
    from qdrant_client.models import Distance, PointStruct, VectorParams

    name = "simlens_test"
    dim = 16
    rng = np.random.default_rng(0)
    client.recreate_collection(name, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
    topics = ["finance", "sports", "medicine"]
    pts = []
    for i in range(60):
        t = i % 3
        v = rng.standard_normal(dim).astype(np.float32)
        v[t] += 4.0  # topical structure
        v /= np.linalg.norm(v)
        pts.append(PointStruct(id=i, vector=v.tolist(), payload={"topic": topics[t]}))
    client.upsert(name, pts)

    store = Qdrant(name, client=client)
    got = store.get([0, 1, 2])
    assert got.shape == (3, dim)
    bundle = simlens.autofit(store, embedder="qdrant-test", metric="cosine", expansion=3, epochs=8)
    assert bundle.n_features > 0
    assert any(n and n.startswith("topic=") for n in bundle.concept_names)


@pytest.mark.integration
def test_pgvector_roundtrip():
    conn = _pg_conn()
    dim = 8
    rng = np.random.default_rng(1)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS simlens_test")
        cur.execute(f"CREATE TABLE simlens_test (id int PRIMARY KEY, embedding vector({dim}))")
        for i in range(10):
            v = rng.standard_normal(dim).astype(np.float32)
            cur.execute("INSERT INTO simlens_test (id, embedding) VALUES (%s, %s)",
                        (i, "[" + ",".join(map(str, v.tolist())) + "]"))
        conn.commit()
    store = Pgvector(conn, "simlens_test")
    got = store.get([0, 3, 7])
    assert got.shape == (3, dim)
    # explanations run on adapter-fetched vectors
    ex = simlens.Explainer(metric="cosine")
    a = ex.explain(got[0], got[1])
    assert a.level == "dim" and not np.isnan(a.score)
    conn.close()
