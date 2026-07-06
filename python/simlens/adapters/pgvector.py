"""pgvector adapter — fetch embedding column rows by id from PostgreSQL."""
from __future__ import annotations

import numpy as np


def _parse_vec(v) -> list:
    if isinstance(v, str):  # pgvector text form: "[1,2,3]"
        return [float(x) for x in v.strip("[]").split(",") if x]
    return list(v)


class Pgvector:
    """Read vectors from a Postgres/pgvector table using any DB-API 2.0 connection."""

    def __init__(self, conn, table: str, id_col: str = "id", vec_col: str = "embedding"):
        self.conn = conn
        self.table = table
        self.id_col = id_col
        self.vec_col = vec_col

    def get(self, ids: list) -> np.ndarray:
        ids = list(ids)
        q = (
            f"SELECT {self.id_col}, {self.vec_col} FROM {self.table} "
            f"WHERE {self.id_col} = ANY(%s)"
        )
        cur = self.conn.cursor()
        cur.execute(q, (ids,))
        by_id = {row[0]: _parse_vec(row[1]) for row in cur.fetchall()}
        cur.close()
        return np.asarray([by_id[i] for i in ids], dtype=np.float32)
