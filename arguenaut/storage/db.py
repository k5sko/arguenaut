"""SQLite metadata store: hypotheses, perspectives, axes, axis verification runs."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL UNIQUE,
    topic        TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS perspectives (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    hypothesis_id INTEGER NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    stance        TEXT NOT NULL,   -- e.g. "strong_agree", "qualified_disagree"
    text          TEXT NOT NULL,
    position      INTEGER NOT NULL,  -- 0..N-1 within a hypothesis
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(hypothesis_id, position)
);

CREATE TABLE IF NOT EXISTS axes (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    layer             INTEGER NOT NULL,
    component_idx     INTEGER NOT NULL,
    label             TEXT,             -- "X vs Y" — null until labeled
    high_pole         TEXT,
    low_pole          TEXT,
    confidence        REAL,             -- final verification correlation
    explained_var     REAL,
    refinement_rounds INTEGER DEFAULT 0,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(layer, component_idx)
);

CREATE TABLE IF NOT EXISTS axis_verifications (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    axis_id     INTEGER NOT NULL REFERENCES axes(id) ON DELETE CASCADE,
    round       INTEGER NOT NULL,
    candidate   TEXT NOT NULL,           -- candidate label tried this round
    score       REAL NOT NULL,
    detail_json TEXT,                    -- predicted vs actual rankings, counterexamples
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_perspectives_hyp ON perspectives(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_axes_layer ON axes(layer);
"""


@dataclass
class Hypothesis:
    id: int
    text: str
    topic: str | None


@dataclass
class Perspective:
    id: int
    hypothesis_id: int
    stance: str
    text: str
    position: int


@dataclass
class Axis:
    id: int
    layer: int
    component_idx: int
    label: str | None
    high_pole: str | None
    low_pole: str | None
    confidence: float | None
    explained_var: float | None
    refinement_rounds: int


class Database:
    """Thin sqlite3 wrapper. One connection per Database instance.

    Use `with Database.open(path) as db:` or instantiate then call `.close()`.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, isolation_level=None)  # autocommit
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self.conn.execute("PRAGMA journal_mode = WAL;")
        self.conn.executescript(SCHEMA)

    # ── lifecycle ──────────────────────────────────────────────────────────
    @classmethod
    @contextmanager
    def open(cls, path: str | Path) -> Iterator["Database"]:
        db = cls(path)
        try:
            yield db
        finally:
            db.close()

    def close(self) -> None:
        self.conn.close()

    # ── hypotheses ─────────────────────────────────────────────────────────
    def add_hypothesis(self, text: str, topic: str | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO hypotheses (text, topic) VALUES (?, ?) "
            "ON CONFLICT(text) DO UPDATE SET topic = COALESCE(excluded.topic, hypotheses.topic) "
            "RETURNING id",
            (text, topic),
        )
        return cur.fetchone()[0]

    def get_hypothesis(self, hypothesis_id: int) -> Hypothesis | None:
        row = self.conn.execute(
            "SELECT id, text, topic FROM hypotheses WHERE id = ?", (hypothesis_id,)
        ).fetchone()
        return Hypothesis(**dict(row)) if row else None

    def list_hypotheses(self) -> list[Hypothesis]:
        rows = self.conn.execute("SELECT id, text, topic FROM hypotheses ORDER BY id").fetchall()
        return [Hypothesis(**dict(r)) for r in rows]

    # ── perspectives ───────────────────────────────────────────────────────
    def add_perspective(
        self, hypothesis_id: int, stance: str, text: str, position: int
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO perspectives (hypothesis_id, stance, text, position) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(hypothesis_id, position) DO UPDATE SET stance=excluded.stance, text=excluded.text "
            "RETURNING id",
            (hypothesis_id, stance, text, position),
        )
        return cur.fetchone()[0]

    def get_perspective(self, perspective_id: int) -> Perspective | None:
        row = self.conn.execute(
            "SELECT id, hypothesis_id, stance, text, position FROM perspectives WHERE id = ?",
            (perspective_id,),
        ).fetchone()
        return Perspective(**dict(row)) if row else None

    def list_perspectives(self, hypothesis_id: int | None = None) -> list[Perspective]:
        if hypothesis_id is None:
            rows = self.conn.execute(
                "SELECT id, hypothesis_id, stance, text, position FROM perspectives "
                "ORDER BY hypothesis_id, position"
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT id, hypothesis_id, stance, text, position FROM perspectives "
                "WHERE hypothesis_id = ? ORDER BY position",
                (hypothesis_id,),
            ).fetchall()
        return [Perspective(**dict(r)) for r in rows]

    # ── axes ───────────────────────────────────────────────────────────────
    def upsert_axis(
        self,
        layer: int,
        component_idx: int,
        explained_var: float,
        label: str | None = None,
        high_pole: str | None = None,
        low_pole: str | None = None,
        confidence: float | None = None,
        refinement_rounds: int = 0,
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO axes (layer, component_idx, explained_var, label, high_pole, low_pole,
                              confidence, refinement_rounds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(layer, component_idx) DO UPDATE SET
                explained_var     = excluded.explained_var,
                label             = COALESCE(excluded.label, axes.label),
                high_pole         = COALESCE(excluded.high_pole, axes.high_pole),
                low_pole          = COALESCE(excluded.low_pole, axes.low_pole),
                confidence        = COALESCE(excluded.confidence, axes.confidence),
                refinement_rounds = excluded.refinement_rounds
            RETURNING id
            """,
            (
                layer,
                component_idx,
                explained_var,
                label,
                high_pole,
                low_pole,
                confidence,
                refinement_rounds,
            ),
        )
        return cur.fetchone()[0]

    def list_axes(self, layer: int | None = None) -> list[Axis]:
        sql = (
            "SELECT id, layer, component_idx, label, high_pole, low_pole, confidence, "
            "explained_var, refinement_rounds FROM axes"
        )
        params: tuple = ()
        if layer is not None:
            sql += " WHERE layer = ?"
            params = (layer,)
        sql += " ORDER BY layer, component_idx"
        rows = self.conn.execute(sql, params).fetchall()
        return [Axis(**dict(r)) for r in rows]

    def add_axis_verification(
        self, axis_id: int, round_: int, candidate: str, score: float, detail: dict | None = None
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO axis_verifications (axis_id, round, candidate, score, detail_json) "
            "VALUES (?, ?, ?, ?, ?) RETURNING id",
            (axis_id, round_, candidate, score, json.dumps(detail) if detail else None),
        )
        return cur.fetchone()[0]

    def list_axis_verifications(self, axis_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT round, candidate, score, detail_json, created_at FROM axis_verifications "
            "WHERE axis_id = ? ORDER BY round",
            (axis_id,),
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["detail"] = json.loads(d.pop("detail_json")) if d["detail_json"] else None
            out.append(d)
        return out
