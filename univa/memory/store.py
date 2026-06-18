import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> Path:
    # univa/memory/store.py -> univa/memory -> univa -> repo root
    return Path(__file__).resolve().parents[2]


def _safe_project_id(project_id: str) -> str:
    project_id = (project_id or "").strip()
    if not project_id:
        raise ValueError("project_id must be non-empty")
    # Keep it filename-safe.
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", project_id)


def default_project_db_path(project_id: str) -> Path:
    pid = _safe_project_id(project_id)
    return _repo_root() / "projects" / pid / "memory.db"


def stable_segment_id(project_id: str, t_start: float, t_end: float, kind: str) -> str:
    # Float formatting keeps ids stable across runs while avoiding long strings.
    key = f"{project_id}|{kind}|{t_start:.3f}|{t_end:.3f}"
    return sha1(key.encode("utf-8")).hexdigest()


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeline_segments (
  segment_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  kind TEXT NOT NULL,                -- source/plan/target/edit
  status TEXT NOT NULL DEFAULT 'planned',
  active_clip_id TEXT,               -- points to clips.clip_id
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  UNIQUE(project_id, t_start, t_end, kind)
);
CREATE INDEX IF NOT EXISTS idx_segments_project_time ON timeline_segments(project_id, t_start, t_end);

CREATE TABLE IF NOT EXISTS clips (
  clip_id TEXT PRIMARY KEY,
  segment_id TEXT NOT NULL,
  take_index INTEGER NOT NULL,
  output_path TEXT NOT NULL,
  prompt TEXT,
  negative_prompt TEXT,
  model TEXT,
  seed INTEGER,
  params_json TEXT,                  -- JSON dict
  created_at REAL NOT NULL,
  is_active INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY(segment_id) REFERENCES timeline_segments(segment_id)
);
CREATE INDEX IF NOT EXISTS idx_clips_segment_take ON clips(segment_id, take_index);
CREATE INDEX IF NOT EXISTS idx_clips_active ON clips(segment_id, is_active);

CREATE TABLE IF NOT EXISTS beats (
  beat_id TEXT PRIMARY KEY,
  segment_id TEXT NOT NULL,
  beat_type TEXT NOT NULL,           -- e.g. "story", "camera", "dialog"
  summary TEXT,
  payload_json TEXT,                 -- JSON dict
  created_at REAL NOT NULL,
  FOREIGN KEY(segment_id) REFERENCES timeline_segments(segment_id)
);
CREATE INDEX IF NOT EXISTS idx_beats_segment ON beats(segment_id);

CREATE TABLE IF NOT EXISTS entity_states (
  state_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  entity_name TEXT NOT NULL,         -- character/scene/etc
  t_start REAL NOT NULL,
  t_end REAL NOT NULL,
  state_json TEXT NOT NULL,          -- JSON dict
  source_clip_id TEXT,               -- optional provenance
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entity_project_time ON entity_states(project_id, t_start, t_end);
CREATE INDEX IF NOT EXISTS idx_entity_name ON entity_states(project_id, entity_name);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  segment_id TEXT,
  clip_id TEXT,
  kind TEXT NOT NULL,                -- keyframe/mask/asr/caption/etc
  path TEXT NOT NULL,
  meta_json TEXT,                    -- JSON dict
  created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_lookup ON artifacts(project_id, segment_id, clip_id);

CREATE TABLE IF NOT EXISTS asset_index (
  asset_id TEXT PRIMARY KEY,
  project_id TEXT NOT NULL,
  kind TEXT NOT NULL,                -- video/image/last_frame/etc
  path TEXT NOT NULL,
  segment_id TEXT,
  clip_id TEXT,
  prompt TEXT,
  negative_prompt TEXT,
  caption TEXT,
  entity_summary TEXT,
  tags TEXT,
  created_at REAL NOT NULL,
  updated_at REAL NOT NULL,
  needs_reindex INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_asset_index_project ON asset_index(project_id, created_at);

CREATE VIRTUAL TABLE IF NOT EXISTS asset_index_fts USING fts5(
  prompt, caption, entity_summary, tags
);

CREATE TABLE IF NOT EXISTS evals (
  eval_id TEXT PRIMARY KEY,
  clip_id TEXT NOT NULL,
  consistency_score REAL,
  story_match_score REAL,
  visual_score REAL,
  note TEXT,
  created_at REAL NOT NULL,
  FOREIGN KEY(clip_id) REFERENCES clips(clip_id)
);
CREATE INDEX IF NOT EXISTS idx_evals_clip ON evals(clip_id);
"""


@dataclass
class ProjectMemoryStore:
    project_id: str
    db_path: Path
    conn: sqlite3.Connection

    @classmethod
    def open(cls, project_id: str, db_path: Optional[os.PathLike] = None) -> "ProjectMemoryStore":
        pid = _safe_project_id(project_id)
        path = Path(db_path) if db_path is not None else default_project_db_path(pid)
        path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(path), isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # Better concurrency for multi-step workflows.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")

        store = cls(project_id=pid, db_path=path, conn=conn)
        store._ensure_schema()
        # Preserve the caller-provided id for display/debugging.
        store.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)",
            ("display_project_id", str(project_id)),
        )
        return store

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    def _ensure_schema(self) -> None:
        self.conn.executescript(SCHEMA_SQL)
        # Schema versioning for future migrations.
        self.conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES(?, ?)", ("schema_version", "1"))
        # Migrate external-content FTS to contentless FTS if needed.
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='asset_index_fts'"
        ).fetchone()
        if row and row[0] and "content='asset_index'" in row[0]:
            self.conn.execute("DROP TABLE IF EXISTS asset_index_fts")
            self.conn.execute(
                "CREATE VIRTUAL TABLE asset_index_fts USING fts5(prompt, caption, entity_summary, tags)"
            )
            self.conn.execute(
                """
                INSERT INTO asset_index_fts(rowid, prompt, caption, entity_summary, tags)
                SELECT rowid, prompt, caption, entity_summary, tags FROM asset_index
                """
            )

    def _now(self) -> float:
        return time.time()

    def _j(self, obj: Any) -> Optional[str]:
        if obj is None:
            return None
        return json.dumps(obj, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    def _ju(self, s: Optional[str]) -> Any:
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return s

    def _asset_id(self, kind: str, path: str) -> str:
        key = f"{self.project_id}|{kind}|{path}"
        return sha1(key.encode("utf-8")).hexdigest()

    def _update_asset_fts(
        self,
        rowid: int,
        prompt: Optional[str],
        caption: Optional[str],
        entity_summary: Optional[str],
        tags: Optional[str],
    ) -> None:
        self.conn.execute("DELETE FROM asset_index_fts WHERE rowid=?", (rowid,))
        self.conn.execute(
            "INSERT INTO asset_index_fts(rowid, prompt, caption, entity_summary, tags) VALUES(?, ?, ?, ?, ?)",
            (
                rowid,
                prompt or "",
                caption or "",
                entity_summary or "",
                tags or "",
            ),
        )

    # --- segments ---
    def get_segment(self, segment_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM timeline_segments WHERE project_id=? AND segment_id=?",
            (self.project_id, segment_id),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def list_segments(
        self,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
        kind: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = ["project_id = ?"]
        params: List[Any] = [self.project_id]

        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        if status:
            conditions.append("status = ?")
            params.append(status)
        if t_start is not None:
            conditions.append("t_end >= ?")
            params.append(float(t_start))
        if t_end is not None:
            conditions.append("t_start <= ?")
            params.append(float(t_end))

        where = " AND ".join(conditions)
        cur = self.conn.execute(
            f"SELECT * FROM timeline_segments WHERE {where} ORDER BY t_start ASC",
            params,
        )
        return [dict(r) for r in cur.fetchall()]

    def upsert_segment(self, t_start: float, t_end: float, kind: str, status: str = "planned") -> str:
        seg_id = stable_segment_id(self.project_id, t_start, t_end, kind)
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO timeline_segments(segment_id, project_id, t_start, t_end, kind, status, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_id, t_start, t_end, kind)
            DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at
            """,
            (seg_id, self.project_id, float(t_start), float(t_end), kind, status, ts, ts),
        )
        return seg_id

    def update_segment_status(self, segment_id: str, status: str) -> bool:
        cur = self.conn.execute(
            "UPDATE timeline_segments SET status=?, updated_at=? WHERE project_id=? AND segment_id=?",
            (status, self._now(), self.project_id, segment_id),
        )
        return cur.rowcount > 0

    def get_segments_in_window(self, t0: float, t1: float) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT * FROM timeline_segments
            WHERE project_id = ?
              AND NOT (t_end < ? OR t_start > ?)
            ORDER BY t_start ASC
            """,
            (self.project_id, float(t0), float(t1)),
        )
        return [dict(r) for r in cur.fetchall()]

    def delete_segment(self, segment_id: str) -> bool:
        # Remove dependent rows first due to FK constraints.
        cur = self.conn.execute("SELECT clip_id FROM clips WHERE segment_id=?", (segment_id,))
        clip_ids = [r["clip_id"] for r in cur.fetchall()]
        if clip_ids:
            qmarks = ",".join(["?"] * len(clip_ids))
            self.conn.execute(f"DELETE FROM evals WHERE clip_id IN ({qmarks})", clip_ids)
            self.conn.execute(f"DELETE FROM artifacts WHERE clip_id IN ({qmarks})", clip_ids)

        self.conn.execute("DELETE FROM clips WHERE segment_id=?", (segment_id,))
        self.conn.execute("DELETE FROM beats WHERE segment_id=?", (segment_id,))
        self.conn.execute("DELETE FROM artifacts WHERE segment_id=?", (segment_id,))
        cur = self.conn.execute(
            "DELETE FROM timeline_segments WHERE project_id=? AND segment_id=?",
            (self.project_id, segment_id),
        )
        return cur.rowcount > 0

    def set_active_clip(self, segment_id: str, clip_id: str) -> None:
        self.conn.execute("UPDATE clips SET is_active=0 WHERE segment_id=?", (segment_id,))
        self.conn.execute("UPDATE clips SET is_active=1 WHERE clip_id=?", (clip_id,))
        self.conn.execute(
            "UPDATE timeline_segments SET active_clip_id=?, updated_at=? WHERE segment_id=?",
            (clip_id, self._now(), segment_id),
        )

    # --- clips ---
    def add_clip_take(
        self,
        segment_id: str,
        output_path: str,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        model: Optional[str] = None,
        seed: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
        make_active: bool = True,
        clip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from uuid import uuid4

        if self.get_segment(segment_id) is None:
            raise ValueError(
                f"segment_id '{segment_id}' does not exist in project '{self.project_id}'. "
                "Create the segment first with memory_upsert_segment, then save the clip take."
            )
        cur = self.conn.execute("SELECT COALESCE(MAX(take_index), -1) AS mx FROM clips WHERE segment_id=?", (segment_id,))
        take_index = int(cur.fetchone()["mx"]) + 1
        cid = clip_id or str(uuid4())
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO clips(clip_id, segment_id, take_index, output_path, prompt, negative_prompt, model, seed, params_json, created_at, is_active)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (cid, segment_id, take_index, output_path, prompt, negative_prompt, model, seed, self._j(params), ts),
        )
        if make_active:
            self.set_active_clip(segment_id=segment_id, clip_id=cid)
        return {
            "clip_id": cid,
            "segment_id": segment_id,
            "take_index": take_index,
            "output_path": output_path,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model": model,
            "seed": seed,
            "params": params or {},
            "created_at": ts,
            "is_active": 1 if make_active else 0,
        }

    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        cur = self.conn.execute("SELECT * FROM clips WHERE clip_id=?", (clip_id,))
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["params"] = self._ju(d.get("params_json"))
        d.pop("params_json", None)
        return d

    def list_clips_for_segment(self, segment_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM clips WHERE segment_id=? ORDER BY take_index ASC",
            (segment_id,),
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["params"] = self._ju(d.get("params_json"))
            d.pop("params_json", None)
            out.append(d)
        return out

    def get_clips_for_segments(self, segment_ids: Iterable[str]) -> Dict[str, List[Dict[str, Any]]]:
        ids = list(segment_ids)
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        cur = self.conn.execute(
            f"SELECT * FROM clips WHERE segment_id IN ({qmarks}) ORDER BY segment_id ASC, take_index ASC",
            ids,
        )
        out: Dict[str, List[Dict[str, Any]]] = {seg_id: [] for seg_id in ids}
        for r in cur.fetchall():
            d = dict(r)
            d["params"] = self._ju(d.get("params_json"))
            d.pop("params_json", None)
            out.setdefault(d["segment_id"], []).append(d)
        return out

    def delete_clip(self, clip_id: str) -> bool:
        cur = self.conn.execute("SELECT segment_id, is_active FROM clips WHERE clip_id=?", (clip_id,))
        row = cur.fetchone()
        if not row:
            return False
        segment_id = row["segment_id"]
        is_active = bool(row["is_active"])

        if is_active:
            cur = self.conn.execute(
                "SELECT clip_id FROM clips WHERE segment_id=? AND clip_id != ? ORDER BY take_index DESC LIMIT 1",
                (segment_id, clip_id),
            )
            next_row = cur.fetchone()
            if next_row:
                self.set_active_clip(segment_id=segment_id, clip_id=next_row["clip_id"])
            else:
                self.conn.execute(
                    "UPDATE timeline_segments SET active_clip_id=NULL, updated_at=? WHERE segment_id=?",
                    (self._now(), segment_id),
                )

        self.conn.execute("DELETE FROM evals WHERE clip_id=?", (clip_id,))
        self.conn.execute("DELETE FROM artifacts WHERE clip_id=?", (clip_id,))
        cur = self.conn.execute("DELETE FROM clips WHERE clip_id=?", (clip_id,))
        return cur.rowcount > 0

    def get_active_clips_for_segments(self, segment_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
        ids = list(segment_ids)
        if not ids:
            return {}
        qmarks = ",".join(["?"] * len(ids))
        cur = self.conn.execute(
            f"SELECT * FROM clips WHERE segment_id IN ({qmarks}) AND is_active=1",
            ids,
        )
        out: Dict[str, Dict[str, Any]] = {}
        for r in cur.fetchall():
            d = dict(r)
            d["params"] = self._ju(d.get("params_json"))
            d.pop("params_json", None)
            out[d["segment_id"]] = d
        return out

    # --- beats ---
    def add_beat(self, segment_id: str, beat_type: str, summary: str = "", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        from uuid import uuid4

        if self.get_segment(segment_id) is None:
            raise ValueError(
                f"segment_id '{segment_id}' does not exist in project '{self.project_id}'. "
                "Create the segment first with memory_upsert_segment, then attach the beat."
            )
        bid = str(uuid4())
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO beats(beat_id, segment_id, beat_type, summary, payload_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (bid, segment_id, beat_type, summary, self._j(payload), ts),
        )
        return {"beat_id": bid, "segment_id": segment_id, "beat_type": beat_type, "summary": summary, "payload": payload or {}, "created_at": ts}

    def delete_beat(self, beat_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM beats WHERE beat_id=?", (beat_id,))
        return cur.rowcount > 0

    def get_beats_for_segments(self, segment_ids: Iterable[str]) -> List[Dict[str, Any]]:
        ids = list(segment_ids)
        if not ids:
            return []
        qmarks = ",".join(["?"] * len(ids))
        cur = self.conn.execute(f"SELECT * FROM beats WHERE segment_id IN ({qmarks}) ORDER BY created_at ASC", ids)
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["payload"] = self._ju(d.get("payload_json"))
            d.pop("payload_json", None)
            out.append(d)
        return out

    # --- entity states ---
    def add_entity_state(
        self,
        entity_name: str,
        t_start: float,
        t_end: float,
        state: Dict[str, Any],
        source_clip_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        from uuid import uuid4

        sid = str(uuid4())
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO entity_states(state_id, project_id, entity_name, t_start, t_end, state_json, source_clip_id, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (sid, self.project_id, entity_name, float(t_start), float(t_end), self._j(state) or "{}", source_clip_id, ts),
        )
        return {"state_id": sid, "project_id": self.project_id, "entity_name": entity_name, "t_start": float(t_start), "t_end": float(t_end), "state": state, "source_clip_id": source_clip_id, "created_at": ts}

    def list_entity_states(
        self,
        entity_name: Optional[str] = None,
        t_start: Optional[float] = None,
        t_end: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        conditions = ["project_id = ?"]
        params: List[Any] = [self.project_id]

        if entity_name:
            conditions.append("entity_name = ?")
            params.append(entity_name)
        if t_start is not None:
            conditions.append("t_end >= ?")
            params.append(float(t_start))
        if t_end is not None:
            conditions.append("t_start <= ?")
            params.append(float(t_end))

        where = " AND ".join(conditions)
        cur = self.conn.execute(
            f"SELECT * FROM entity_states WHERE {where} ORDER BY t_start ASC, created_at ASC",
            params,
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["state"] = self._ju(d.get("state_json"))
            d.pop("state_json", None)
            out.append(d)
        return out

    def delete_entity_state(self, state_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM entity_states WHERE state_id=?", (state_id,))
        return cur.rowcount > 0

    def get_entity_states_at(self, t: float) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            """
            SELECT * FROM entity_states
            WHERE project_id = ?
              AND t_start <= ?
              AND t_end >= ?
            ORDER BY created_at DESC
            """,
            (self.project_id, float(t), float(t)),
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["state"] = self._ju(d.get("state_json"))
            d.pop("state_json", None)
            out.append(d)
        return out

    # --- asset index ---
    def upsert_asset_index(
        self,
        kind: str,
        path: str,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        prompt: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        caption: Optional[str] = None,
        entity_summary: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> str:
        asset_id = self._asset_id(kind=kind, path=path)
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO asset_index(
              asset_id, project_id, kind, path, segment_id, clip_id, prompt, negative_prompt,
              caption, entity_summary, tags, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id)
            DO UPDATE SET prompt=excluded.prompt,
                          negative_prompt=excluded.negative_prompt,
                          caption=excluded.caption,
                          entity_summary=excluded.entity_summary,
                          tags=excluded.tags,
                          updated_at=excluded.updated_at
            """,
            (
                asset_id,
                self.project_id,
                kind,
                path,
                segment_id,
                clip_id,
                prompt,
                negative_prompt,
                caption,
                entity_summary,
                tags,
                ts,
                ts,
            ),
        )
        cur = self.conn.execute("SELECT rowid FROM asset_index WHERE asset_id=?", (asset_id,))
        row = cur.fetchone()
        if row:
            try:
                self._update_asset_fts(
                    int(row["rowid"]),
                    prompt=prompt,
                    caption=caption,
                    entity_summary=entity_summary,
                    tags=tags,
                )
                self.conn.execute("UPDATE asset_index SET needs_reindex=0 WHERE asset_id=?", (asset_id,))
            except Exception:
                self.conn.execute("UPDATE asset_index SET needs_reindex=1 WHERE asset_id=?", (asset_id,))
        return asset_id

    def search_assets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        sql = """
        SELECT ai.* FROM asset_index_fts fts
        JOIN asset_index ai ON ai.rowid = fts.rowid
        WHERE asset_index_fts MATCH ?
        ORDER BY bm25(asset_index_fts) LIMIT ?
        """
        try:
            cur = self.conn.execute(sql, (query, int(limit)))
        except sqlite3.OperationalError:
            safe = query.replace('"', '""')
            cur = self.conn.execute(sql, (f"\"{safe}\"", int(limit)))
        rows = [dict(r) for r in cur.fetchall()]
        if rows:
            return rows
        if "/" in query or "." in query:
            cur = self.conn.execute(
                "SELECT * FROM asset_index WHERE path LIKE ? ORDER BY created_at DESC LIMIT ?",
                (f"%{query}%", int(limit)),
            )
            return [dict(r) for r in cur.fetchall()]
        return rows

    def update_asset_caption(
        self,
        asset_id: str,
        caption: str,
        entity_summary: Optional[str] = None,
        tags: Optional[str] = None,
    ) -> bool:
        ts = self._now()
        cur = self.conn.execute(
            """
            UPDATE asset_index
            SET caption=?,
                entity_summary=COALESCE(?, entity_summary),
                tags=COALESCE(?, tags),
                updated_at=?
            WHERE asset_id=?
            """,
            (caption, entity_summary, tags, ts, asset_id),
        )
        if cur.rowcount <= 0:
            return False
        row = self.conn.execute(
            "SELECT rowid, prompt, caption, entity_summary, tags FROM asset_index WHERE asset_id=?",
            (asset_id,),
        ).fetchone()
        if not row:
            return False
        try:
            self._update_asset_fts(
                int(row["rowid"]),
                prompt=row["prompt"],
                caption=row["caption"],
                entity_summary=row["entity_summary"],
                tags=row["tags"],
            )
            self.conn.execute("UPDATE asset_index SET needs_reindex=0 WHERE asset_id=?", (asset_id,))
        except Exception:
            self.conn.execute("UPDATE asset_index SET needs_reindex=1 WHERE asset_id=?", (asset_id,))
        return True

    # --- artifacts ---
    def add_artifact(
        self,
        kind: str,
        path: str,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        from uuid import uuid4

        aid = str(uuid4())
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO artifacts(artifact_id, project_id, segment_id, clip_id, kind, path, meta_json, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (aid, self.project_id, segment_id, clip_id, kind, path, self._j(meta), ts),
        )
        return {
            "artifact_id": aid,
            "project_id": self.project_id,
            "segment_id": segment_id,
            "clip_id": clip_id,
            "kind": kind,
            "path": path,
            "meta": meta or {},
            "created_at": ts,
        }

    def list_artifacts(
        self,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        conditions = ["project_id = ?"]
        params: List[Any] = [self.project_id]
        if segment_id:
            conditions.append("segment_id = ?")
            params.append(segment_id)
        if clip_id:
            conditions.append("clip_id = ?")
            params.append(clip_id)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        where = " AND ".join(conditions)
        cur = self.conn.execute(
            f"SELECT * FROM artifacts WHERE {where} ORDER BY created_at ASC",
            params,
        )
        out: List[Dict[str, Any]] = []
        for r in cur.fetchall():
            d = dict(r)
            d["meta"] = self._ju(d.get("meta_json"))
            d.pop("meta_json", None)
            out.append(d)
        return out

    def get_latest_artifact(
        self,
        segment_id: Optional[str] = None,
        clip_id: Optional[str] = None,
        kind: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        conditions = ["project_id = ?"]
        params: List[Any] = [self.project_id]
        if segment_id:
            conditions.append("segment_id = ?")
            params.append(segment_id)
        if clip_id:
            conditions.append("clip_id = ?")
            params.append(clip_id)
        if kind:
            conditions.append("kind = ?")
            params.append(kind)
        where = " AND ".join(conditions)
        cur = self.conn.execute(
            f"SELECT * FROM artifacts WHERE {where} ORDER BY created_at DESC LIMIT 1",
            params,
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["meta"] = self._ju(d.get("meta_json"))
        d.pop("meta_json", None)
        return d

    def delete_artifact(self, artifact_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM artifacts WHERE artifact_id=?", (artifact_id,))
        return cur.rowcount > 0

    # --- evals ---
    def add_eval(
        self,
        clip_id: str,
        consistency_score: Optional[float] = None,
        story_match_score: Optional[float] = None,
        visual_score: Optional[float] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        from uuid import uuid4

        eid = str(uuid4())
        ts = self._now()
        self.conn.execute(
            """
            INSERT INTO evals(eval_id, clip_id, consistency_score, story_match_score, visual_score, note, created_at)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (eid, clip_id, consistency_score, story_match_score, visual_score, note, ts),
        )
        return {
            "eval_id": eid,
            "clip_id": clip_id,
            "consistency_score": consistency_score,
            "story_match_score": story_match_score,
            "visual_score": visual_score,
            "note": note,
            "created_at": ts,
        }

    def list_evals(self, clip_id: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(
            "SELECT * FROM evals WHERE clip_id=? ORDER BY created_at DESC",
            (clip_id,),
        )
        return [dict(r) for r in cur.fetchall()]
