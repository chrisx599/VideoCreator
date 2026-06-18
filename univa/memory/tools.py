from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .service import ProjectMemoryService, init_project_memory


logger = logging.getLogger(__name__)

_services: Dict[str, ProjectMemoryService] = {}


def _tool_envelope(func: Callable[..., Any]) -> Callable[..., Any]:
    """Give every memory tool a consistent result shape.

    The frontend (and the agent's tool-result summarizer) keys off a ``success``
    field. Memory tools historically returned bare dicts with no such field, so
    successful calls were displayed as "Tool execution failed". This wrapper:
      - adds ``success: True`` to successful dict results (without clobbering an
        explicit value), and
      - converts exceptions (e.g. a FOREIGN KEY violation when a segment_id does
        not exist) into ``{"success": False, "error": ...}`` so the failure is
        reported clearly to both the user and the agent instead of crashing the
        tool call with an opaque traceback.

    ``functools.wraps`` preserves ``__name__``/``__doc__``/``__signature__`` so the
    agent's reflection-based tool wiring keeps working unchanged.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # surface a structured, actionable error
            logger.warning("memory tool %s failed: %s", func.__name__, exc)
            return {"success": False, "error": str(exc), "tool": func.__name__}
        if isinstance(result, dict):
            result.setdefault("success", True)
        return result

    return wrapper


def _svc(project_id: str) -> ProjectMemoryService:
    """
    Keep a per-process cache of open services. This makes repeated tool calls fast.
    """
    if project_id not in _services:
        _services[project_id] = ProjectMemoryService.open(project_id=project_id)
    return _services[project_id]


def memory_init_project(project_id: str) -> Dict[str, Any]:
    """
    Initialize a project's timeline memory DB (creates schema if missing).
    """
    return init_project_memory(project_id=project_id)


def memory_upsert_segment(
    project_id: str,
    t_start: float,
    t_end: float,
    kind: str = "target",
    status: str = "planned",
) -> Dict[str, Any]:
    """
    Create/update a timeline segment and return its segment_id.
    """
    seg_id = _svc(project_id).upsert_segment(t_start=t_start, t_end=t_end, kind=kind, status=status)
    return {"project_id": project_id, "segment_id": seg_id, "t_start": float(t_start), "t_end": float(t_end), "kind": kind, "status": status}


def memory_get_segment(project_id: str, segment_id: str) -> Dict[str, Any]:
    """
    Fetch a single segment by id.
    """
    return {"project_id": project_id, "segment": _svc(project_id).get_segment(segment_id=segment_id)}


def memory_list_segments(
    project_id: str,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "",
    status: str = "",
) -> Dict[str, Any]:
    """
    List segments with optional filters.
    """
    return {
        "project_id": project_id,
        "segments": _svc(project_id).list_segments(
            t_start=t_start,
            t_end=t_end,
            kind=kind or None,
            status=status or None,
        ),
    }


def memory_update_segment_status(project_id: str, segment_id: str, status: str) -> Dict[str, Any]:
    """
    Update a segment's status.
    """
    ok = _svc(project_id).update_segment_status(segment_id=segment_id, status=status)
    return {"project_id": project_id, "segment_id": segment_id, "status": status, "updated": ok}


def memory_delete_segment(project_id: str, segment_id: str) -> Dict[str, Any]:
    """
    Delete a segment and its dependent rows.
    """
    ok = _svc(project_id).delete_segment(segment_id=segment_id)
    return {"project_id": project_id, "segment_id": segment_id, "deleted": ok}


def memory_save_clip_take(
    project_id: str,
    segment_id: str,
    output_path: str,
    prompt: str = "",
    negative_prompt: str = "",
    model: str = "",
    seed: Optional[int] = None,
    params: Optional[Dict[str, Any]] = None,
    make_active: bool = True,
) -> Dict[str, Any]:
    """
    Save a generated clip take for a segment.
    """
    return _svc(project_id).save_clip_take(
        segment_id=segment_id,
        output_path=output_path,
        prompt=prompt or None,
        negative_prompt=negative_prompt or None,
        model=model or None,
        seed=seed,
        params=params,
        make_active=make_active,
    )


def memory_get_clip(project_id: str, clip_id: str) -> Dict[str, Any]:
    """
    Fetch a single clip by id.
    """
    return {"project_id": project_id, "clip": _svc(project_id).get_clip(clip_id=clip_id)}


def memory_list_clips(project_id: str, segment_id: str) -> Dict[str, Any]:
    """
    List all takes for a segment.
    """
    return {"project_id": project_id, "segment_id": segment_id, "clips": _svc(project_id).list_clips_for_segment(segment_id=segment_id)}


def memory_get_clips_for_segments(project_id: str, segment_ids: List[str]) -> Dict[str, Any]:
    """
    Fetch clips grouped by segment ids.
    """
    return {"project_id": project_id, "clips_by_segment": _svc(project_id).get_clips_for_segments(segment_ids=segment_ids)}


def memory_delete_clip(project_id: str, clip_id: str) -> Dict[str, Any]:
    """
    Delete a clip (and related evals/artifacts).
    """
    ok = _svc(project_id).delete_clip(clip_id=clip_id)
    return {"project_id": project_id, "clip_id": clip_id, "deleted": ok}


def memory_set_active_take(project_id: str, segment_id: str, clip_id: str) -> Dict[str, Any]:
    """
    Mark a clip as the active take for its segment.
    """
    _svc(project_id).set_active_take(segment_id=segment_id, clip_id=clip_id)
    return {"project_id": project_id, "segment_id": segment_id, "active_clip_id": clip_id}


def memory_add_beat(
    project_id: str,
    segment_id: str,
    beat_type: str,
    summary: str = "",
    payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Attach a structured beat/event to a segment.
    """
    return _svc(project_id).add_beat(segment_id=segment_id, beat_type=beat_type, summary=summary, payload=payload)


def memory_delete_beat(project_id: str, beat_id: str) -> Dict[str, Any]:
    """
    Delete a beat by id.
    """
    ok = _svc(project_id).delete_beat(beat_id=beat_id)
    return {"project_id": project_id, "beat_id": beat_id, "deleted": ok}


def memory_add_entity_state(
    project_id: str,
    entity_name: str,
    t_start: float,
    t_end: float,
    state: Dict[str, Any],
    source_clip_id: str = "",
) -> Dict[str, Any]:
    """
    Record an entity state over a time range (for consistency constraints).
    """
    return _svc(project_id).add_entity_state(
        entity_name=entity_name,
        t_start=t_start,
        t_end=t_end,
        state=state,
        source_clip_id=source_clip_id or None,
    )


def memory_list_entity_states(
    project_id: str,
    entity_name: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> Dict[str, Any]:
    """
    List entity states with optional filters.
    """
    return {
        "project_id": project_id,
        "entity_states": _svc(project_id).list_entity_states(
            entity_name=entity_name or None,
            t_start=t_start,
            t_end=t_end,
        ),
    }


def memory_delete_entity_state(project_id: str, state_id: str) -> Dict[str, Any]:
    """
    Delete an entity state by id.
    """
    ok = _svc(project_id).delete_entity_state(state_id=state_id)
    return {"project_id": project_id, "state_id": state_id, "deleted": ok}


def memory_get_entity_states_at(project_id: str, t: float) -> Dict[str, Any]:
    """
    Get entity states applicable at time t.
    """
    return {"project_id": project_id, "t": float(t), "entity_states": _svc(project_id).get_entity_states_at(t=float(t))}


def memory_save_eval(
    project_id: str,
    clip_id: str,
    consistency_score: Optional[float] = None,
    story_match_score: Optional[float] = None,
    visual_score: Optional[float] = None,
    note: str = "",
) -> Dict[str, Any]:
    """
    Save evaluation scores for a clip.
    """
    return _svc(project_id).add_eval(
        clip_id=clip_id,
        consistency_score=consistency_score,
        story_match_score=story_match_score,
        visual_score=visual_score,
        note=note,
    )


def memory_list_evals(project_id: str, clip_id: str) -> Dict[str, Any]:
    """
    List evals for a clip.
    """
    return {"project_id": project_id, "clip_id": clip_id, "evals": _svc(project_id).list_evals(clip_id=clip_id)}


def memory_add_artifact(
    project_id: str,
    kind: str,
    path: str,
    segment_id: str = "",
    clip_id: str = "",
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Add an artifact linked to a segment or clip.
    """
    return _svc(project_id).add_artifact(
        kind=kind,
        path=path,
        segment_id=segment_id or None,
        clip_id=clip_id or None,
        meta=meta,
    )


def memory_list_artifacts(
    project_id: str,
    segment_id: str = "",
    clip_id: str = "",
    kind: str = "",
) -> Dict[str, Any]:
    """
    List artifacts with optional filters.
    """
    return {
        "project_id": project_id,
        "artifacts": _svc(project_id).list_artifacts(
            segment_id=segment_id or None,
            clip_id=clip_id or None,
            kind=kind or None,
        ),
    }


def memory_search_assets(project_id: str, query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Search assets by semantic query (FTS).
    """
    return {
        "project_id": project_id,
        "query": query,
        "assets": _svc(project_id).search_assets(query=query, limit=limit),
    }


def memory_update_asset_caption(
    project_id: str,
    asset_id: str,
    caption: str,
    entity_summary: str = "",
    tags: str = "",
) -> Dict[str, Any]:
    """
    Update caption and optional entity summary/tags for an asset.
    """
    ok = _svc(project_id).update_asset_caption(
        asset_id=asset_id,
        caption=caption,
        entity_summary=entity_summary or None,
        tags=tags or None,
    )
    return {"project_id": project_id, "asset_id": asset_id, "updated": ok}


def memory_get_latest_artifact(
    project_id: str,
    segment_id: str = "",
    clip_id: str = "",
    kind: str = "",
) -> Dict[str, Any]:
    """
    Fetch the most recent artifact with optional filters.
    """
    return {
        "project_id": project_id,
        "artifact": _svc(project_id).get_latest_artifact(
            segment_id=segment_id or None,
            clip_id=clip_id or None,
            kind=kind or None,
        ),
    }


def memory_get_last_frame(
    project_id: str,
    segment_id: str = "",
    clip_id: str = "",
) -> Dict[str, Any]:
    """
    Fetch the most recent last-frame artifact (prefers active clip for segment).
    """
    return {
        "project_id": project_id,
        "segment_id": segment_id or None,
        "clip_id": clip_id or None,
        "artifact": _svc(project_id).get_last_frame(
            segment_id=segment_id or None,
            clip_id=clip_id or None,
        ),
    }


def memory_delete_artifact(project_id: str, artifact_id: str) -> Dict[str, Any]:
    """
    Delete an artifact by id.
    """
    ok = _svc(project_id).delete_artifact(artifact_id=artifact_id)
    return {"project_id": project_id, "artifact_id": artifact_id, "deleted": ok}


def memory_get_context_window(project_id: str, t_start: float, t_end: float, pad_sec: float = 8.0) -> Dict[str, Any]:
    """
    Get timeline context around the focus window [t_start, t_end].
    """
    return _svc(project_id).get_context_window(t_start=float(t_start), t_end=float(t_end), pad_sec=float(pad_sec))


def memory_backfill_asset_index(project_id: str) -> Dict[str, Any]:
    """
    Backfill asset index rows from existing artifacts.
    """
    count = _svc(project_id).backfill_asset_index()
    return {"project_id": project_id, "indexed": count}


def get_memory_tools() -> List[Any]:
    """
    Convenience for agent wiring: return a list of callables exposed as tools.

    Each callable is wrapped so it returns a consistent ``success``-bearing
    envelope (see ``_tool_envelope``).
    """
    funcs = [
        memory_init_project,
        memory_upsert_segment,
        memory_get_segment,
        memory_list_segments,
        memory_update_segment_status,
        memory_delete_segment,
        memory_save_clip_take,
        memory_get_clip,
        memory_list_clips,
        memory_get_clips_for_segments,
        memory_delete_clip,
        memory_set_active_take,
        memory_add_beat,
        memory_delete_beat,
        memory_add_entity_state,
        memory_list_entity_states,
        memory_delete_entity_state,
        memory_get_entity_states_at,
        memory_save_eval,
        memory_list_evals,
        memory_add_artifact,
        memory_list_artifacts,
        memory_search_assets,
        memory_update_asset_caption,
        memory_get_latest_artifact,
        memory_get_last_frame,
        memory_delete_artifact,
        memory_get_context_window,
        memory_backfill_asset_index,
    ]
    return [_tool_envelope(f) for f in funcs]
