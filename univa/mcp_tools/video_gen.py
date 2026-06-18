import yaml
import json
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from univa.mcp_tools.base import ToolResponse, setup_logger
from univa.utils.video_process import merge_videos, storyboard_generate, save_last_frame_decord, save_first_frame_decord
from univa.utils.query_llm import refine_gen_prompt, audio_prompt_gen
from univa.utils.image_process import download_image
from univa.utils.wavespeed_api import (
    text_to_video_generate,
    image_to_video_generate,
    frame_to_frame_video,
    text_to_image_generate,
    image_to_image_generate,
    audio_gen,
    hailuo_i2v_pro,
)

# Load configuration
config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config/mcp_tools_config")
config_path = os.path.join(config_dir, "config.yaml")
if not os.path.exists(config_path):
    config_path = os.path.join(config_dir, "config.example.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

video_gen_config = config.get('video_gen', {})
image_gen_config = config.get('image_gen', {})
# base_output_path = video_gen_config.get('base_output_path', '/share/project/liangzy/liangzy2/UniVideo/generated_videos')

def _get_wavespeed_api_key() -> str:
    """API keys must come from environment variables (.env)."""
    key = os.getenv("WAVESPEED_API_KEY") or ""
    if not key:
        raise RuntimeError("Missing WAVESPEED_API_KEY (set it in .env or environment).")
    return key

# Configure logging
logger = setup_logger(__name__)
logger.info(f"Loaded video_gen_config: {video_gen_config}")

try:
    from univa.memory.service import ProjectMemoryService
except Exception:
    ProjectMemoryService = None

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CACHE_TTL_SEC = 180
_DEFAULT_SEGMENT_DURATION_SEC = 5.0
_RECENT_RESULTS: Dict[Tuple, Dict] = {}


def _resolve_output_dir(base_output_path: str | None) -> Path:
    base = Path(base_output_path) if base_output_path else Path("results")
    if not base.is_absolute():
        base = _REPO_ROOT / base
    return base


def _slugify(text: str, max_len: int = 30) -> str:
    s = re.sub(r"[^a-zA-Z0-9_-]+", "_", text.strip())[:max_len]
    return s.strip("_") or "output"


def _make_save_path(prompt: str, ext: str = ".mp4") -> str:
    base_dir = _resolve_output_dir(video_gen_config.get("base_output_path"))
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    slug = _slugify(prompt)
    save_dir = base_dir / f"{stamp}_{slug}"
    save_dir.mkdir(parents=True, exist_ok=True)
    fname = datetime.now().strftime("%m%d%H%M%S") + ext
    return str(save_dir / fname)


def _cache_key(tool: str, prompt: str, *parts: str) -> Tuple:
    return (tool, prompt, *parts)


def _cache_get(key: Tuple) -> Optional[Dict]:
    entry = _RECENT_RESULTS.get(key)
    if not entry:
        return None
    if time.time() - entry["ts"] > _CACHE_TTL_SEC:
        _RECENT_RESULTS.pop(key, None)
        return None
    return entry["data"]


def _cache_put(key: Tuple, data: Dict) -> None:
    _RECENT_RESULTS[key] = {"ts": time.time(), "data": data}


def _to_tool_response(data: Dict, success: bool, message: str) -> ToolResponse:
    output_path = data.get("output_path")
    if output_path:
        try:
            output_path = str(Path(output_path).resolve())
            data["output_path"] = output_path
        except Exception:
            pass
    output_url = data.get("output_url")
    segment_id = data.get("segment_id")
    clip_id = data.get("clip_id")
    last_frame_path = data.get("last_frame_path")
    cached = data.get("cached")
    return ToolResponse(
        success=success,
        message=message,
        output_path=output_path,
        output_url=output_url,
        segment_id=segment_id,
        clip_id=clip_id,
        last_frame_path=last_frame_path,
        cached=cached,
        content=data,
    )


def _maybe_open_memory(project_id: Optional[str]):
    if not project_id or ProjectMemoryService is None:
        return None
    try:
        return ProjectMemoryService.open(project_id=project_id)
    except Exception as exc:
        logger.warning(f"Failed to open memory DB for project_id={project_id}: {exc}")
        return None


def _auto_segment_range(svc) -> Optional[tuple[float, float]]:
    if not svc:
        return None
    try:
        segments = svc.list_segments()
    except Exception:
        segments = []
    if segments:
        last_end = max(float(seg.get("t_end", 0.0)) for seg in segments)
        t_start = last_end
    else:
        t_start = 0.0
    return t_start, t_start + _DEFAULT_SEGMENT_DURATION_SEC


def _resolve_segment_id(svc, segment_id: str, t_start: Optional[float], t_end: Optional[float], kind: str, status: str) -> Optional[str]:
    if not svc:
        return None
    if segment_id:
        try:
            existing = svc.get_segment(segment_id=segment_id)
        except Exception:
            existing = None
        if existing:
            return segment_id
        logger.warning(f"segment_id not found in memory: {segment_id}")
    if t_start is None and t_end is None:
        auto_range = _auto_segment_range(svc)
        if auto_range:
            t_start, t_end = auto_range
    elif t_start is None and t_end is not None:
        t_start = max(0.0, float(t_end) - _DEFAULT_SEGMENT_DURATION_SEC)
    elif t_start is not None and t_end is None:
        t_end = float(t_start) + _DEFAULT_SEGMENT_DURATION_SEC
    if t_start is None or t_end is None:
        return None
    try:
        return svc.upsert_segment(t_start=t_start, t_end=t_end, kind=kind, status=status)
    except Exception as exc:
        logger.warning(f"Failed to upsert segment: {exc}")
        return None


def _extract_last_frame(video_path: str) -> Optional[str]:
    if not video_path:
        return None
    try:
        return save_last_frame_decord(video_path)
    except Exception as exc:
        logger.warning(f"Failed to extract last frame: {exc}")
        return None


# Extensions that image-conditioned tools accept directly. Anything else (e.g. a
# video the agent passed by mistake) is treated as a clip to extract a frame from.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tiff"}
_VIDEO_EXTS = {".mp4", ".mov", ".webm", ".avi", ".mkv", ".m4v", ".gif"}


def _ensure_image_input(image_path: str) -> str:
    """Return a path that is safe to send to an image-conditioned generator.

    The agent sometimes passes a *video* path to image2video_gen (e.g. the user
    attached a clip). The downstream WaveSpeed call base64-encodes the bytes and
    hard-labels them ``data:image/jpeg``, so a video silently becomes a corrupt
    "jpeg" and the provider rejects it. To match intent ("animate this subject"),
    extract the first frame and use that as the conditioning image instead.
    """
    if not image_path:
        return image_path
    ext = os.path.splitext(image_path)[1].lower()
    if ext in _IMAGE_EXTS:
        return image_path
    if ext in _VIDEO_EXTS or os.path.isfile(image_path):
        # Treat unknown-but-existing inputs as possible video and try a frame.
        try:
            frame = save_first_frame_decord(image_path)
            if frame:
                logger.info(f"image input '{image_path}' is a video; using first frame '{frame}'")
                return frame
        except Exception as exc:
            logger.warning(f"Could not extract a frame from '{image_path}': {exc}")
    return image_path


def _maybe_store_last_frame(
    svc,
    segment_id: Optional[str],
    clip_id: Optional[str],
    last_frame_path: Optional[str],
    video_path: Optional[str],
) -> None:
    if not svc or not last_frame_path:
        return
    try:
        svc.add_artifact(
            kind="last_frame",
            path=last_frame_path,
            segment_id=segment_id or None,
            clip_id=clip_id or None,
            meta={"source": "video_gen", "video_path": video_path},
        )
    except Exception as exc:
        logger.warning(f"Failed to save last frame artifact: {exc}")

async def text2video_gen(
    prompt: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) from a text description.
    If segment_id is provided, it updates the corresponding segment in the server timeline.
    Otherwise, it generates a video without adding it to the timeline.

    Args:
        prompt (str): The prompt to generate the video.
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("text_to_video")
    cache_key = _cache_key("text2video", prompt)
    cached = _cache_get(cache_key)
    if cached and cached.get("success"):
        cached["cached"] = True
        return _to_tool_response(cached, True, "Video generated (cached).")

    if model == "seedance":
        api_key = _get_wavespeed_api_key()
        save_path = _make_save_path(prompt, ext=".mp4")
        return_dict = text_to_video_generate(api_key, prompt, save_path=save_path)

        last_frame_path = None
        if return_dict.get("success") and save_last_frame:
            last_frame_path = _extract_last_frame(return_dict.get("output_path"))
            if last_frame_path:
                return_dict["last_frame_path"] = last_frame_path

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={"tool": "text2video_gen"},
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame and last_frame_path:
                            _maybe_store_last_frame(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                last_frame_path=last_frame_path,
                                video_path=return_dict.get("output_path"),
                            )
        finally:
            if svc:
                svc.close()

        if return_dict.get("success"):
            _cache_put(cache_key, return_dict)
            return _to_tool_response(return_dict, True, "Video generated successfully.")
        return _to_tool_response(return_dict, False, return_dict.get("error", "Video generation failed."))

    return _to_tool_response(
        {"success": False, "error": f"Unsupported text_to_video model: {model}"},
        False,
        f"Unsupported text_to_video model: {model}",
    )


async def image2video_gen(
    prompt: str,
    image_path: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) using a text prompt and an input image as a visual reference.
    This tool is useful for creating videos that maintain visual consistency with a provided image while incorporating new elements described in the prompt.

    Args:
        prompt (str): The prompt to generate the video.
        image_path (str): Input image path for use as video content reference, supporting common formats (.jpg/.png/.bmp, etc.).
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.

    Returns:
        dict: A dictionary containing the success status, output video path, and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str, optional): The path to the generated video if successful.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("image_to_video")
    cache_key = _cache_key("image2video", prompt, image_path)
    cached = _cache_get(cache_key)
    if cached and cached.get("success"):
        cached["cached"] = True
        return _to_tool_response(cached, True, "Video generated (cached).")

    if model == "seedance":
        api_key = _get_wavespeed_api_key()
        # If a video (or non-image) was passed, condition on its first frame so the
        # provider receives a real image instead of a mislabeled video blob.
        image_path = _ensure_image_input(image_path)
        save_path = _make_save_path(prompt, ext=".mp4")
        return_dict = image_to_video_generate(api_key, prompt, image_path, save_path=save_path)

        last_frame_path = None
        if return_dict.get("success") and save_last_frame:
            last_frame_path = _extract_last_frame(return_dict.get("output_path"))
            if last_frame_path:
                return_dict["last_frame_path"] = last_frame_path

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={"tool": "image2video_gen", "image_path": image_path},
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame and last_frame_path:
                            _maybe_store_last_frame(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                last_frame_path=last_frame_path,
                                video_path=return_dict.get("output_path"),
                            )
        finally:
            if svc:
                svc.close()

        if return_dict.get("success"):
            _cache_put(cache_key, return_dict)
            return _to_tool_response(return_dict, True, "Video generated successfully.")
        return _to_tool_response(return_dict, False, return_dict.get("error", "Video generation failed."))

    return _to_tool_response(
        {"success": False, "error": f"Unsupported image_to_video model: {model}"},
        False,
        f"Unsupported image_to_video model: {model}",
    )



async def frame2frame_video_gen(
    prompt: str,
    first_frame_path: str,
    last_frame_path: str,
    project_id: str = "",
    segment_id: str = "",
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    kind: str = "target",
    status: str = "planned",
    save_last_frame: bool = True,
) -> str:
    """
    Generates a short video (approx. 5 seconds) that transitions between a specified first frame and a last frame, guided by a text prompt.
    This tool is effective for creating dynamic action sequences or smooth transitions between two distinct visual states.

    Args:
        prompt (str): The prompt to generate the video.
        first_frame_path (str): The path to the first frame.
        last_frame_path (str): The path to the last frame.
        project_id (str): Optional project id to persist memory.
        segment_id (str): Optional segment id to attach the clip to.
        t_start (float): Optional segment start time (seconds).
        t_end (float): Optional segment end time (seconds).
        kind (str): Segment kind if created (default: target).
        status (str): Segment status if created (default: planned).
        save_last_frame (bool): Whether to save last frame into memory artifacts.
        save_path (str): The path to save the generated video. Suggest numbering each video when naming them.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'message' (str, optional): A success message.
              - 'error' (str, optional): An error message if the generation failed.
    """
    model = video_gen_config.get("frame_to_frame_video")
    cache_key = _cache_key("frame2frame", prompt, first_frame_path, last_frame_path)
    cached = _cache_get(cache_key)
    if cached and cached.get("success"):
        cached["cached"] = True
        return _to_tool_response(cached, True, "Video generated (cached).")

    if model == "wan_api":
        api_key = _get_wavespeed_api_key()
        save_path = _make_save_path(prompt, ext=".mp4")
        return_dict = hailuo_i2v_pro(api_key, prompt, first_frame_path, last_frame_path, save_path=save_path)

        last_frame_path_out = None
        if return_dict.get("success") and save_last_frame:
            last_frame_path_out = _extract_last_frame(return_dict.get("output_path"))
            if last_frame_path_out:
                return_dict["last_frame_path"] = last_frame_path_out

        svc = _maybe_open_memory(project_id)
        try:
            if svc and return_dict.get("success"):
                seg_id = _resolve_segment_id(svc, segment_id, t_start, t_end, kind, status)
                if seg_id:
                    try:
                        clip = svc.save_clip_take(
                            segment_id=seg_id,
                            output_path=return_dict.get("output_path"),
                            prompt=prompt,
                            model=model or "",
                            params={
                                "tool": "frame2frame_video_gen",
                                "first_frame_path": first_frame_path,
                                "last_frame_path": last_frame_path,
                            },
                            make_active=True,
                        )
                    except Exception as exc:
                        logger.warning(f"Failed to save clip take: {exc}")
                        clip = None
                    if clip:
                        return_dict["segment_id"] = seg_id
                        return_dict["clip_id"] = clip.get("clip_id")
                        if save_last_frame and last_frame_path_out:
                            _maybe_store_last_frame(
                                svc,
                                segment_id=seg_id,
                                clip_id=clip.get("clip_id"),
                                last_frame_path=last_frame_path_out,
                                video_path=return_dict.get("output_path"),
                            )
        finally:
            if svc:
                svc.close()

        if return_dict.get("success"):
            _cache_put(cache_key, return_dict)
            return _to_tool_response(return_dict, True, "Video generated successfully.")
        return _to_tool_response(return_dict, False, return_dict.get("error", "Video generation failed."))

    return _to_tool_response(
        {"success": False, "error": f"Unsupported frame_to_frame_video model: {model}"},
        False,
        f"Unsupported frame_to_frame_video model: {model}",
    )


async def merge2videos(video_paths: list[str]):
    """
    Merges multiple video files into a single video file.

    Args:
        video_paths (list[str]): A list of paths to the video files to merge, or a folder path containing videos.

    Returns:
        dict: A dictionary containing the success status and a message.
              - 'success' (bool): True if the video was generated successfully, False otherwise.
              - 'output_path' (str): The path to the merged video.
              - 'message' (str): A success message.
    """
    save_dir = f"results/{datetime.now().strftime('%Y%m%d%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    _time = datetime.now().strftime("%m%d%H%M%S")
    save_path = f"{save_dir}/{_time}.mp4"
    video_path = merge_videos(video_paths, output_file=save_path)

    return ToolResponse(
        success=True,
        output_path=video_path,
        message="Videos merged successfully."
    )
