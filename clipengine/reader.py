# reader.py
# frame access for analysis. windows decode downscaled to the analysis
# width and keep frames one step apart, 1/30 s at the defaults: every
# frame up to 30 fps, a stride of whole frames above it, so a window
# spans its full length at any fps. displacements stay small enough for
# farneback to track even fast whips, and per frame noise does not grow
# with fps.
# frame times come from the decoder, so variable frame rate files seek
# and scale right. only ever called on materialized files; the catalog
# guards eviction upstream.

import logging
import math
from typing import Optional

import cv2
import numpy as np

from clipengine import config

logger = logging.getLogger(__name__)

# a step this close to a whole number of frames takes that many, so
# 30 fps strides 1 and 60 fps strides 2
_STEP_SLACK = 0.98


class ClipReadError(Exception):
    """raised when a clip cannot be opened or yields no frames."""


def _downscale(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, max(2, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


def _open(path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise ClipReadError(f"decoder could not open {path}")
    return cap


# the one fps rule for probe and reader: a decoder rate outside this
# range is noise, so we assume 30
FPS_MIN, FPS_MAX, FPS_FALLBACK = 1.0, 240.0, 30.0
# sources already warned about, so one analysis logs a bad rate once
_warned: set[str] = set()


def sane_fps(raw, source: str = "") -> float:
    """raw decoder fps kept when it falls in 1.0 to 240.0, else 30 with
    a logged warning, once per source."""
    try:
        fps = float(raw)
    except (TypeError, ValueError):
        fps = float("nan")
    if FPS_MIN <= fps <= FPS_MAX:
        return fps
    if not source or source not in _warned:
        if source:
            _warned.add(source)
        logger.warning("fps %r out of range%s, using %.0f", raw,
                       f" for {source}" if source else "", FPS_FALLBACK)
    return FPS_FALLBACK


def _fps(cap: cv2.VideoCapture, source: str = "") -> float:
    return sane_fps(cap.get(cv2.CAP_PROP_FPS), source)


def _guessed(cap: cv2.VideoCapture) -> bool:
    """true when the decoder rate was out of range and 30 is a guess."""
    try:
        return not FPS_MIN <= float(cap.get(cv2.CAP_PROP_FPS)) <= FPS_MAX
    except (TypeError, ValueError):
        return True


def _stride(step_s: float, fps: float) -> int:
    """whole frames per step at the nominal fps, 1 up to 30 fps."""
    return max(1, math.ceil(_STEP_SLACK * step_s * fps - 1e-9))


def _keep_gap(step_s: float, fps: float) -> float:
    """seconds after the last kept frame that count as one step. half a
    frame of slack on the stride absorbs phone timestamp jitter, so a
    30 fps clip on 19 and 21 tick frames still keeps every frame."""
    return min(_STEP_SLACK * step_s, (_stride(step_s, fps) - 0.5) / fps)


def _clock(cap: cv2.VideoCapture, guess: float, after: float = -1.0) -> float:
    """time in seconds of the frame just grabbed. the decoder timestamp
    when it is usable, else the guess from the nominal fps."""
    t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if t > after and (t > 0.0 or guess <= 0.0):
        return t
    return guess


def _seek(cap: cv2.VideoCapture, t_s: float, fps: float) -> Optional[float]:
    """grab the first frame no more than half a nominal frame before t_s
    and return its time, none past the end. the first guess assumes
    constant fps, which lands exactly on constant rate files; on variable
    rate files the timestamps walk it the rest of the way, so in a slow
    stretch it can land one frame after t_s."""
    half = 0.5 / fps
    target = max(0, round(t_s * fps))
    t, fresh = None, True
    for _ in range(4):
        if target > 0 or not fresh:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        fresh = False
        if cap.grab():
            t = _clock(cap, target / fps)
            if target == 0 or t <= t_s + half + 1e-6:
                break
            # landed late: back up past the overshoot, walk forward below
            target = max(0, target - round((t - t_s) * fps) - 1)
        elif target == 0:
            return None
        else:
            # a variable rate seek can land past the end; back up a second
            target = max(0, target - round(fps))
            t = None
    if t is None:
        return None
    while t < t_s - half - 1e-6:
        if not cap.grab():
            return None
        t = _clock(cap, t + 1.0 / fps, t)
    return t


def _collect(cap: cv2.VideoCapture, start_s: float, duration_s: float,
             step_s: float, fps: float, width: int, max_frames: int,
             keep_s: Optional[float] = None
             ) -> tuple[list[np.ndarray], list[float]]:
    """frames one stride apart, about step_s, for duration_s from
    start_s. skipped frames are only grabbed, never converted or
    downscaled. keep_s overrides the gap between kept frames."""
    frames: list[np.ndarray] = []
    times: list[float] = []
    if keep_s is None:
        keep_s = _keep_gap(step_s, fps)
    t = _seek(cap, start_s, fps)
    if t is None:
        return frames, times
    # half a frame short, so the frame count matches round(duration_s * fps)
    end = t + duration_s - 0.5 / fps
    while len(frames) < max_frames:
        if t >= end and len(frames) >= 2:
            break
        if not times or t - times[-1] >= keep_s:
            ok, frame = cap.retrieve()
            if not ok:
                break
            frames.append(_downscale(frame, width))
            times.append(t)
        if not cap.grab():
            break
        t = _clock(cap, t + 1.0 / fps, t)
    return frames, times


def _finite(t_s: float, path) -> float:
    """t_s as a float, a clear error when it is nan or infinite."""
    t_s = float(t_s)
    if not math.isfinite(t_s):
        raise ClipReadError(f"time {t_s} is not a finite number for {path}")
    return t_s


def _tail(cap: cv2.VideoCapture, start_s: float, duration_s: float,
          step_s: float, fps: float, width: int, max_frames: int, path,
          keep_s: Optional[float] = None
          ) -> tuple[list[np.ndarray], list[float]]:
    """the window for a seek near the end that ran out of frames, as when
    the frame count runs past the real end or the file is cut short.
    step back a window at a time, read to the end and keep the last
    window of frames that decode. never the head of the clip: when no
    frame decodes past the last frame of the start window this raises."""
    want = max(2, round(duration_s * fps))
    back = max(1, round(start_s * fps))
    best: tuple[list[np.ndarray], list[float]] = ([], [])
    while back > 0:
        back = max(0, back - want)
        cap.set(cv2.CAP_PROP_POS_FRAMES, back)
        reach = start_s - back / fps + duration_s
        frames, times = _collect(cap, back / fps, reach, step_s, fps, width,
                                 math.ceil(reach / step_s) + 2, keep_s)
        if len(frames) < 2:
            continue
        cut = times[-1] - duration_s + 0.5 / fps
        keep = [i for i, t in enumerate(times) if t >= cut][-max_frames:]
        best = [frames[i] for i in keep], [times[i] for i in keep]
        # a run that starts before the cut covers a whole window
        if times[0] <= cut:
            break
    frames, times = best
    if len(frames) < 2 or times[-1] < (want - 0.5) / fps:
        raise ClipReadError(
            f"end of {path} does not decode, the file may be truncated")
    return frames, times


def _short(cap: cv2.VideoCapture, times: list[float], duration_s: float,
           step_s: float, fps: float, max_frames: int) -> bool:
    """true when a window ran into the end of the stream before covering
    duration_s, as when the frame count runs past the real end or the
    file is cut short. a full window spans its frames less one stride."""
    if len(times) < 2:
        return True
    if len(times) >= max_frames:
        return False
    full = round(duration_s * fps) - _stride(step_s, fps) - 0.5
    return times[-1] - times[0] < full / fps and not cap.grab()


def read_window(path, start_s: float,
                duration_s: Optional[float] = None,
                width: Optional[int] = None,
                max_frames: Optional[int] = None
                ) -> tuple[list[np.ndarray], list[float], float]:
    """decode downscaled bgr frames covering duration_s from start_s.
    returns the frames, their times in seconds and the nominal fps.
    frames sit whole frames apart, about duration_s / max_frames, so a
    fast source strides through the window instead of stopping short at
    the cap. defaults come from config when called. a window near the
    end that runs out of frames before it covers duration_s steps back
    to the last full window that decodes, or raises; it never falls back
    to the head."""
    start_s = _finite(start_s, path)
    duration_s = config.WINDOW_SECONDS if duration_s is None else duration_s
    width = config.ANALYSIS_WIDTH if width is None else width
    max_frames = config.MAX_WINDOW_FRAMES if max_frames is None else max_frames
    cap = _open(path)
    try:
        fps = _fps(cap, str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        span = max(2, round(duration_s * fps))
        start_s = max(0.0, start_s)
        if total > 0:
            start_s = min(start_s, max(0, total - span) / fps)
        step_s = duration_s / max_frames
        # with a guessed rate the strides mean nothing, so keep frames a
        # step apart by their decoder times
        keep_s = _STEP_SLACK * step_s if _guessed(cap) else None
        frames, times = _collect(cap, start_s, duration_s, step_s, fps,
                                 width, max_frames, keep_s)
        if start_s > 0 and _short(cap, times, duration_s, step_s, fps,
                                  max_frames):
            frames, times = _tail(cap, start_s, duration_s, step_s, fps,
                                  width, max_frames, path, keep_s)
        elif len(frames) < 2:
            # a clip shorter than one step reads every frame instead
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            again = _collect(cap, 0.0, duration_s, 0.0, fps, width,
                             max_frames)
            if len(again[0]) > len(frames):
                frames, times = again
        if not frames:
            raise ClipReadError(f"no decodable frames in {path}")
        if len(frames) < 2:
            raise ClipReadError(
                f"only one frame decodes in {path}, a window needs two")
        return frames, times, fps
    finally:
        cap.release()


def read_pair(path, t_s: float, width: Optional[int] = None
              ) -> Optional[tuple[np.ndarray, np.ndarray, float]]:
    """two frames one window step apart at t_s for spot flow probes, and
    the seconds between them. returns none on seek or decode failure so
    callers can skip the position; a file that will not open or a time
    that is not finite raises ClipReadError."""
    t_s = _finite(t_s, path)
    width = config.ANALYSIS_WIDTH if width is None else width
    cap = _open(path)
    try:
        fps = _fps(cap, str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        step_s = config.WINDOW_SECONDS / config.MAX_WINDOW_FRAMES
        keep_s = _keep_gap(step_s, fps)
        t_s = max(0.0, t_s)
        if total > 0:
            # leave room for the second frame of the pair
            t_s = min(t_s, max(0, total - 1 - _stride(step_s, fps)) / fps)
        t1 = _seek(cap, t_s, fps)
        if t1 is None:
            return None
        ok1, f1 = cap.retrieve()
        if not ok1:
            return None
        t2 = t1
        while t2 - t1 < keep_s:
            if not cap.grab():
                return None
            t2 = _clock(cap, t2 + 1.0 / fps, t2)
        ok2, f2 = cap.retrieve()
        if not ok2:
            return None
        return _downscale(f1, width), _downscale(f2, width), t2 - t1
    finally:
        cap.release()


def read_frame(path, t_s: float, width: int) -> Optional[np.ndarray]:
    """single frame at t_s, used for thumbnails."""
    pair = read_pair(path, t_s, width)
    return pair[0] if pair else None
