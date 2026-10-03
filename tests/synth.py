# synth.py
# synthetic clip factory. every helper produces frames whose motion or
# color is known by construction, so analyzer tests assert against
# ground truth instead of eyeballs. rolling a random blocky texture
# gives optical flow strong features and a mathematically exact
# translation speed.

import contextlib
import os
import sys
from typing import Optional

import cv2
import numpy as np

FPS = 24.0
W, H = 160, 120


def texture(seed: int = 7, block: int = 8) -> np.ndarray:
    """blocky random texture the exact frame size; rolling it wraps."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (H // block, W // block, 3), np.uint8)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)


def frames_for(kind: str, n: int = 72, seed: int = 7,
               px: int = 3) -> list[np.ndarray]:
    """n frames of a named camera move. px is speed in pixels per frame.

    conventions match analysis.classify_motion: a camera pan right makes
    content flow left, so 'pan_right' rolls the texture left each frame.
    whips use a coarser texture: real scenes keep large features visible
    through fast motion, and fine blocks alias at 14 px/frame.
    """
    tex = texture(seed, block=20 if kind.startswith("whip") else 8)
    rng = np.random.default_rng(seed + 1)
    frames = []
    scale = 1.0
    for i in range(n):
        k = i * px
        if kind == "static":
            noise = rng.integers(0, 4, (H, W, 3), np.uint8)
            frames.append(cv2.add(tex // 2 + 60, noise))
        elif kind == "pan_right" or kind == "whip_right":
            frames.append(np.roll(tex, -k, axis=1))
        elif kind == "pan_left" or kind == "whip_left":
            frames.append(np.roll(tex, k, axis=1))
        elif kind == "tilt_up":
            frames.append(np.roll(tex, k, axis=0))
        elif kind == "tilt_down":
            frames.append(np.roll(tex, -k, axis=0))
        elif kind == "handheld":
            shake = px if i % 2 == 0 else -px
            frames.append(np.roll(tex, shake, axis=1))
        elif kind in ("push_in", "pull_out"):
            rate = 1.015 if kind == "push_in" else 1 / 1.015
            scale *= rate
            m = cv2.getRotationMatrix2D((W / 2, H / 2), 0, scale)
            frames.append(cv2.warpAffine(tex, m, (W, H),
                                         borderMode=cv2.BORDER_REFLECT))
        else:
            raise ValueError(f"unknown kind: {kind}")
    return frames


def color_frames(bgr: tuple[int, int, int], n: int = 30,
                 noise: int = 3, seed: int = 5) -> list[np.ndarray]:
    """flat colored frames with a little noise, for color stat tests."""
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), bgr, np.uint8)
    return [cv2.add(base, rng.integers(0, noise + 1, (H, W, 3), np.uint8))
            for _ in range(n)]


def gray_ramp_frames(n: int = 30, lo: int = 110, hi: int = 140) -> list[np.ndarray]:
    """low contrast gray gradient: mimics log footage before normalization."""
    ramp = np.linspace(lo, hi, H, dtype=np.uint8)[:, None]
    frame = np.repeat(np.repeat(ramp, W, axis=1)[..., None], 3, axis=2)
    return [frame.copy() for _ in range(n)]


class WriterUnavailable(RuntimeError):
    """opencv has no writer for this fourcc and container here."""


# x265 prints its banner and stats straight to fd 2
_LOUD = {"hvc1", "hev1", "hevc"}


@contextlib.contextmanager
def _quiet_stderr(on: bool):
    if not on:
        yield
        return
    sys.stderr.flush()
    saved, sink = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(sink, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(sink)


def write_clip(path, frames: list[np.ndarray], fps: Optional[float] = None,
               fourcc: str = "mp4v") -> None:
    """write frames as a clip. the size comes from the first frame and fps
    defaults to FPS. raises WriterUnavailable when the writer will not
    open, so codec subtests can skip."""
    if not frames:
        raise ValueError("no frames to write")
    h, w = frames[0].shape[:2]
    with _quiet_stderr(fourcc.lower() in _LOUD):
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc),
                             FPS if fps is None else fps, (w, h),
                             frames[0].ndim == 3)
        if not vw.isOpened():
            vw.release()
            raise WriterUnavailable(
                f"cv2 VideoWriter failed to open {fourcc} for {path}")
        for f in frames:
            vw.write(f)
        vw.release()


def make_clip(path, kinds: list[str], n_each: int = 36, seed: int = 7,
              px: int = 3) -> None:
    """write one clip made of consecutive phases, e.g. static then pan.
    windows only inspect the first and last ~1.2 s, so a phase of 36
    frames (1.5 s at 24 fps) fully owns its window."""
    frames = []
    for kind in kinds:
        frames.extend(frames_for(kind, n=n_each, seed=seed, px=px))
    write_clip(path, frames)
