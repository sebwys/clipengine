# media_fixtures.py
# media the synth factory cannot make on its own: frames that carry their
# own index through a lossy codec, and byte patches on the mp4 boxes
# opencv writes. variable frame rate timing, mvhd values, a moov first
# layout and display rotation all come from rewriting boxes, so no test
# needs ffmpeg.

import struct
from bisect import bisect_right
from pathlib import Path
from typing import Iterator, NamedTuple, Optional

import cv2
import numpy as np

from tests import synth

# -- frame codes ----------------------------------------------------------------
# the index sits in a 2 by 8 grid of blocks over the top half, lowest bit
# first, and its complement in the same grid over the bottom half. a
# misread breaks the complement, so decode_index raises instead of
# returning a wrong index.

CODE_BITS = 16
_COLS, _ROWS = 8, 2
_CELL = 16  # cell size on the grid decode_index reads at


def coded_frame(i: int, w: int = synth.W, h: int = synth.H) -> np.ndarray:
    """a w by h bgr frame that spells i in black and white blocks."""
    if not 0 <= i < 1 << CODE_BITS:
        raise ValueError(f"frame code out of range: {i}")
    bits = np.array([(i >> b) & 1 for b in range(CODE_BITS)], np.uint8)
    grid = bits.reshape(_ROWS, _COLS)
    cells = np.vstack([grid, 1 - grid]) * 255
    small = np.repeat(cells[..., None], 3, axis=2)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def decode_index(frame: np.ndarray) -> int:
    """the index coded_frame drew, read from a decoded frame of any size.
    raises ValueError when the frame carries no readable code."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rows = 2 * _ROWS
    grid = cv2.resize(gray, (_COLS * _CELL, rows * _CELL),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    # read the middle of each cell, away from codec ringing at the edges
    q = _CELL // 4
    cells = grid.reshape(rows, _CELL, _COLS, _CELL)[:, q:-q, :, q:-q]
    level = cells.mean(axis=(1, 3))
    on = level > 128
    if (np.abs(level - 128) < 48).any() or (on[:_ROWS] == on[_ROWS:]).any():
        raise ValueError(f"no frame code in this frame: {level.round()}")
    bits = on[:_ROWS].reshape(-1)
    return int(sum(1 << b for b in range(CODE_BITS) if bits[b]))


def write_coded(path, n: int, fps: Optional[float] = None,
                fourcc: str = "mp4v") -> None:
    """a clip of n coded frames at the synth frame size."""
    synth.write_clip(path, [coded_frame(i) for i in range(n)], fps, fourcc)


# -- mp4 boxes ------------------------------------------------------------------
# an mp4 or mov file is a tree of length prefixed boxes. edits rebuild the
# whole tree, so every ancestor size follows a payload that grew or
# shrank, and chunk offsets follow mdat wherever it lands.

CONTAINERS = frozenset({b"moov", b"trak", b"mdia", b"minf", b"stbl", b"edts",
                        b"dinf", b"udta", b"mvex"})


class Box(NamedTuple):
    kind: bytes
    start: int  # first header byte
    body: int   # first payload byte
    end: int    # one past the last byte

    @property
    def large(self) -> bool:
        return self.body - self.start == 16


def boxes(blob: bytes, start: int = 0, end: Optional[int] = None) -> Iterator[Box]:
    """boxes back to back in blob[start:end]. size 1 reads the 64 bit
    largesize, size 0 runs to end. stops at a header that does not fit."""
    end = len(blob) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, kind = struct.unpack_from(">I4s", blob, pos)
        body = pos + 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", blob, pos + 8)[0]
            body = pos + 16
        elif size == 0:
            size = end - pos
        if size < body - pos or pos + size > end:
            return
        yield Box(kind, pos, body, pos + size)
        pos += size


def walk(blob: bytes, start: int = 0, end: Optional[int] = None,
         depth: int = 0) -> Iterator[tuple[int, Box]]:
    """every box in file order with its depth, inside containers too."""
    for box in boxes(blob, start, end):
        yield depth, box
        if box.kind in CONTAINERS:
            yield from walk(blob, box.body, box.end, depth + 1)


def find_box(blob: bytes, kind: bytes, within: Optional[Box] = None
             ) -> Optional[Box]:
    """the first box of this kind, inside within when given."""
    start, end = (within.body, within.end) if within else (0, len(blob))
    return next((b for _, b in walk(blob, start, end) if b.kind == kind), None)


def payload(blob: bytes, box: Box) -> bytes:
    return blob[box.body:box.end]


def video_trak(blob: bytes) -> Box:
    """the trak whose handler is vide."""
    moov = find_box(blob, b"moov")
    for trak in boxes(blob, moov.body, moov.end) if moov else ():
        hdlr = find_box(blob, b"hdlr", trak) if trak.kind == b"trak" else None
        if hdlr and payload(blob, hdlr)[8:12] == b"vide":
            return trak
    raise ValueError("no video track")


def _header(box: Box, size: int) -> bytes:
    if box.large:
        return struct.pack(">I4sQ", 1, box.kind, 16 + size)
    return struct.pack(">I4s", 8 + size, box.kind)


def _emit(blob: bytes, box: Box, edits: dict, fix) -> bytes:
    if box.start in edits:
        body = edits[box.start]
    elif box.kind in CONTAINERS:
        kids = list(boxes(blob, box.body, box.end))
        last = kids[-1].end if kids else box.body
        # keep trailing bytes, like the zero word that ends a quicktime udta
        body = b"".join(_emit(blob, k, edits, fix) for k in kids) \
            + blob[last:box.end]
    elif box.kind in (b"stco", b"co64") and fix is not None:
        body = fix(box.kind, payload(blob, box))
    else:
        body = payload(blob, box)
    return _header(box, len(body)) + body


def rebuild(blob: bytes, edits: Optional[dict] = None,
            order: Optional[list[Box]] = None) -> bytes:
    """the file with new payloads for the leaf boxes in edits (keyed by
    box start) and its top level boxes in order. sizes are recomputed
    up the tree, and stco and co64 offsets move with the box they point
    into."""
    edits = edits or {}
    top = list(boxes(blob))
    tail = blob[top[-1].end:] if top else blob
    layout = order if order is not None else top
    # sizes first: chunk offsets keep their width, so this gives every start
    shift, pos = {}, 0
    for box in layout:
        shift[box.start] = pos - box.start
        pos += len(_emit(blob, box, edits, None))
    if not any(shift.values()):
        return b"".join(_emit(blob, b, edits, None) for b in layout) + tail
    starts = sorted(b.start for b in top)
    ends = {b.start: b.end for b in top}

    def moved(offset: int) -> int:
        owner = starts[bisect_right(starts, offset) - 1] if starts else 0
        if offset < owner or offset >= ends[owner] or owner not in shift:
            raise ValueError(f"chunk offset {offset} points outside the kept boxes")
        return offset + shift[owner]

    def fix(kind: bytes, body: bytes) -> bytes:
        count = struct.unpack_from(">I", body, 4)[0]
        fmt = ">%d%s" % (count, "I" if kind == b"stco" else "Q")
        offsets = struct.unpack_from(fmt, body, 8)
        size = struct.calcsize(fmt)
        return body[:8] + struct.pack(fmt, *map(moved, offsets)) + body[8 + size:]

    return b"".join(_emit(blob, b, edits, fix) for b in layout) + tail


# -- timing ---------------------------------------------------------------------
# field offsets inside each payload, by box version

def _times(kind: bytes, body: bytes) -> dict:
    """where each time field sits in an mvhd, mdhd or tkhd payload."""
    v1 = body[0] == 1
    if kind == b"tkhd":
        return {"duration": (28, ">Q") if v1 else (20, ">I")}
    return {"ctime": (4, ">Q") if v1 else (4, ">I"),
            "mtime": (12, ">Q") if v1 else (8, ">I"),
            "timescale": (20, ">I") if v1 else (12, ">I"),
            "duration": (24, ">Q") if v1 else (16, ">I")}


def _get(kind: bytes, body: bytes, field: str) -> int:
    at, fmt = _times(kind, body)[field]
    return struct.unpack_from(fmt, body, at)[0]


def _put(kind: bytes, body: bytes, **values) -> bytes:
    out = bytearray(body)
    for field, value in values.items():
        at, fmt = _times(kind, body)[field]
        struct.pack_into(fmt, out, at, value)
    return bytes(out)


def rewrite_stts(path, deltas, timescale: Optional[int] = None) -> np.ndarray:
    """show frame i of the video track for deltas[i] seconds. rewrites
    stts and the mdhd, tkhd, elst and mvhd durations to the new length.
    timescale replaces the media timescale when ticks need to be finer.
    meant for opencv mp4v files: one track, no b frames, no ctts.
    returns each frame's presentation time in seconds."""
    path = Path(path)
    blob = path.read_bytes()
    trak = video_trak(blob)
    if find_box(blob, b"ctts", trak) is not None:
        raise ValueError("rewrite_stts needs a file without ctts (b frames)")
    stts, mdhd, tkhd, elst = (find_box(blob, k, trak)
                              for k in (b"stts", b"mdhd", b"tkhd", b"elst"))
    mvhd = find_box(blob, b"mvhd")
    old_scale = _get(b"mdhd", payload(blob, mdhd), "timescale")
    media_scale = timescale or old_scale
    movie_scale = _get(b"mvhd", payload(blob, mvhd), "timescale")

    body = payload(blob, stts)
    entries = struct.unpack_from(">I", body, 4)[0]
    runs = struct.unpack_from(">%dI" % (2 * entries), body, 8)
    samples = sum(runs[0::2])
    if len(deltas) != samples:
        raise ValueError(f"{len(deltas)} deltas for {samples} frames")
    ticks = [round(d * media_scale) for d in deltas]
    if min(ticks) < 1:
        raise ValueError("every frame needs at least one tick")
    packed: list[list[int]] = []
    for d in ticks:
        if packed and packed[-1][1] == d:
            packed[-1][0] += 1
        else:
            packed.append([1, d])
    media = sum(ticks)
    movie = round(media * movie_scale / media_scale)

    edits = {
        stts.start: body[:4] + struct.pack(">I", len(packed))
        + b"".join(struct.pack(">II", n, d) for n, d in packed),
        mdhd.start: _put(b"mdhd", payload(blob, mdhd), timescale=media_scale,
                         duration=media),
        tkhd.start: _put(b"tkhd", payload(blob, tkhd), duration=movie),
        mvhd.start: _put(b"mvhd", payload(blob, mvhd), duration=movie),
    }
    if elst is not None:
        edits[elst.start] = _retime_elst(payload(blob, elst), movie,
                                         media_scale / old_scale)
    path.write_bytes(rebuild(blob, edits))
    return np.concatenate([[0], np.cumsum(ticks)[:-1]]) / media_scale


def _retime_elst(body: bytes, movie: int, media_ratio: float) -> bytes:
    """one edit that plays the media: its length becomes movie, and its
    media start follows a new media timescale. other lists stay as they are."""
    v1 = body[0] == 1
    count = struct.unpack_from(">I", body, 4)[0]
    fmt = ">Qq" if v1 else ">Ii"
    if count != 1:
        return body
    _, media_time = struct.unpack_from(fmt, body, 8)
    if media_time > 0:
        media_time = round(media_time * media_ratio)
    out = bytearray(body)
    struct.pack_into(fmt, out, 8, movie, media_time)
    return bytes(out)


def set_mvhd(path, version: int, ctime: Optional[int] = None,
             duration: Optional[int] = None) -> int:
    """rewrite mvhd as version 0 or 1 with ctime (seconds since 1904,
    also used as mtime) and duration (in mvhd timescale units). none
    keeps the current value. returns the mvhd timescale."""
    if version not in (0, 1):
        raise ValueError(f"mvhd version must be 0 or 1, not {version}")
    path = Path(path)
    blob = path.read_bytes()
    mvhd = find_box(blob, b"mvhd")
    body = payload(blob, mvhd)
    old = {f: _get(b"mvhd", body, f)
           for f in ("ctime", "mtime", "timescale", "duration")}
    rest = body[32:] if body[0] == 1 else body[20:]
    fmt = ">QQIQ" if version == 1 else ">IIII"
    new = bytes([version]) + body[1:4] + struct.pack(
        fmt, old["ctime"] if ctime is None else ctime,
        old["mtime"] if ctime is None else ctime, old["timescale"],
        old["duration"] if duration is None else duration) + rest
    path.write_bytes(rebuild(blob, {mvhd.start: new}))
    return old["timescale"]


# -- layout and display ---------------------------------------------------------

def faststart(blob: bytes) -> bytes:
    """moov moved ahead of the first mdat, like a web export, with chunk
    offsets shifted to match. a file already in that order comes back
    unchanged."""
    top = list(boxes(blob))
    kinds = [b.kind for b in top]
    if b"moov" not in kinds or b"mdat" not in kinds:
        raise ValueError("faststart needs a moov and an mdat")
    moov, first = kinds.index(b"moov"), kinds.index(b"mdat")
    if moov < first:
        return blob
    order = top[:first] + [top[moov]] + [b for b in top[first:] if b.kind != b"moov"]
    return rebuild(blob, order=order)


# a b c d of the display matrix, and where the turned picture is moved
# back into view, as ffmpeg and phones write them
_TURNS = {0: (1, 0, 0, 1), 90: (0, 1, -1, 0), 180: (-1, 0, 0, -1),
          270: (0, -1, 1, 0)}


def set_rotation(path, degrees: int) -> None:
    """display the video track turned degrees clockwise, the way a phone
    tags a portrait clip. the stored frames stay as they are."""
    degrees %= 360
    if degrees not in _TURNS:
        raise ValueError(f"rotation must be a multiple of 90, not {degrees}")
    path = Path(path)
    blob = path.read_bytes()
    tkhd = find_box(blob, b"tkhd", video_trak(blob))
    body = payload(blob, tkhd)
    at = 52 if body[0] == 1 else 40
    w, h = (v >> 16 for v in struct.unpack_from(">II", body, at + 36))
    tx, ty = {0: (0, 0), 90: (h, 0), 180: (w, h), 270: (0, w)}[degrees]
    a, b, c, d = _TURNS[degrees]
    matrix = struct.pack(">9i", a << 16, b << 16, 0, c << 16, d << 16, 0,
                         tx << 16, ty << 16, 1 << 30)
    new = body[:at] + matrix + body[at + 36:]
    path.write_bytes(rebuild(blob, {tkhd.start: new}))


# -- log footage ----------------------------------------------------------------

def slog3(linear):
    """sony slog3 encoding, linear reflectance to a 0..1 code value."""
    lin = np.asarray(linear, np.float64)
    hi = (420.0 + np.log10((np.maximum(lin, 0) + 0.01) / 0.19) * 261.5) / 1023.0
    lo = (lin * (171.2102946929 - 95.0) / 0.01125 + 95.0) / 1023.0
    return np.where(lin >= 0.01125, hi, lo)
