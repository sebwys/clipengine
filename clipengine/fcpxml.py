# fcpxml.py
# timeline handoff to real editors. final cut pro imports fcpxml
# natively and davinci resolve imports it via file > import timeline,
# so one exporter serves both. the tricky part is time: fcpxml times
# are rational numbers that must land exactly on frame boundaries, so
# every duration is expressed as frames * frame_duration, never as a
# float of seconds.

import math
import os
import re
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Optional

from clipengine import config

# whole rates that cameras also shoot at k*1000/1001; 25 and 50 have no
# ntsc twin, so a jittery 24.98 stays 25
_NTSC = (24, 30, 48, 60, 72, 96, 120, 240)


def frame_duration(fps: float) -> tuple[int, int]:
    """exact rational seconds per frame. a rate near an ntsc rate
    (k*1000/1001 for k in _NTSC) or a whole rate snaps to it, so a
    measured 29.92 is still 29.97. near means within 0.1 fps, or
    fps/300 for slow rates, so 1.05 is not 1. anything else keeps
    its own exact fraction, so 12.5 never turns into 12."""
    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = 24.0
    k = round(fps * 1.001)
    whole = round(fps)
    snaps = []
    if k in _NTSC:
        snaps.append((abs(fps - k / 1.001), 1001, k * 1000))
    if whole >= 1:
        snaps.append((abs(fps - whole), 100, whole * 100))
    if snaps:
        off, num, den = min(snaps)
        if off <= min(0.1, fps / 300):
            return num, den
    rate = Fraction(fps).limit_denominator(100)
    if not rate:
        # too slow to name as a fraction, treat it as unknown
        return 100, 2400
    return rate.denominator, rate.numerator


def _rational(frames: int, num: int, den: int) -> str:
    return f"{frames * num}/{den}s"


# xml 1.0 cannot carry most c0 controls or lone surrogates, not even
# escaped, so names lose them
_NOT_XML = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd"
                      "\U00010000-\U0010ffff]")


def xml_safe(text: str) -> str:
    return _NOT_XML.sub("", text)


def write_text_atomic(path: Path, text: str) -> None:
    """write utf8 text to a hidden temp file beside path and swap it in,
    so path holds either what it held before or all of the new text"""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def claim_tag(out_dir: Path, stamp: str) -> str:
    """reserve a free sequence_<tag> name and return the tag. every
    exporter claims by creating the empty fcpxml exclusively, so two
    exports in the same second, even concurrent ones, never share a
    name. a leftover json or m3u8 pushes the tag on too."""
    n = 1
    while True:
        tag = stamp if n == 1 else f"{stamp}_{n}"
        others = (out_dir / f"sequence_{tag}{ext}"
                  for ext in (".json", ".m3u8"))
        if not any(p.exists() for p in others):
            try:
                (out_dir / f"sequence_{tag}.fcpxml").open("x").close()
                return tag
            except FileExistsError:
                pass
        n += 1


def export_fcpxml(meta_rows: list[dict], mode: str,
                  out_dir: Optional[Path] = None,
                  stamp: Optional[str] = None) -> Path:
    """write a timeline of the given clips and return its path. with no
    stamp it claims a free name from the clock; a given stamp names the
    file outright and replaces whatever is there."""
    if out_dir is None:
        out_dir = config.EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    claimed = stamp is None
    if claimed:
        stamp = claim_tag(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    path = out_dir / f"sequence_{stamp}.fcpxml"
    try:
        write_text_atomic(path, timeline_text(meta_rows, mode, stamp))
    except BaseException:
        if claimed:
            # the name was ours alone, so nothing else is lost
            path.unlink(missing_ok=True)
        raise
    return path


def timeline_text(meta_rows: list[dict], mode: str, stamp: str) -> str:
    """the fcpxml for the given clips in order. each clip appears full
    length, floored to whole sequence frames; trimming stays in the
    editor where it belongs. media is referenced in place by file url,
    never copied."""
    root = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(root, "resources")

    formats: dict[tuple, str] = {}
    def format_id(fps: float, width: int, height: int) -> str:
        num, den = frame_duration(fps)
        key = (num, den, width, height)
        if key not in formats:
            fid = f"r{len(formats) + 1}"
            formats[key] = fid
            attrs = {"id": fid, "frameDuration": f"{num}/{den}s"}
            if width and height:
                attrs["width"] = str(width)
                attrs["height"] = str(height)
            ET.SubElement(resources, "format", attrs)
        return formats[key]

    clips = []
    for i, m in enumerate(meta_rows):
        fps = float(m.get("fps") or 24.0)
        num, den = frame_duration(fps)
        fid = format_id(fps, int(m.get("width") or 0),
                        int(m.get("height") or 0))
        secs = float(m.get("duration_s") or 1.0)
        if not math.isfinite(secs):
            # an unreadable duration exports like a missing one
            secs = 1.0
        frames = max(1, round(secs * den / num))
        duration = _rational(frames, num, den)
        if i == 0:
            # the first clip sets the sequence rate
            seq_num, seq_den = num, den
        # cut length in whole sequence frames, floored so it stays
        # inside its media. a clip under one sequence frame still gets one
        seq_frames = max(1, (frames * num * seq_den) // (den * seq_num))
        aid = f"a{i + 1}"
        name = xml_safe(str(m["name"]))
        asset = ET.SubElement(
            resources, "asset",
            {"id": aid, "name": name, "start": "0s",
             "duration": duration, "hasVideo": "1", "hasAudio": "1",
             "format": fid})
        ET.SubElement(asset, "media-rep",
                      {"kind": "original-media",
                       "src": Path(m["path"]).absolute().as_uri()})
        clips.append((aid, name,
                      _rational(seq_frames, seq_num, seq_den)))

    library = ET.SubElement(root, "library")
    event = ET.SubElement(library, "event", name="ClipEngine")
    project = ET.SubElement(event, "project",
                            name=xml_safe(f"clipengine {mode} {stamp}"))
    first_format = next(iter(formats.values())) if formats else "r1"
    seq = ET.SubElement(project, "sequence", format=first_format)
    spine = ET.SubElement(seq, "spine")
    for aid, name, duration in clips:
        # no offsets: spine children without offsets lay out end to end,
        # which both fcp and resolve honor
        ET.SubElement(spine, "asset-clip",
                      {"ref": aid, "name": name, "duration": duration})

    body = ET.tostring(root, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE fcpxml>\n\n" + body + "\n")
