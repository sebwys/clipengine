# sequence.py
# orders clips into a chain by maximizing summed cut scores over the
# transition graph. greedy would lock in early mistakes and exhaustive
# search is factorial, so a beam search keeps the best b partial chains
# alive at each step: the standard middle ground.

import heapq
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from clipengine import config

COUNTRY_MODES = ("any", "same", "travel")


def build_chain(matrix: np.ndarray, seed_row: int, length: int,
                beam_width: int = 12,
                countries: Optional[list[str]] = None,
                country_mode: str = "any") -> tuple[list[int], list[float]]:
    """returns (rows, edge_scores): matrix row indices in play order and
    the score of each cut. country_mode 'same' keeps the seed's country,
    'travel' dampens consecutive cuts within one country to force variety."""
    n = matrix.shape[0]
    if country_mode not in COUNTRY_MODES:
        raise ValueError(f"unknown country mode: {country_mode!r},"
                         f" use {', '.join(COUNTRY_MODES)}")
    if beam_width < 1:
        raise ValueError(f"beam width must be at least 1, got {beam_width}")
    if countries is not None and len(countries) != n:
        raise ValueError(f"countries has {len(countries)} entries"
                         f" for {n} clips")
    if not 0 <= seed_row < n:
        raise IndexError(f"seed row {seed_row} outside matrix of {n}")
    length = max(1, min(length, n))
    beams: list[tuple[float, tuple[int, ...]]] = [(0.0, (seed_row,))]
    country_arr = np.asarray(countries) if countries is not None else None

    for _ in range(length - 1):
        candidates: list[tuple[float, tuple[int, ...]]] = []
        for total, path in beams:
            last = path[-1]
            # apply the country rules before taking the top b, so a
            # dampened home clip can lose its place to a foreign one
            row = matrix[last].astype(np.float64)
            # nan sorts first and passes the <= 0 stop, so it is no cut
            row[~np.isfinite(row)] = -1.0
            row[list(path)] = -1.0
            if country_arr is not None:
                if country_mode == "same":
                    row[country_arr != countries[seed_row]] = -1.0
                if country_mode == "travel":
                    row[country_arr == countries[last]] *= 0.6
            for j in np.argsort(row)[::-1][:beam_width]:
                if row[j] <= 0:
                    break
                candidates.append((total + float(row[j]), path + (int(j),)))
        if not candidates:
            break
        beams = heapq.nlargest(beam_width, candidates, key=lambda b: b[0])

    _, best_path = max(beams, key=lambda b: b[0])
    edges = [float(matrix[a, b]) for a, b in zip(best_path, best_path[1:])]
    return list(best_path), edges


def _m3u_text(meta_rows: list[dict]) -> str:
    """the playlist, one title line and one path line per clip. a raw cr
    or lf would split an entry, and a name that is not valid utf8 cannot
    be written, so those paths go out as percent encoded file urls.
    paths go out absolute, like the fcpxml urls"""
    lines = ["#EXTM3U"]
    for m in meta_rows:
        title = str(m["name"]).replace("\r", " ").replace("\n", " ")
        title = title.encode("utf-8", "replace").decode("utf-8")
        dur = m.get("duration_s") or 0
        if not np.isfinite(float(dur)):
            # an unreadable duration is unknown, like a missing one
            dur = 0
        lines.append(f"#EXTINF:{dur},{title}")
        path = str(m["path"])
        if not Path(path).is_absolute():
            # players read a relative entry against the exports folder
            path = str(Path(path).absolute())
        try:
            path.encode("utf-8")
            plain = "\r" not in path and "\n" not in path
        except UnicodeEncodeError:
            plain = False
        lines.append(path if plain else Path(path).as_uri())
    return "\n".join(lines) + "\n"


def export_chain(meta_rows: list[dict], edges: list[float], mode: str,
                 out_dir: Optional[Path] = None) -> tuple[Path, Path, Path]:
    """write the sequence three ways and return the paths: a json cut
    plan (machine readable), an m3u8 playlist (rough order preview in
    iina/vlc), and an fcpxml timeline (import into final cut pro or
    davinci resolve)."""
    from clipengine import fcpxml
    if out_dir is None:
        out_dir = config.EXPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan = {"mode": mode,
            "created": stamp,
            "total_score": round(sum(edges), 4),
            "clips": [{"order": i,
                       "id": m["id"],
                       "name": m["name"],
                       "country": m["country"],
                       "path": m["path"],
                       "cut_score_from_prev": (round(edges[i - 1], 4)
                                               if i else None)}
                      for i, m in enumerate(meta_rows)]}
    # build the texts first so a bad plan claims no name
    text = json.dumps(plan, indent=2)
    playlist = _m3u_text(meta_rows)
    # all three files share one claimed tag. each file goes in whole
    # through a temp file, and the timeline goes last
    tag = fcpxml.claim_tag(out_dir, stamp)
    json_path = out_dir / f"sequence_{tag}.json"
    m3u_path = out_dir / f"sequence_{tag}.m3u8"
    try:
        fcpxml.write_text_atomic(json_path, text)
        fcpxml.write_text_atomic(m3u_path, playlist)
        xml_path = fcpxml.export_fcpxml(meta_rows, mode, out_dir, tag)
    except BaseException:
        # a failed export leaves no partial files. the tag is ours alone,
        # so these names hold nothing anyone else wrote
        for p in (json_path, m3u_path, out_dir / f"sequence_{tag}.fcpxml"):
            p.unlink(missing_ok=True)
        raise
    return json_path, m3u_path, xml_path
