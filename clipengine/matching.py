# matching.py
# pairwise transition scoring. the unit being scored is always a cut:
# the end state of clip a against the start state of clip b. components
# are similarities in [0, 1]; a mode blends them with weights and a
# physical gate, so "momentum" cannot recommend a cut between two
# locked off shots and "whip" cannot fire on slow footage.

import json
import logging
import sqlite3
from typing import Optional

import numpy as np

from clipengine import catalog, config
from clipengine.features import (END_HUE, HUE_BINS, INDEX, START_HUE,
                                 VECTOR_LEN, from_bytes)

logger = logging.getLogger(__name__)

_EPS = 1e-4
COUNTRY_FILTERS = ("any", "same", "different")


class Library:
    """in memory matrix of every analyzed clip. row i of F belongs to
    ids[i] and meta[i]; row_of maps a clip id back to its row."""

    def __init__(self, ids: np.ndarray, F: np.ndarray, meta: list[dict]):
        self.ids = ids
        self.F = F
        self.meta = meta
        self.row_of = {int(cid): i for i, cid in enumerate(ids)}

    def __len__(self) -> int:
        return len(self.meta)


def _no_constant(name: str):
    # nan or infinity in a summary would break every strict json reply
    raise ValueError(f"summary holds {name}")


def _finite_float(text: str) -> float:
    value = float(text)
    if not np.isfinite(value):
        raise ValueError(f"summary holds {text}")
    return value


def load_library(conn: sqlite3.Connection,
                 country: Optional[str] = None) -> Library:
    """pull every fresh feature vector out of the catalog. a corrupt row
    costs its own clip, with a warning, never the whole library."""
    rows = catalog.analyzed(conn, country)
    ids, vecs, meta = [], [], []
    for r in rows:
        try:
            vec = from_bytes(r["vector"])
            if not np.isfinite(vec).all():
                raise ValueError("vector holds nan or infinity")
            summary = json.loads(r["summary"] or "{}",
                                 parse_constant=_no_constant,
                                 parse_float=_finite_float)
            if not isinstance(summary, dict):
                raise ValueError("summary is not a json object")
            for key in ("fps", "duration_s", "width", "height"):
                v = summary.get(key)
                if v is None:
                    continue
                if (isinstance(v, bool) or not isinstance(v, (int, float))
                        or not np.isfinite(float(v))):
                    raise ValueError(f"summary {key} is not a finite number")
        except (ValueError, TypeError, OverflowError) as exc:
            logger.warning("skipping clip %s (%s): %s",
                           r["id"], r["name"], exc)
            continue
        ids.append(r["id"])
        vecs.append(vec)
        meta.append({"id": r["id"], "name": r["name"],
                     "country": r["country"], "profile": r["profile"],
                     "path": r["path"],
                     "duration_s": summary.get("duration_s"),
                     "fps": summary.get("fps"),
                     "width": summary.get("width"),
                     "height": summary.get("height"),
                     "start_class": summary.get("start_class"),
                     "end_class": summary.get("end_class")})
    if not ids:
        return Library(np.zeros(0, np.int64),
                       np.zeros((0, VECTOR_LEN), np.float32), [])
    return Library(np.asarray(ids, np.int64), np.stack(vecs), meta)


def _col(F: np.ndarray, name: str) -> np.ndarray:
    return F[:, INDEX[name]]


def components(lib: Library, a_row: int) -> dict:
    """similarity components between clip a's end state and every clip's
    start state, vectorized over the whole library. arrays are (n,)."""
    F = lib.F
    a = F[a_row]
    ax, ay = a[INDEX["end_flow_x"]], a[INDEX["end_flow_y"]]
    ar, aj = a[INDEX["end_radial"]], a[INDEX["end_jitter"]]
    ae = a[INDEX["end_energy"]]
    bx, by = _col(F, "start_flow_x"), _col(F, "start_flow_y")
    br, bj = _col(F, "start_radial"), _col(F, "start_jitter")
    be = _col(F, "start_energy")

    # translation only counts when it is steadier than it is shaky, the
    # same rule the labels use: shake moves every pixel but goes nowhere
    a_steady = np.hypot(ax, ay) >= config.STEADY_RATIO * aj
    b_steady = np.hypot(bx, by) >= config.STEADY_RATIO * bj
    ax, ay = ax * a_steady, ay * a_steady
    bx, by = bx * b_steady, by * b_steady
    at, bt = np.hypot(ax, ay), np.hypot(bx, by)
    # radial under the push threshold is noise, the same as for the labels
    ar = ar * (np.abs(ar) >= config.PUSH_MIN)
    br = br * (np.abs(br) >= config.PUSH_MIN)

    # motion direction: cosine of the (flow_x, flow_y, radial) vectors
    # mapped to [0, 1], so a push in carries into a push in and a pull
    # out reads as its reverse. when either side is nearly still,
    # direction is meaningless, so the component goes neutral instead
    # of rewarding noise alignment.
    an = np.sqrt(ax * ax + ay * ay + ar * ar)
    bn = np.sqrt(bx * bx + by * by + br * br)
    dot = ax * bx + ay * by + ar * br
    cos = np.clip(dot / np.maximum(an * bn, _EPS), -1.0, 1.0)
    motion = (cos + 1.0) / 2.0
    still = ((ae < config.STATIC_MAX) | (be < config.STATIC_MAX)
             | (an < config.STATIC_MAX) | (bn < config.STATIC_MAX))
    motion = np.where(still, 0.5, motion)

    # energy: speed ratio through the cut, 1.0 when both sides move alike.
    # anything under the static threshold is a locked off shot, so both
    # sides floor there and sensor noise cannot split two stills
    floor = config.STATIC_MAX
    energy = np.exp(-np.abs(np.log(np.maximum(ae, floor)
                                   / np.maximum(be, floor))))

    # color: lab like distance on (luma, tint, warmth) plus palette overlap
    dl = a[INDEX["end_luma"]] - _col(F, "start_luma")
    da = a[INDEX["end_tint"]] - _col(F, "start_tint")
    db = a[INDEX["end_warmth"]] - _col(F, "start_warmth")
    de_sim = np.exp(-np.sqrt(dl * dl + da * da + db * db) / 40.0)
    hue_overlap = np.minimum(a[END_HUE][None, :], F[:, START_HUE]).sum(axis=1)
    color = 0.55 * de_sim + 0.45 * hue_overlap

    luma = 1.0 - np.abs(dl) / 255.0

    # physical gates per mode, each in [0, 1]
    gates = {
        "momentum": np.clip(np.minimum(ae, be) / config.PAN_MIN, 0.0, 1.0),
        # whip speed is steady translation, not energy: shake has the
        # energy of a whip with no camera move behind it. 0.8x keeps the
        # whip gate near whip speed on both sides
        "whip": np.clip(np.minimum(at, bt) / (0.8 * config.WHIP_MIN),
                        0.0, 1.0) ** 2,
        "calm": np.clip(1.0 - np.maximum(ae, be) / config.PAN_MIN, 0.0, 1.0),
    }
    gates["contrast"] = np.ones_like(motion)

    return {"motion": motion, "energy": energy, "color": color,
            "luma": luma, "gates": gates}


def _in_unit(x: np.ndarray) -> np.ndarray:
    """true where x is a finite value in [0, 1], with float slack."""
    with np.errstate(invalid="ignore"):
        return np.isfinite(x) & (x >= -_EPS) & (x <= 1.0 + _EPS)


def score_against(lib: Library, a_row: int,
                  mode: str = config.DEFAULT_MODE
                  ) -> tuple[np.ndarray, dict]:
    """score clip a's cut into every other clip under one mode."""
    if mode not in config.SCORING_MODES:
        raise ValueError(f"unknown mode: {mode}")
    w = config.SCORING_MODES[mode]
    c = components(lib, a_row)
    # contrast mode rewards a deliberate vibe flip, so color and
    # brightness similarities invert while energy still has to carry
    color = 1.0 - c["color"] if mode == "contrast" else c["color"]
    luma = 1.0 - c["luma"] if mode == "contrast" else c["luma"]
    raw = (w["motion"] * c["motion"] + w["energy"] * c["energy"]
           + w["color"] * color + w["luma"] * luma)
    scores = raw * c["gates"][mode]
    # a clip with any nan or infinity is no cut either way: nan would
    # sort first and an infinity can still score finite. a huge but
    # finite value shows up as a component outside [0, 1], or one that
    # overflows to nan, and that cut goes too, even when the mode gives
    # the component no weight. with every component in [0, 1] the
    # score is too, since the weights sum to one and gates stay in [0, 1]
    finite = np.isfinite(lib.F).all(axis=1)
    ok = finite & finite[a_row]
    for name in ("motion", "energy", "color", "luma"):
        ok &= _in_unit(c[name])
    scores = np.where(ok, scores, -1.0)
    scores[a_row] = -1.0
    return scores.astype(np.float32), c


def rank(lib: Library, clip_id: int, mode: str = config.DEFAULT_MODE,
         n: int = 10, country: str = "any") -> list[dict]:
    """top n candidate cuts out of clip_id, with score breakdowns."""
    if country not in COUNTRY_FILTERS:
        raise ValueError(f"unknown country filter: {country!r},"
                         f" use {', '.join(COUNTRY_FILTERS)}")
    if clip_id not in lib.row_of:
        raise KeyError(f"clip {clip_id} has no features yet")
    a_row = lib.row_of[clip_id]
    scores, c = score_against(lib, a_row, mode)
    a_country = lib.meta[a_row]["country"]
    out = []
    for i in np.argsort(scores)[::-1]:
        i = int(i)
        if len(out) >= n or scores[i] <= 0:
            break
        m = lib.meta[i]
        if country == "same" and m["country"] != a_country:
            continue
        if country == "different" and m["country"] == a_country:
            continue
        out.append({**m,
                    "score": round(float(scores[i]), 4),
                    "breakdown": {
                        "motion": round(float(c["motion"][i]), 3),
                        "energy": round(float(c["energy"][i]), 3),
                        "color": round(float(c["color"][i]), 3),
                        "luma": round(float(c["luma"][i]), 3),
                        "gate": round(float(c["gates"][mode][i]), 3)}})
    return out


def full_matrix(lib: Library, mode: str = config.DEFAULT_MODE) -> np.ndarray:
    """dense n x n cut score matrix; entry [i, j] scores the cut i -> j.
    the diagonal is -1 so no clip can follow itself."""
    n = len(lib)
    M = np.full((n, n), -1.0, dtype=np.float32)
    for i in range(n):
        M[i], _ = score_against(lib, i, mode)
    return M
