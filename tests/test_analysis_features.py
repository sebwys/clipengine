# test_analysis_features.py
# what one clip turns into: the feature vector, the summary and the
# thumbnails. motion reads the same at any fps and resolution, labels
# sit exactly on their thresholds, short clips still analyze, thumbnails
# come out THUMB_WIDTH wide with their own window's normalization, a
# short clean whip is not shake while body sway still is, the hue palette
# has no cliffs at its bin edges, the end state of a cut short file is
# its last full window that decodes, relabel is stable, and a fixed clip
# still gives the golden vector of its FEATURE_VERSION.

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from clipengine import analysis, config, features, matching, reader
from tests import synth
from tests.media_fixtures import coded_frame, decode_index, faststart, find_box
from tests.test_matching import lib_of, make_vec
from tests.util import TempDirsMixin

SPEED = 0.45  # widths per second, mid zone between PAN_MIN and WHIP_MIN
DIRECTIONAL = ("pan_right", "whip_right")
SUMMARY_KEYS = {"duration_s", "fps", "width", "height", "codec", "created",
                "flat", "start_class", "end_class", "start", "end",
                "tempo_mean", "tempo_var", "energy_profile"}
WINDOW_KEYS = {"class", "flow", "energy", "radial", "jitter", "luma",
               "contrast", "sat", "warmth", "tint", "flat"}

# the vector of the TestAnalyzedClip clip (static then pan right, 5 s at
# 24 fps, mp4v) per FEATURE_VERSION. a change to the extractor that moves
# any slot past GOLDEN_TOL needs a version bump and a new golden here
GOLDEN_TOL = 1e-3
GOLDEN = {
    2: {
        "start_flow_x": -2.5e-05, "start_flow_y": -6e-06,
        "start_energy": 0.000234, "start_radial": -2e-06,
        "start_jitter": 9.6e-05, "end_flow_x": -0.450412,
        "end_flow_y": -2e-06, "end_energy": 0.450416, "end_radial": 0.000187,
        "end_jitter": 0.000849, "start_luma": 130.554199,
        "start_contrast": 69.0, "start_sat": 104.337288,
        "start_warmth": -0.276184, "start_tint": 4.982956,
        "end_luma": 137.063583, "end_contrast": 129.0, "end_sat": 170.810883,
        "end_warmth": 0.162949, "end_tint": 12.087418,
        "global_luma": 134.069778, "global_sat": 139.147293,
        "global_warmth": -0.037132, "tempo_mean": 0.225537,
        "tempo_var": 0.050808, "duration_s": 5.0, "start_hue_0": 0.079372,
        "start_hue_1": 0.098193, "start_hue_2": 0.077858,
        "start_hue_3": 0.084822, "start_hue_4": 0.050503,
        "start_hue_5": 0.076806, "start_hue_6": 0.071911,
        "start_hue_7": 0.088718, "start_hue_8": 0.086024,
        "start_hue_9": 0.097936, "start_hue_10": 0.102772,
        "start_hue_11": 0.085086, "end_hue_0": 0.080042,
        "end_hue_1": 0.095902, "end_hue_2": 0.075687, "end_hue_3": 0.081732,
        "end_hue_4": 0.049706, "end_hue_5": 0.073352, "end_hue_6": 0.069372,
        "end_hue_7": 0.088853, "end_hue_8": 0.090497, "end_hue_9": 0.102416,
        "end_hue_10": 0.104534, "end_hue_11": 0.087907,
    },
}


def golden_misses(vec: np.ndarray) -> list[str]:
    """the slots of vec that miss the golden for the current
    FEATURE_VERSION by more than GOLDEN_TOL. fails outright when that
    version has no golden yet."""
    version = config.FEATURE_VERSION
    if version not in GOLDEN:
        raise AssertionError(
            f"no golden vector for FEATURE_VERSION {version}; analyze the"
            " TestAnalyzedClip clip and add its vector to GOLDEN as a new"
            " golden")
    gold = GOLDEN[version]
    if set(gold) != set(features.FIELDS):
        raise AssertionError("golden fields differ from features.FIELDS;"
                             " the layout changed, add a new golden")
    return [f"{name}: {features.get(vec, name):.6f} vs {want}"
            for name, want in gold.items()
            if not abs(features.get(vec, name) - want) <= GOLDEN_TOL]


def shifted(img: np.ndarray, dx: float) -> np.ndarray:
    """img moved dx pixels right, subpixel, wrapping at the edges."""
    h, w = img.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, 0]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_WRAP)


def hold_then_whip(moving_pairs: int, n: int = 29, px: int = 14,
                   block: int = 20, at_end: bool = True) -> list[np.ndarray]:
    """one scene held still, then whipped right px per frame over the
    last moving_pairs frame pairs. at_end False whips first, then holds."""
    steps = [0] * (n - 1 - moving_pairs) + [px] * moving_pairs
    if not at_end:
        steps.reverse()
    tex = synth.texture(7, block=block)
    shifts = np.concatenate([[0], np.cumsum(steps)])
    return [np.roll(tex, -int(s), axis=1) for s in shifts]


def walking_frames(n: int = 29, drift: float = 0.08, shake: float = 0.27,
                   seed: int = 3) -> list[np.ndarray]:
    """a walking shot: slow drift right under random frame to frame shake,
    both in widths per second, like the fx30 walking shot in test_analysis_motion.py."""
    rng = np.random.default_rng(seed)
    per_frame = synth.W / synth.FPS
    steps = (drift + rng.normal(0.0, shake, n - 1)) * per_frame
    xs = np.concatenate([[0.0], np.cumsum(steps)])
    tex = synth.texture(7)
    return [shifted(tex, -x) for x in xs]


def swaying_frames(hz: float, drift: float, sway: float, phase: float,
                   n: int = 29) -> list[np.ndarray]:
    """a walking shot the way a body moves: slow drift right under a smooth
    side to side sway of hz cycles a second, both in widths per second."""
    t = np.arange(n - 1) / synth.FPS
    speed = drift + sway * np.sin(2 * np.pi * hz * t + phase)
    xs = np.concatenate([[0.0], np.cumsum(speed * synth.W / synth.FPS)])
    tex = synth.texture(7)
    return [shifted(tex, -x) for x in xs]


def jpeg(blob: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(blob, np.uint8), cv2.IMREAD_COLOR)


def sat(img: np.ndarray) -> float:
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean())


def overlap(a: np.ndarray, b: np.ndarray) -> float:
    """palette overlap the way matching scores it."""
    return float(np.minimum(a, b).sum())


def solid_hue(h: int) -> np.ndarray:
    """hue histogram of a flat color at opencv hue h."""
    bgr = cv2.cvtColor(np.uint8([[[h, 200, 180]]]), cv2.COLOR_HSV2BGR)[0, 0]
    frames = synth.color_frames(tuple(int(x) for x in bgr), n=6, noise=0)
    return analysis.color_stats(frames, force_log=False)["hue"]


def spread_hue(mean: float, seed: int, sd: float = 4.0) -> np.ndarray:
    """hue histogram of a narrow palette, hue normal around mean."""
    rng = np.random.default_rng(seed)
    h = np.mod(rng.normal(mean, sd, (synth.H, synth.W)), 180).astype(np.uint8)
    s = np.full_like(h, 200)
    v = np.full_like(h, 180)
    bgr = cv2.cvtColor(np.dstack([h, s, v]), cv2.COLOR_HSV2BGR)
    return analysis.color_stats([bgr] * 6, force_log=False)["hue"]


def two_phase(fps: float, px: int, seconds: float = 1.5) -> list[np.ndarray]:
    """static, then pan right at px per frame, each phase seconds long."""
    n = int(round(seconds * fps))
    return (synth.frames_for("static", n=n)
            + synth.frames_for("pan_right", n=n, px=px))


def windows_seen(path, is_log: bool = False):
    """analyze path and return the result plus the frames of every window
    analyze_clip read, start first."""
    seen = []
    real = reader.read_window

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out[0])
        return out

    with mock.patch.object(reader, "read_window", spy):
        result = analysis.analyze_clip(path, is_log=is_log)
    return result, seen


class ClipDir(unittest.TestCase):
    """one temp folder per class for clips written once in setUpClass."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._dir = tempfile.TemporaryDirectory(prefix="clipengine-test-")
        cls.dir = Path(cls._dir.name)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()
        super().tearDownClass()


# -- motion math ------------------------------------------------------------------

class TestMotionUnits(unittest.TestCase):
    """flow is in frame widths per second, so a camera speed reads the same
    whatever the frame rate or the source size."""

    def test_translation_is_fps_independent(self):
        tex = synth.texture(7)
        for fps in (24.0, 30.0, 60.0, 120.0):
            with self.subTest(fps=fps):
                # frames as the reader keeps them: whole frames about 1/30 s apart
                k = max(1, round(fps / 30))
                n = min(config.MAX_WINDOW_FRAMES,
                        round(config.WINDOW_SECONDS * fps / k))
                times = [i * k / fps for i in range(n)]
                frames = [shifted(tex, -SPEED * synth.W * t) for t in times]
                m = analysis.window_motion(frames, fps, times)
                self.assertEqual(m.label, "pan_right")
                self.assertAlmostEqual(m.flow_x, -SPEED, delta=0.05 * SPEED)
                self.assertAlmostEqual(m.energy, SPEED, delta=0.05 * SPEED)

    def test_translation_is_resolution_independent(self):
        rng = np.random.default_rng(7)
        for w, h in ((640, 360), (1280, 720), (1920, 1080), (3840, 2160),
                     (1080, 1920)):
            with self.subTest(size=f"{w}x{h}"):
                # square blocks 10 px wide once downscaled, at any aspect
                grid = (round(32 * h / w), 32, 3)
                big = cv2.resize(rng.integers(0, 255, grid, np.uint8), (w, h),
                                 interpolation=cv2.INTER_NEAREST)
                px = round(SPEED * w / synth.FPS)
                frames = [reader._downscale(np.roll(big, -i * px, axis=1),
                                            config.ANALYSIS_WIDTH)
                          for i in range(8)]
                self.assertEqual(frames[0].shape[1], config.ANALYSIS_WIDTH)
                m = analysis.window_motion(frames, synth.FPS)
                truth = -px * synth.FPS / w
                self.assertAlmostEqual(m.flow_x, truth, delta=0.02 * SPEED)


class TestClassifyBoundaries(unittest.TestCase):
    """each threshold is a closed edge: a value exactly on it takes the
    label above it. nudges are relative, so a retuned config still holds."""

    def label(self, **kw) -> str:
        m = analysis.WindowMotion(
            flow_x=kw.get("flow_x", 0.0), flow_y=kw.get("flow_y", 0.0),
            energy=kw.get("energy", 0.3), radial=kw.get("radial", 0.0),
            jitter=kw.get("jitter", 0.0))
        return analysis.classify_motion(m)

    def test_classify_boundaries(self):
        up, down = 1 + 1e-9, 1 - 1e-9
        pan, whip = config.PAN_MIN, config.WHIP_MIN
        mid = (config.PAN_MIN + config.WHIP_MIN) / 2
        t = 0.1  # translation for the radial dominance edge
        cases = [
            ("static below STATIC_MAX", dict(energy=config.STATIC_MAX * down),
             "static"),
            ("moving at STATIC_MAX", dict(energy=config.STATIC_MAX), "drift"),
            ("pan at PAN_MIN", dict(flow_x=-pan), "pan_right"),
            ("drift below PAN_MIN", dict(flow_x=-pan * down), "drift"),
            ("whip at WHIP_MIN", dict(flow_x=-whip, energy=whip), "whip_right"),
            ("pan below WHIP_MIN", dict(flow_x=-whip * down, energy=whip),
             "pan_right"),
            ("steady exactly at STEADY_RATIO",
             dict(flow_x=-config.STEADY_RATIO * 0.4, jitter=0.4), "pan_right"),
            ("steady just inside STEADY_RATIO",
             dict(flow_x=-mid, jitter=mid / config.STEADY_RATIO * down),
             "pan_right"),
            ("unsteady past STEADY_RATIO",
             dict(flow_x=-mid, jitter=mid / config.STEADY_RATIO * up), "drift"),
            ("handheld past 1.5x translation",
             dict(flow_x=-pan / 2, jitter=1.5 * pan / 2 * up), "handheld"),
            ("drift at 1.5x translation",
             dict(flow_x=-pan / 2, jitter=1.5 * pan / 2 * down), "drift"),
            ("pan at AXIS_DOMINANCE",
             dict(flow_x=-mid, flow_y=-mid / config.AXIS_DOMINANCE), "pan_right"),
            ("diagonal past AXIS_DOMINANCE",
             dict(flow_x=-mid, flow_y=-mid / config.AXIS_DOMINANCE * up),
             "move_diagonal"),
            ("tilt at AXIS_DOMINANCE",
             dict(flow_y=mid, flow_x=mid / config.AXIS_DOMINANCE), "tilt_up"),
            ("diagonal whip", dict(flow_x=-whip, flow_y=-whip, energy=1.5),
             "whip_diagonal"),
            ("push at PUSH_MIN", dict(radial=config.PUSH_MIN), "push_in"),
            ("pull at PUSH_MIN", dict(radial=-config.PUSH_MIN), "pull_out"),
            ("no push below PUSH_MIN", dict(radial=config.PUSH_MIN * down),
             "drift"),
            ("push at RADIAL_DOMINANCE",
             dict(flow_x=-t, radial=config.RADIAL_DOMINANCE * t * up), "push_in"),
            ("pan below RADIAL_DOMINANCE",
             dict(flow_x=-t, radial=config.RADIAL_DOMINANCE * t * down),
             "pan_right"),
            ("push beats whip",
             dict(flow_x=-whip, energy=1.5,
                  radial=config.RADIAL_DOMINANCE * whip * up), "push_in"),
        ]
        for name, kw, want in cases:
            with self.subTest(name):
                self.assertEqual(self.label(**kw), want, kw)


class TestShortWhipIsNotShake(unittest.TestCase):
    """steadiness asks whether a window shakes, not whether its speed
    changed. one clean start or stop inside the window is not shake."""

    def test_short_clean_whip_is_not_handheld(self):
        # a 0.29 s whip into the cut, and its mirror out of a cut. the
        # window mean cannot reach WHIP_MIN, so pan_right is enough
        for at_end in (True, False):
            with self.subTest(at_end=at_end):
                m = analysis.window_motion(
                    hold_then_whip(7, at_end=at_end), synth.FPS)
                self.assertLess(m.flow_x, -config.PAN_MIN)
                self.assertIn(m.label, DIRECTIONAL,
                              f"clean 0.29 s whip read as {m.label} (t "
                              f"{abs(m.flow_x):.3f}, jitter {m.jitter:.3f})")

    def test_real_shake_stays_unsteady(self):
        m = analysis.window_motion(
            synth.frames_for("handheld", n=29, px=4), synth.FPS)
        self.assertEqual(m.label, "handheld")
        m = analysis.window_motion(walking_frames(), synth.FPS)
        self.assertNotIn(m.label, DIRECTIONAL,
                         f"walking shot read as a pan (t {abs(m.flow_x):.3f},"
                         f" jitter {m.jitter:.3f})")
        self.assertLess(np.hypot(m.flow_x, m.flow_y),
                        config.STEADY_RATIO * m.jitter)

    def test_walking_sway_stays_unsteady(self):
        # body sway is smooth from frame to frame, but it is still shake:
        # the window wobbles more than it travels
        for hz in (0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0):
            for drift, sway in ((0.10, 0.30), (0.15, 0.40)):
                with self.subTest(hz=hz, drift=drift, sway=sway):
                    m = analysis.window_motion(
                        swaying_frames(hz, drift, sway, phase=0.7), synth.FPS)
                    t = float(np.hypot(m.flow_x, m.flow_y))
                    self.assertNotIn(m.label, DIRECTIONAL,
                                     f"sway read as a pan (t {t:.3f},"
                                     f" jitter {m.jitter:.3f})")
                    self.assertLess(t, config.STEADY_RATIO * m.jitter)

    def test_slow_sway_is_never_one_step(self):
        # below 1 hz half a cycle can carry most of the variance, like one
        # change of speed. at any phase sway keeps its full spread
        spy = mock.patch.object(analysis, "_jitter", wraps=analysis._jitter)
        for hz in (0.7, 0.8, 0.9, 1.0, 1.5, 2.0, 3.0):
            for drift, sway in ((0.10, 0.30), (0.15, 0.40)):
                for phase in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                    with self.subTest(hz=hz, drift=drift, sway=sway,
                                      phase=phase), spy as jit:
                        m = analysis.window_motion(
                            swaying_frames(hz, drift, sway, phase), synth.FPS)
                        x = np.column_stack(jit.call_args.args)
                        plain = float(np.sqrt(np.mean(
                            np.sum((x - x.mean(axis=0)) ** 2, axis=1))))
                        self.assertAlmostEqual(m.jitter, plain, places=9)

    def test_jitter_reads_shake_at_its_spread(self):
        # shake that is random per frame reads at its own spread, the
        # plain deviation of the per pair translation
        rng = np.random.default_rng(5)
        for sd in (0.05, 0.3):
            with self.subTest(sd=sd):
                txs = list(0.2 + rng.normal(0.0, sd, 28))
                tys = list(rng.normal(0.0, sd / 2, 28))
                self.assertAlmostEqual(analysis._jitter(txs, tys),
                                       float(np.hypot(np.std(txs),
                                                      np.std(tys))),
                                       delta=1e-9)
        # one clean start or stop is not shake, only the noise around it is
        for at in (4, 14, 24):
            with self.subTest(step_at=at):
                txs = [0.0] * at + [1.2] * (28 - at)
                txs = list(np.add(txs, rng.normal(0.0, 0.02, 28)))
                self.assertLess(analysis._jitter(txs, [0.0] * 28), 0.03)
        # but a single jolt is a bump, not a start, even on the last pair
        for at in (0, 27):
            with self.subTest(jolt_at=at):
                txs = [0.3] * 28
                txs[at] = 1.3
                self.assertAlmostEqual(analysis._jitter(txs, [0.0] * 28),
                                       float(np.std(txs)), delta=1e-9)


# -- color math -------------------------------------------------------------------

class TestColorStats(unittest.TestCase):
    def test_tint_and_contrast_signs(self):
        green = analysis.color_stats(synth.color_frames((40, 200, 40)), False)
        magenta = analysis.color_stats(synth.color_frames((200, 40, 200)), False)
        self.assertLess(green["tint"], -20)
        self.assertGreater(magenta["tint"], 20)
        tex = synth.texture(7)
        stats = analysis.color_stats([tex] * 6, force_log=False)
        self.assertFalse(stats["flat"])
        l_chan = cv2.cvtColor(tex, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
        p10, p90 = np.percentile(l_chan, (10, 90))
        self.assertAlmostEqual(stats["contrast"], float(p90 - p10), places=3)

    def test_hue_overlap_has_no_bin_edge_cliffs(self):
        near = overlap(solid_hue(58), solid_hue(61))
        far = overlap(solid_hue(58), solid_hue(47))
        self.assertGreaterEqual(near, far, "hue 61 overlaps 58 less than 47 does")
        self.assertGreater(overlap(solid_hue(178), solid_hue(2)), 0.3,
                           "red wrap neighbors share no palette")
        inside = overlap(spread_hue(4, 1), spread_hue(10, 2))
        across = overlap(spread_hue(12, 1), spread_hue(18, 2))
        self.assertLess(abs(inside - across), 0.2,
                        f"inside a bin {inside:.3f}, across an edge {across:.3f}")

    def test_hue_on_a_bin_center_fills_that_bin(self):
        # bins center on the middle of the hues they held as hard bins:
        # 7 for 0 to 14, 22 for 15 to 29, 172 for 165 to 179
        width = 180 // features.HUE_BINS
        for b in (0, 1, 3, features.HUE_BINS - 1):
            h = b * width + width // 2
            with self.subTest(hue=h):
                self.assertAlmostEqual(float(solid_hue(h)[b]), 1.0, places=3)
        # either side of a center leans the same amount
        left, right = solid_hue(22 - 4), solid_hue(22 + 4)
        self.assertAlmostEqual(float(left[0]), float(right[2]), places=3)

    def test_hue_histogram_stays_a_distribution(self):
        for h in (0, 7, 15, 58, 90, 172, 179):
            with self.subTest(hue=h):
                hist = solid_hue(h)
                self.assertEqual(hist.shape, (features.HUE_BINS,))
                self.assertEqual(hist.dtype, np.float32)
                self.assertAlmostEqual(float(hist.sum()), 1.0, places=5)
                self.assertGreaterEqual(float(hist.min()), 0.0)


# -- one analyzed clip -------------------------------------------------------------

class TestAnalyzedClip(ClipDir):
    """a 5 s clip, static for 2.5 s and then panning right at 0.45 w/s,
    analyzed twice as is and once as log."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.path = cls.dir / "static_pan.mp4"
        synth.make_clip(cls.path, ["static", "pan_right"], n_each=60)
        cls.result = analysis.analyze_clip(cls.path, is_log=False)
        cls.again = analysis.analyze_clip(cls.path, is_log=False)
        cls.log = analysis.analyze_clip(cls.path, is_log=True)

    def test_energy_profile_tracks_static_then_pan(self):
        prof = analysis.energy_profile(self.path, 5.0)
        self.assertEqual(prof.dtype, np.float32)
        self.assertEqual(len(prof), config.ENERGY_SAMPLES)
        for i, e in enumerate(prof[:4]):
            with self.subTest(sample=i):
                self.assertLess(e, config.STATIC_MAX)
        for i, e in enumerate(prof[4:], 4):
            with self.subTest(sample=i):
                self.assertAlmostEqual(float(e), SPEED, delta=0.05 * SPEED)

    def test_energy_profile_nonpositive_duration_is_empty(self):
        for duration in (0.0, -1.0):
            with self.subTest(duration=duration):
                prof = analysis.energy_profile(self.path, duration)
                self.assertEqual(prof.dtype, np.float32)
                self.assertEqual(prof.size, 0)

    def test_analyze_vector_contract(self):
        vec, s = self.result.vector, self.result.summary
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(vec.shape, (features.VECTOR_LEN,))
        self.assertEqual(features.VECTOR_LEN, 50)
        self.assertTrue(np.all(np.isfinite(vec)))
        self.assertAlmostEqual(features.get(vec, "duration_s"),
                               s["duration_s"], delta=1e-3)
        self.assertAlmostEqual(s["duration_s"], 5.0, delta=0.1)
        for part in (features.START_HUE, features.END_HUE):
            self.assertAlmostEqual(float(vec[part].sum()), 1.0, places=5)
        self.assertEqual(len(s["energy_profile"]), config.ENERGY_SAMPLES)
        self.assertAlmostEqual(features.get(vec, "tempo_mean"),
                               float(np.mean(s["energy_profile"])), delta=1e-3)
        self.assertAlmostEqual(s["tempo_mean"],
                               features.get(vec, "tempo_mean"), delta=1e-3)
        self.assertEqual(set(s), SUMMARY_KEYS)
        self.assertEqual(set(s["start"]), WINDOW_KEYS)
        self.assertEqual(set(s["end"]), WINDOW_KEYS)
        self.assertEqual(set(self.result.thumbs), {"start", "mid", "end"})
        json.dumps(s, allow_nan=False)

    def test_labels_from_vector_match_summary(self):
        vec, s = self.result.vector, self.result.summary
        self.assertEqual((s["start_class"], s["end_class"]),
                         ("static", "pan_right"))
        self.assertEqual(analysis.motion_from_vector(vec, "start").label,
                         s["start_class"])
        self.assertEqual(analysis.motion_from_vector(vec, "end").label,
                         s["end_class"])
        self.assertEqual(s["start"]["class"], s["start_class"])
        self.assertEqual(s["end"]["class"], s["end_class"])

    def test_analysis_is_deterministic(self):
        self.assertEqual(features.to_bytes(self.result.vector),
                         features.to_bytes(self.again.vector))
        self.assertEqual(self.result.summary, self.again.summary)

    def test_golden_vector_matches_feature_version(self):
        self.assertEqual(golden_misses(self.result.vector), [])

    def test_unknown_feature_version_asks_for_a_new_golden(self):
        unknown = max(GOLDEN) + 1
        with mock.patch.object(config, "FEATURE_VERSION", unknown):
            with self.assertRaisesRegex(AssertionError, "new golden"):
                golden_misses(self.result.vector)

    def test_a_moved_slot_misses_the_golden(self):
        vec = self.result.vector.copy()
        vec[features.INDEX["end_flow_x"]] += 2 * GOLDEN_TOL
        self.assertEqual(len(golden_misses(vec)), 1)
        self.assertIn("end_flow_x", golden_misses(vec)[0])

    def test_is_log_sets_flat_and_vivid_is_not_flat(self):
        self.assertTrue(self.log.summary["flat"])
        self.assertFalse(self.result.summary["flat"])
        # log thumbnails get the display stretch, plain ones do not
        for pos in ("start", "mid", "end"):
            with self.subTest(pos=pos):
                self.assertGreater(sat(jpeg(self.log.thumbs[pos])),
                                   sat(jpeg(self.result.thumbs[pos])) + 10)


class TestClipShapes(ClipDir):
    def test_two_phase_clip_at_25_and_30_fps(self):
        for fps in (25.0, 30.0):
            with self.subTest(fps=fps):
                path = self.dir / f"two_phase_{fps:g}.mp4"
                synth.write_clip(path, two_phase(fps, px=2), fps)
                s = analysis.analyze_clip(path, is_log=False).summary
                self.assertEqual(s["start_class"], "static", s["start"])
                self.assertEqual(s["end_class"], "pan_right", s["end"])
                self.assertAlmostEqual(s["duration_s"], 3.0, delta=0.1)
                self.assertAlmostEqual(s["fps"], fps, places=2)

    def test_mxf_mpeg2_analyzes(self):
        path = self.dir / "broadcast.mxf"
        frames = (synth.frames_for("static", n=60)
                  + synth.frames_for("pan_right", n=60))
        try:
            # ffmpeg warns that mxf has no mpg2 tag, then writes mpeg2 anyway
            with synth._quiet_stderr(True):
                synth.write_clip(path, frames, fourcc="MPG2")
        except synth.WriterUnavailable as err:
            self.skipTest(str(err))
        s = analysis.analyze_clip(path, is_log=False).summary
        self.assertEqual((s["start_class"], s["end_class"]),
                         ("static", "pan_right"))
        self.assertAlmostEqual(s["duration_s"], 5.0, delta=0.1)

    def test_short_clips_give_finite_vectors(self):
        tex = synth.texture(7)
        one = self.dir / "one.mp4"
        synth.write_clip(one, [tex])
        with self.assertRaises(reader.ClipReadError):
            analysis.analyze_clip(one, is_log=False)
        for n in (2, 3, 12):
            with self.subTest(frames=n):
                path = self.dir / f"short_{n}.mp4"
                synth.write_clip(path, [np.roll(tex, -3 * i, axis=1)
                                        for i in range(n)])
                r = analysis.analyze_clip(path, is_log=False)
                self.assertTrue(np.all(np.isfinite(r.vector)), r.summary)
                self.assertEqual(len(r.summary["energy_profile"]),
                                 config.ENERGY_SAMPLES)
                json.dumps(r.summary, allow_nan=False)


# -- thumbnails -------------------------------------------------------------------

class TestThumbnails(ClipDir):
    def test_thumbs_reach_thumb_width(self):
        # each thumb is the middle frame of its window, decoded wide enough
        # for THUMB_WIDTH, and never upscaled past the source. color stats
        # still see only ANALYSIS_WIDTH frames, the mid sample included
        real, widths = analysis.color_stats, []

        def spy(frames, force_log):
            widths.extend(f.shape[1] for f in frames)
            return real(frames, force_log)

        def still(frames, fps, times=None):
            return analysis.WindowMotion(0.0, 0.0, 0.0, 0.0, 0.0, "static")

        def flat(path, duration_s):
            return np.zeros(config.ENERGY_SAMPLES, np.float32)

        for w, h in ((640, 360), (360, 640), (400, 300), (160, 120)):
            with self.subTest(source=f"{w}x{h}"):
                path = self.dir / f"coded_{w}x{h}.mp4"
                synth.write_clip(path, [coded_frame(i, w, h) for i in range(36)])
                widths.clear()
                # flow is not under test here and costs most of the time
                with mock.patch.object(analysis, "color_stats", spy), \
                        mock.patch.object(analysis, "window_motion", still), \
                        mock.patch.object(analysis, "energy_profile", flat):
                    result, seen = windows_seen(path)
                self.assertEqual(set(widths), {min(w, config.ANALYSIS_WIDTH)})
                want_w = min(w, config.THUMB_WIDTH)
                want_h = round(h * want_w / w)
                for pos, blob in result.thumbs.items():
                    self.assertEqual(jpeg(blob).shape[:2], (want_h, want_w),
                                     f"{pos} thumb")
                for pos, frames in (("start", seen[0]), ("end", seen[-1])):
                    self.assertEqual(
                        decode_index(jpeg(result.thumbs[pos])),
                        decode_index(frames[len(frames) // 2]),
                        f"{pos} thumb is not its window's middle frame")
                mid = decode_index(jpeg(result.thumbs["mid"]))
                self.assertLessEqual(abs(mid - 18), 1, "mid thumb")

    def test_one_flat_window_does_not_flag_vivid_clip(self):
        # summary flat follows is_log or the clip level sample, not
        # either window alone
        tex = synth.texture(seed=11)
        path = self.dir / "vivid_then_black.mp4"
        synth.write_clip(path, [tex] * 72 + [np.zeros_like(tex)] * 36)
        result, seen = windows_seen(path)
        self.assertFalse(result.summary["flat"])
        plain = jpeg(analysis.make_thumb(seen[0][len(seen[0]) // 2], flat=False))
        self.assertAlmostEqual(sat(jpeg(result.thumbs["start"])), sat(plain),
                               delta=5.0)
        self.assertAlmostEqual(sat(jpeg(result.thumbs["mid"])), sat(plain),
                               delta=5.0)

    def test_each_window_thumb_follows_its_own_flat_flag(self):
        # a gray open into a vivid close, and the reverse: only the flat
        # window's thumb gets the display stretch
        gray = synth.gray_ramp_frames(n=48)
        vivid = [synth.texture(seed=11)] * 48
        for name, frames, flat_pos in (("gray_then_vivid", gray + vivid, "start"),
                                       ("vivid_then_gray", vivid + gray, "end")):
            with self.subTest(name):
                path = self.dir / f"{name}.mp4"
                synth.write_clip(path, frames)
                result, seen = windows_seen(path)
                # the clip is not flat, but the window that was stretched
                # says so in its own summary
                self.assertFalse(result.summary["flat"])
                for pos, window in (("start", seen[0]), ("end", seen[-1])):
                    self.assertEqual(result.summary[pos]["flat"],
                                     pos == flat_pos, f"{pos} flat flag")
                    raw = sat(jpeg(analysis.make_thumb(
                        window[len(window) // 2], flat=False)))
                    got = sat(jpeg(result.thumbs[pos]))
                    if pos == flat_pos:
                        self.assertGreater(got, raw + 10, f"{pos} thumb")
                    else:
                        self.assertAlmostEqual(got, raw, delta=5.0,
                                               msg=f"{pos} thumb")

    def test_flat_footage_outside_log_folders_is_still_flat(self):
        # a whole clip of low contrast gray still reads as log capture
        path = self.dir / "gray.mp4"
        synth.write_clip(path, synth.gray_ramp_frames(n=48))
        s = analysis.analyze_clip(path, is_log=False).summary
        self.assertTrue(s["flat"])


# -- whip cuts end to end -----------------------------------------------------------

class TestWhipIntoTheCut(ClipDir):
    """a clip that holds and then whips inside its last 1.2 s must stay a
    steady move, or the whip gate, which counts only steady translation,
    throws the cut away."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        def analyze(name, frames):
            path = cls.dir / f"{name}.mp4"
            synth.write_clip(path, frames)
            return analysis.analyze_clip(path, is_log=False)

        # 2.6 s held, then a 0.37 s whip at 1.5 w/s, so 8 of the end
        # window's 28 frame pairs move
        cls.late = analyze("late_whip",
                           hold_then_whip(8, n=72, px=10, block=10))
        cls.whip_in = analyze("whip_in", synth.frames_for("whip_right", 36, px=14)
                              + synth.frames_for("static", 36))
        cls.shaky = analyze("shaky", synth.frames_for("static", 36, seed=10)
                            + synth.frames_for("handheld", 36, seed=10, px=2))

    def gate(self, a, b) -> float:
        lib = lib_of([("A", a.vector), ("B", b.vector)])
        _, c = matching.score_against(lib, 0, "whip")
        return float(c["gates"]["whip"][1])

    def test_static_then_late_whip_passes_the_whip_gate(self):
        end = self.late.summary["end"]
        t = float(np.hypot(*end["flow"]))
        self.assertEqual(self.late.summary["start_class"], "static")
        self.assertIn(end["class"], DIRECTIONAL, end)
        self.assertGreaterEqual(t, config.STEADY_RATIO * end["jitter"], end)
        self.assertTrue(self.whip_in.summary["start_class"].startswith("whip"),
                        self.whip_in.summary["start"])
        self.assertGreater(self.gate(self.late, self.whip_in), 0.5)

    def test_handheld_end_still_fails_the_whip_gate(self):
        self.assertEqual(self.shaky.summary["end_class"], "handheld",
                         self.shaky.summary["end"])
        self.assertLess(self.gate(self.shaky, self.whip_in), 0.05)


class TestTheLastSecond(ClipDir):
    """the end state is the last second before the cut, even when the
    file is cut short. test_analysis_windows pins it at high fps."""

    def test_truncated_moov_first_reads_a_full_real_tail(self):
        # a moov first clip with its mdat cut short still claims every
        # frame. the end window is the last full window that decodes, read
        # once: never the head, never a scrap of a few frames. at 75 percent
        # the cut lands inside the end window the clip claims
        for keep in (0.5, 0.6, 0.75):
            with self.subTest(keep=keep):
                self.check_cut_tail(keep)

    def check_cut_tail(self, keep: float) -> None:
        """analyze the moov first clip with its mdat cut to keep, a 1.25 s
        hold then a pan, and check the end window."""
        if not hasattr(type(self), "blob"):
            src = self.dir / "whole.mp4"
            synth.write_clip(src, synth.frames_for("static", n=30)
                             + synth.frames_for("pan_right", n=90))
            type(self).blob = faststart(src.read_bytes())
        blob, real = self.blob, reader.read_window
        mdat = find_box(blob, b"mdat")
        path = self.dir / f"cut_{int(keep * 100)}.mp4"
        path.write_bytes(
            blob[:mdat.body + int((mdat.end - mdat.body) * keep)])
        reads = []

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            reads.append(out[1])
            return out

        # the decoder complains about the cut on stderr
        with synth._quiet_stderr(True), \
                mock.patch.object(reader, "read_window", spy):
            s = analysis.analyze_clip(path, is_log=False).summary
        self.assertEqual(len(reads), 2, "windows read")
        start, end = reads
        self.assertGreater(end[0], start[-1], "end window is the head")
        self.assertGreaterEqual(end[-1] - end[0],
                                0.9 * config.WINDOW_SECONDS, end)
        self.assertEqual(s["end_class"], "pan_right", s["end"])


class TestRelabelNullSummary(TempDirsMixin, unittest.TestCase):
    """test_cli pins the fill itself. this pins that labels rebuilt from
    the vector are stable: a second relabel writes the same summary."""

    def test_relabel_a_second_time_changes_nothing(self):
        from clipengine import catalog, cli
        conn = catalog.connect()
        cur = conn.execute(
            "INSERT INTO clips (path, rel_path, name, profile, is_log,"
            " country, ext, size_bytes, mtime_ns, content_key, available,"
            " oversize, missing, first_seen, last_seen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            ("/x/a.mp4", "a.mp4", "a.mp4", "t", 0, "X", ".mp4",
             1, 1, "1-1", 1, 0, "t", "t"))
        clip_id = cur.lastrowid
        vec = make_vec(end_flow_x=-0.5, end_energy=0.5, end_jitter=0.05)
        catalog.save_features(conn, clip_id, "1-1", features.to_bytes(vec), None)
        conn.close()

        def summary():
            c = catalog.connect()
            row = c.execute("SELECT summary FROM features WHERE clip_id=?",
                            (clip_id,)).fetchone()
            c.close()
            return row["summary"]

        with mock.patch("sys.stdout"):
            cli.cmd_relabel([])
        first = summary()
        got = json.loads(first)
        self.assertEqual(got["start_class"], "static")
        self.assertEqual(got["end_class"], "pan_right")
        with mock.patch("sys.stdout"):
            cli.cmd_relabel([])
        self.assertEqual(summary(), first)


if __name__ == "__main__":
    unittest.main()
