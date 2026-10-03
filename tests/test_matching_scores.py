# test_matching_scores.py
# scoring components on the cases that can fool them: shake passing
# the whip gate, push and pull read as sideways drift, and sensor noise
# splitting two locked off shots. hand built vectors pin the math,
# analyzed synthetic clips pin it end to end.

import unittest

import cv2
import numpy as np

from clipengine import analysis, config, matching
from tests import synth
from tests.test_matching import lib_of, make_vec
from tests.util import TempDirsMixin


def shake(side: str, drift: float = 0.0, energy: float = 0.6) -> dict:
    """a handheld window: all jitter, almost no net translation."""
    return {f"{side}_flow_x": drift, f"{side}_energy": energy,
            f"{side}_jitter": energy}


def walk(side: str, drift: float, radial: float, energy: float = 0.12,
         jitter: float = 0.0) -> dict:
    """a push in (radial > 0) or pull out (radial < 0) drifting sideways."""
    return {f"{side}_flow_x": drift, f"{side}_energy": energy,
            f"{side}_radial": radial, f"{side}_jitter": jitter}


def zoom_frames(rate: float, drift: float, n: int = 72,
                seed: int = 7) -> list[np.ndarray]:
    """zoom by rate per frame while sliding drift px per frame."""
    tex = synth.texture(seed)
    frames, scale, shift = [], 1.0, 0.0
    for _ in range(n):
        scale *= rate
        shift += drift
        m = cv2.getRotationMatrix2D((synth.W / 2, synth.H / 2), 0, scale)
        m[0, 2] += shift
        frames.append(cv2.warpAffine(tex, m, (synth.W, synth.H),
                                     borderMode=cv2.BORDER_REFLECT))
    return frames


def still_frames(noise: int, n: int = 72, seed: int = 7) -> list[np.ndarray]:
    """a locked off shot with uniform sensor noise of the given level."""
    tex = synth.texture(seed) // 2 + 60
    rng = np.random.default_rng(seed + 1)
    return [cv2.add(tex, rng.integers(0, noise + 1, tex.shape, np.uint8))
            for _ in range(n)]


class TestWhipGateNeedsSteadyTranslation(unittest.TestCase):
    def test_pure_shake_fails_the_whip_gate(self):
        end, start = make_vec(**shake("end")), make_vec(**shake("start"))
        self.assertEqual(analysis.motion_from_vector(end, "end").label,
                         "handheld")
        self.assertEqual(analysis.motion_from_vector(start, "start").label,
                         "handheld")
        _, c = matching.score_against(lib_of([("A", end), ("B", start)]),
                                      0, "whip")
        self.assertLess(c["gates"]["whip"][1], 0.1)

    def test_steady_whip_keeps_the_full_gate(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.8, end_energy=0.8, end_jitter=0.05)),
            ("B", make_vec(start_flow_x=-0.8, start_energy=0.8,
                           start_jitter=0.05)),
        ])
        _, c = matching.score_against(lib, 0, "whip")
        self.assertAlmostEqual(float(c["gates"]["whip"][1]), 1.0, places=6)

    def test_shaky_start_scores_nothing_after_a_real_whip(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.8, end_energy=0.8, end_jitter=0.05)),
            ("B", make_vec(start_flow_x=-0.8, start_energy=0.8,
                           start_jitter=0.05)),
            ("C", make_vec(**shake("start", drift=-0.02, energy=0.8))),
        ])
        scores, _ = matching.score_against(lib, 0, "whip")
        self.assertEqual(float(scores[2]), 0.0)
        self.assertEqual([r["id"] for r in matching.rank(lib, 1, "whip")],
                         [2])


class TestNoiseDirectionIsNeutral(unittest.TestCase):
    def setUp(self):
        # three handheld windows whose net flow is noise of either sign
        self.lib = lib_of([
            ("A", make_vec(**shake("end", drift=0.01))),
            ("B", make_vec(**shake("start", drift=0.01))),
            ("C", make_vec(**shake("start", drift=-0.01))),
        ])

    def test_shake_direction_leaves_motion_neutral(self):
        c = matching.components(self.lib, 0)
        self.assertAlmostEqual(float(c["motion"][1]), 0.5, places=6)
        self.assertAlmostEqual(float(c["motion"][2]), 0.5, places=6)

    def test_noise_sign_does_not_decide_momentum_rank(self):
        scores, _ = matching.score_against(self.lib, 0, "momentum")
        self.assertAlmostEqual(float(scores[1]), float(scores[2]), places=6)

    def test_radial_noise_sign_leaves_motion_neutral(self):
        # shake masks the translation, so only radial noise is left
        lib = lib_of([
            ("A", make_vec(**walk("end", 0.05, 0.01, 0.3, jitter=0.3))),
            ("B", make_vec(**walk("start", 0.05, 0.01, 0.3, jitter=0.3))),
            ("C", make_vec(**walk("start", 0.05, -0.01, 0.3, jitter=0.3))),
        ])
        c = matching.components(lib, 0)
        self.assertAlmostEqual(float(c["motion"][1]), 0.5, places=6)
        self.assertAlmostEqual(float(c["motion"][2]), 0.5, places=6)
        scores, _ = matching.score_against(lib, 0, "momentum")
        self.assertAlmostEqual(float(scores[1]), float(scores[2]), places=6)

    def test_shake_into_a_push_ignores_radial_below_push_min(self):
        # radial just past STATIC_MAX but under PUSH_MIN is not a push
        for radial in (-0.021, 0.021):
            with self.subTest(radial=radial):
                lib = lib_of([
                    ("A", make_vec(**walk("end", 0.1, radial, 0.6,
                                          jitter=0.3))),
                    ("B", make_vec(**walk("start", 0.0, 0.2, 0.2))),
                ])
                c = matching.components(lib, 0)
                self.assertAlmostEqual(float(c["motion"][1]), 0.5, places=6)

    def test_tiny_steady_drift_leaves_motion_neutral(self):
        # steady but nearly still: the drift sign is noise
        lib = lib_of([
            ("A", make_vec(**walk("end", 0.01, 0.0, 0.3))),
            ("B", make_vec(**walk("start", 0.01, 0.0, 0.3))),
            ("C", make_vec(**walk("start", -0.01, 0.0, 0.3))),
        ])
        c = matching.components(lib, 0)
        self.assertAlmostEqual(float(c["motion"][1]), 0.5, places=6)
        self.assertAlmostEqual(float(c["motion"][2]), 0.5, places=6)


class TestMomentumReadsRadial(unittest.TestCase):
    def setUp(self):
        self.lib = lib_of([
            ("A", make_vec(**walk("end", 0.01, 0.12))),     # 1: ends pushing in
            ("A", make_vec(**walk("start", 0.01, 0.12))),   # 2: push in, same drift
            ("A", make_vec(**walk("start", 0.01, -0.12))),  # 3: pull out, same drift
            ("A", make_vec(**walk("start", -0.01, 0.12))),  # 4: push in, other drift
        ])
        self.s, _ = matching.score_against(self.lib, 0, "momentum")

    def test_push_in_to_push_in_beats_push_in_to_pull_out(self):
        self.assertGreater(self.s[1], self.s[2])

    def test_tiny_lateral_drift_does_not_decide_push_in_cuts(self):
        self.assertAlmostEqual(float(self.s[1]), float(self.s[3]), delta=0.05)

    def test_reversal_is_not_a_good_cut(self):
        self.assertGreater(self.s[3], self.s[2])
        results = matching.rank(self.lib, 1, "momentum", n=5)
        self.assertNotEqual(results[0]["id"], 3)
        self.assertLess(matching.components(self.lib, 0)["motion"][2], 0.1)

    def test_shaky_push_in_keeps_its_direction(self):
        # sideways shake is noise, but the push itself is still forward
        lib = lib_of([
            ("A", make_vec(**walk("end", 0.01, 0.12, jitter=0.3))),
            ("B", make_vec(**walk("start", -0.01, 0.12, jitter=0.3))),
            ("C", make_vec(**walk("start", 0.01, -0.12, jitter=0.3))),
        ])
        c = matching.components(lib, 0)
        self.assertGreater(c["motion"][1], 0.9)
        self.assertLess(c["motion"][2], 0.1)

    def test_slow_aligned_moves_read_fully_aligned(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.03, end_energy=0.03)),
            ("B", make_vec(start_flow_x=-0.03, start_energy=0.03)),
        ])
        self.assertGreater(matching.components(lib, 0)["motion"][1], 0.99)


class TestStillEnergyIgnoresNoiseFloor(unittest.TestCase):
    def test_two_static_windows_have_alike_energy(self):
        ae, be = 0.0003, 0.003
        self.assertLess(max(ae, be), config.STATIC_MAX)
        lib = lib_of([("A", make_vec(end_energy=ae)),
                      ("B", make_vec(start_energy=be))])
        self.assertGreater(matching.components(lib, 0)["energy"][1], 0.95)

    def test_calm_prefers_color_match_over_matching_grain(self):
        lib = lib_of([
            ("A", make_vec(end_energy=0.0001)),
            ("B", make_vec(start_energy=0.001)),                    # same color
            ("C", make_vec(start_energy=0.0001, start_luma=100.0)),  # same grain
        ])
        scores, _ = matching.score_against(lib, 0, "calm")
        self.assertGreater(scores[1], scores[2])

    def test_still_against_moving_reads_unlike(self):
        lib = lib_of([("A", make_vec(end_energy=0.004)),
                      ("B", make_vec(start_flow_x=-0.5, start_energy=0.5))])
        self.assertLess(matching.components(lib, 0)["energy"][1], 0.1)

    def test_moving_speed_ratio_is_unchanged(self):
        lib = lib_of([("A", make_vec(end_flow_x=-0.3, end_energy=0.3)),
                      ("B", make_vec(start_flow_x=-0.6, start_energy=0.6))])
        self.assertAlmostEqual(
            float(matching.components(lib, 0)["energy"][1]), 0.5, delta=0.01)


class TestAnalyzedClips(TempDirsMixin, unittest.TestCase):
    def analyze(self, name: str, frames=None, kinds=None, **kw) -> np.ndarray:
        path = self.tmp / f"{name}.mp4"
        if frames is not None:
            synth.write_clip(path, frames)
        else:
            synth.make_clip(path, kinds, **kw)
        return analysis.analyze_clip(path, False)

    def test_handheld_clips_get_no_whip_cut(self):
        a = self.analyze("a", kinds=["static", "handheld"], seed=10, px=2)
        b = self.analyze("b", kinds=["handheld", "static"], seed=11, px=2)
        self.assertEqual(a.summary["end_class"], "handheld")
        self.assertEqual(b.summary["start_class"], "handheld")
        lib = lib_of([("A", a.vector), ("B", b.vector)])
        self.assertEqual(matching.rank(lib, 1, "whip"), [])

    def test_real_whip_beats_handheld_in_whip_mode(self):
        a = self.analyze("a", kinds=["static", "whip_left"], px=14)
        b = self.analyze("b", kinds=["whip_left", "static"], px=14)
        c = self.analyze("c", kinds=["handheld", "static"], seed=11, px=2)
        self.assertTrue(a.summary["end_class"].startswith("whip"))
        self.assertTrue(b.summary["start_class"].startswith("whip"))
        lib = lib_of([("A", a.vector), ("B", b.vector), ("C", c.vector)])
        results = matching.rank(lib, 1, "whip")
        self.assertEqual([r["id"] for r in results], [2])
        self.assertGreater(results[0]["breakdown"]["gate"], 0.99)

    def test_push_in_follows_push_in_not_pull_out(self):
        a = self.analyze("a", frames=zoom_frames(1.015, 0.2))
        b = self.analyze("b", frames=zoom_frames(1.015, -0.2))
        c = self.analyze("c", frames=zoom_frames(1 / 1.015, 0.2))
        self.assertEqual(a.summary["end_class"], "push_in")
        self.assertEqual(b.summary["start_class"], "push_in")
        self.assertEqual(c.summary["start_class"], "pull_out")
        lib = lib_of([("A", a.vector), ("B", b.vector), ("C", c.vector)])
        results = matching.rank(lib, 1, "momentum", n=5)
        self.assertEqual(results[0]["id"], 2)
        by_id = {r["id"]: r for r in results}
        self.assertGreater(by_id[2]["breakdown"]["motion"], 0.9)
        self.assertLess(by_id[3]["breakdown"]["motion"], 0.1)

    def test_stills_with_different_grain_have_alike_energy(self):
        a = self.analyze("a", frames=still_frames(3))
        b = self.analyze("b", frames=still_frames(12))
        self.assertEqual(a.summary["end_class"], "static")
        self.assertEqual(b.summary["start_class"], "static")
        lib = lib_of([("A", a.vector), ("B", b.vector)])
        self.assertGreaterEqual(matching.components(lib, 0)["energy"][1], 0.95)


class TestScoresStayInRange(unittest.TestCase):
    def test_every_mode_scores_in_unit_range_and_repeats(self):
        vecs = [make_vec(), make_vec(end_energy=0.0001, start_energy=0.019),
                make_vec(**shake("end"), **shake("start", drift=-0.02)),
                make_vec(**walk("end", 0.01, 0.12), **walk("start", 0.0, -0.1)),
                make_vec(end_flow_x=-1.8, end_energy=1.8,
                         start_flow_x=0.03, start_flow_y=-0.03,
                         start_energy=0.05, start_jitter=0.01),
                make_vec(end_flow_x=0.5, end_energy=0.5, end_radial=-0.4,
                         start_luma=10.0, start_hue_bin=3, end_hue_bin=9)]
        lib = lib_of([("A", v) for v in vecs])
        for mode in config.SCORING_MODES:
            M = matching.full_matrix(lib, mode)
            off = M[~np.eye(len(vecs), dtype=bool)]
            self.assertTrue(np.all(off >= 0.0) and np.all(off <= 1.0), mode)
            np.testing.assert_array_equal(M, matching.full_matrix(lib, mode))


if __name__ == "__main__":
    unittest.main()
