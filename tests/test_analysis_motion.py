# test_analysis_motion.py
# motion analysis against ground truth. synthetic frames roll a texture
# at an exact pixel speed, so expected flow in widths/sec is arithmetic:
# px_per_frame / width * fps. a 3 px pan at 160 px and 24 fps is 0.45,
# chosen mid zone: fixture speeds must not sit on class boundaries.

import tempfile
import unittest
from pathlib import Path

from clipengine import analysis, config
from tests import synth

FPS = synth.FPS
N = 30  # ~1.25 s of frames, one analysis window


def motion_for(kind: str, px: int = 4):
    return analysis.window_motion(synth.frames_for(kind, n=N, px=px), FPS)


class TestClassification(unittest.TestCase):
    def test_static_is_static(self):
        m = motion_for("static")
        self.assertEqual(m.label, "static")
        self.assertLess(m.energy, config.STATIC_MAX)

    def test_pan_right_direction_and_speed(self):
        m = motion_for("pan_right", px=3)
        expected = -3 * FPS / synth.W  # content flows left: negative x
        self.assertEqual(m.label, "pan_right")
        self.assertLess(m.flow_x, -config.PAN_MIN)
        self.assertAlmostEqual(m.flow_x, expected, delta=abs(expected) * 0.4)
        self.assertLess(abs(m.flow_y), abs(m.flow_x) / 3)

    def test_pan_left_flips_sign(self):
        m = motion_for("pan_left", px=3)
        self.assertEqual(m.label, "pan_left")
        self.assertGreater(m.flow_x, config.PAN_MIN)

    def test_tilt_up_is_positive_y(self):
        # camera tilts up, content flows down, y grows downward in images
        m = motion_for("tilt_up", px=3)
        self.assertEqual(m.label, "tilt_up")
        self.assertGreater(m.flow_y, config.PAN_MIN)

    def test_tilt_down(self):
        m = motion_for("tilt_down", px=3)
        self.assertEqual(m.label, "tilt_down")
        self.assertLess(m.flow_y, -config.PAN_MIN)

    def test_whip_speed_crosses_threshold(self):
        m = motion_for("whip_right", px=14)
        self.assertTrue(m.label.startswith("whip"),
                        f"expected whip, got {m.label} at energy {m.energy}")
        self.assertGreater(abs(m.flow_x), config.WHIP_MIN * 0.8)

    def test_push_in_reads_radial(self):
        m = analysis.window_motion(synth.frames_for("push_in", n=N), FPS)
        self.assertEqual(m.label, "push_in")
        self.assertGreater(m.radial, config.PUSH_MIN)

    def test_pull_out_reads_negative_radial(self):
        m = analysis.window_motion(synth.frames_for("pull_out", n=N), FPS)
        self.assertEqual(m.label, "pull_out")
        self.assertLess(m.radial, -config.PUSH_MIN)

    def test_handheld_high_jitter_no_direction(self):
        m = motion_for("handheld", px=4)
        self.assertEqual(m.label, "handheld")
        self.assertGreater(m.jitter, abs(m.flow_x))


class TestSteadiness(unittest.TestCase):
    """directional labels must be steadier than they are shaky. the
    fixture numbers come from a real fx30 walking shot that measured
    translation 0.08 with jitter 0.27: drift, not a cuttable pan."""

    def _classify(self, **kw):
        m = analysis.WindowMotion(
            flow_x=kw.get("flow_x", 0.0), flow_y=kw.get("flow_y", 0.0),
            energy=kw.get("energy", 0.2), radial=kw.get("radial", 0.0),
            jitter=kw.get("jitter", 0.0))
        return analysis.classify_motion(m)

    def test_clean_pan_still_pans(self):
        self.assertEqual(self._classify(flow_x=-0.2, jitter=0.02),
                         "pan_right")

    def test_shaky_drift_is_handheld_not_pan(self):
        label = self._classify(flow_x=-0.08, jitter=0.27, energy=0.2)
        self.assertEqual(label, "handheld")

    def test_steady_whip_still_whips(self):
        self.assertEqual(
            self._classify(flow_x=-1.2, energy=1.3, jitter=0.4),
            "whip_right")

    def test_chaotic_speed_is_handheld_not_whip(self):
        label = self._classify(flow_x=-1.2, energy=2.5, jitter=2.5)
        self.assertEqual(label, "handheld")


class TestMotionFromVector(unittest.TestCase):
    def test_roundtrip_from_packed_vector(self):
        from tests.test_matching import make_vec
        vec = make_vec(end_flow_x=-0.5, end_energy=0.5, end_jitter=0.05)
        end = analysis.motion_from_vector(vec, "end")
        self.assertEqual(end.label, "pan_right")
        self.assertAlmostEqual(end.flow_x, -0.5, places=4)
        start = analysis.motion_from_vector(vec, "start")
        self.assertEqual(start.label, "static")


class TestFullClipWindows(unittest.TestCase):
    """the only file backed motion test: analyze_clip must see different
    states at the two ends of a two phase clip."""

    def test_static_start_pan_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "two_phase.mp4"
            synth.make_clip(path, ["static", "pan_right"], n_each=36)
            result = analysis.analyze_clip(path, is_log=False)
            s = result.summary
            self.assertEqual(s["start_class"], "static")
            self.assertEqual(s["end_class"], "pan_right")
            self.assertAlmostEqual(s["duration_s"], 3.0, delta=0.3)
            self.assertEqual(set(result.thumbs), {"start", "mid", "end"})
            for blob in result.thumbs.values():
                self.assertGreater(len(blob), 500)  # real jpegs, not stubs


if __name__ == "__main__":
    unittest.main()
