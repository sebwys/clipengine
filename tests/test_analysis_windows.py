# test_analysis_windows.py
# how frames are sampled and timed. the end window has to reach the cut
# at any frame rate, speeds have to use the real time between compared
# frames on variable frame rate files, a locked off shot has to stay
# static at high fps, and log luma has to keep exposure order.

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from clipengine import analysis, config, reader
from tests import synth
from tests.media_fixtures import decode_index, rewrite_stts, slog3, write_coded


def end_window_indices(path) -> list[int]:
    """frame indices analyze_clip decodes for its end window."""
    seen = []
    real = reader.read_window

    def spy(*args, **kwargs):
        out = real(*args, **kwargs)
        seen.append(out[0])
        return out

    with mock.patch.object(reader, "read_window", spy):
        analysis.analyze_clip(path, is_log=False)
    return [decode_index(f) for f in seen[-1]]


def hold_then_pan(fps: float, seconds: float, tail_s: float,
                  speed: float) -> list[np.ndarray]:
    """one texture held still, then panned right at speed widths per
    second for the last tail_s seconds. one scene, so no cut inside."""
    tex = synth.texture(7, block=8)
    n, tail = int(round(seconds * fps)), int(round(tail_s * fps))
    px = int(round(speed * synth.W / fps))
    shifts = [0] * (n - tail) + [px * (k + 1) for k in range(tail)]
    return [np.roll(tex, -s, axis=1) for s in shifts]


# -- tests ----------------------------------------------------------------------

class TestEndWindowAtHighFps(unittest.TestCase):
    """the end window covers the last WINDOW_SECONDS before the cut at any
    frame rate. the frame cap may thin it out but never shorten it."""

    def test_end_window_reaches_the_last_frames(self):
        for fps, seconds in ((60.0, 5.0), (120.0, 5.0), (120.0, 1.0)):
            with self.subTest(fps=fps, seconds=seconds), \
                    tempfile.TemporaryDirectory() as tmp:
                n = int(round(seconds * fps))
                path = Path(tmp) / "count.mp4"
                write_coded(path, n, fps)
                idx = end_window_indices(path)
                self.assertLessEqual(len(idx), config.MAX_WINDOW_FRAMES)
                self.assertGreaterEqual(idx[-1], n - 1 - round(0.1 * fps),
                                        f"end window {idx[0]}..{idx[-1]} of {n}")
                span = (idx[-1] - idx[0]) / fps
                want = min(seconds, config.WINDOW_SECONDS) - 0.1
                self.assertGreaterEqual(span, want,
                                        f"end window {idx[0]}..{idx[-1]} of {n}")

    def test_end_window_spans_full_length_when_fps_is_guessed(self):
        # a decoder that reports no rate falls back to 30, but the frames
        # still keep their real times, so the window spans its full length
        real = cv2.VideoCapture

        class NoRate:
            def __init__(self, *args):
                self._cap = real(*args)

            def __getattr__(self, name):
                return getattr(self._cap, name)

            def get(self, prop):
                return 0.0 if prop == cv2.CAP_PROP_FPS else self._cap.get(prop)

        for fps in (60.0, 120.0):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as tmp:
                n = int(round(4.0 * fps))
                path = Path(tmp) / "count.mp4"
                write_coded(path, n, fps)
                with mock.patch.object(reader.cv2, "VideoCapture", NoRate), \
                        mock.patch.object(reader, "_warned", set()), \
                        self.assertLogs("clipengine.reader", "WARNING"):
                    idx = end_window_indices(path)
                span = (idx[-1] - idx[0]) / fps
                self.assertGreaterEqual(span, 1.1,
                                        f"end window {idx[0]}..{idx[-1]} of {n}")
                self.assertGreaterEqual(idx[-1], n - 1 - round(0.1 * fps),
                                        f"end window {idx[0]}..{idx[-1]} of {n}")

    def test_end_window_at_24_and_30_fps_is_unchanged(self):
        # the frames these rates always read, 1 to 2 frames short of the end
        for fps, first, last in ((24.0, 90, 118), (30.0, 112, 147)):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as tmp:
                n = int(round(5.0 * fps))
                path = Path(tmp) / "count.mp4"
                write_coded(path, n, fps)
                idx = end_window_indices(path)
                self.assertEqual(idx, list(range(first, last + 1)))

    def test_clip_shorter_than_one_step_still_reads(self):
        # fewer frames than one 1/30 s stride, so no two frames a step apart
        for fps, n in ((60.0, 2), (120.0, 4), (240.0, 8)):
            with self.subTest(fps=fps), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "blink.mp4"
                write_coded(path, n, fps)
                frames, times, _ = reader.read_window(path, 0.0)
                self.assertEqual([decode_index(f) for f in frames], list(range(n)))
                still = Path(tmp) / "still.mp4"
                synth.write_clip(still, [synth.texture(7, block=8)] * n, fps)
                s = analysis.analyze_clip(still, is_log=False).summary
                self.assertEqual(s["start_class"], "static", s["start"])

    def test_pan_into_the_cut_reads_alike_at_24_60_and_120_fps(self):
        # hold, then pan right at 0.75 widths per second for the last 0.6 s
        energy = {}
        with tempfile.TemporaryDirectory() as tmp:
            for fps in (24.0, 60.0, 120.0):
                path = Path(tmp) / f"settle_{fps:g}.mp4"
                synth.write_clip(path, hold_then_pan(fps, 3.0, 0.6, 0.75), fps)
                s = analysis.analyze_clip(path, is_log=False).summary
                with self.subTest(fps=fps):
                    self.assertEqual(s["start_class"], "static")
                    self.assertEqual(s["end_class"], "pan_right", s["end"])
                energy[fps] = s["end"]["energy"]
        for fps in (60.0, 120.0):
            with self.subTest(fps=fps):
                self.assertAlmostEqual(energy[fps], energy[24.0],
                                       delta=0.25 * energy[24.0], msg=energy)


class TestVariableFrameRate(unittest.TestCase):
    """phones drop their frame rate in low light. seeks have to land on
    the frame shown at the asked time, and speeds have to divide by the
    real time between frames, not by one average fps."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_read_pair_lands_on_the_frame_shown_at_t(self):
        # 1 s at 48 fps, then 2 s at 24 fps: the decoder reports 32 fps
        path = self.tmp / "vfr.mp4"
        write_coded(path, 96, 48.0)
        pts = rewrite_stts(path, [1 / 48] * 48 + [1 / 24] * 48)
        for t in (0.5, 1.5, 2.0, 2.5):
            with self.subTest(t=t):
                got = decode_index(reader.read_pair(path, t)[0])
                self.assertLessEqual(abs(pts[got] - t), 1 / 48 + 1e-6,
                                     f"frame {got} is shown at {pts[got]:.3f} s")

    def test_end_window_covers_the_last_second_by_time(self):
        # 1 s at 60 fps, then 2 s at 30 fps, so the end window starts at
        # 1.75 s, on frame 82 or 83, and runs to the last frames
        path = self.tmp / "vfr_end.mp4"
        write_coded(path, 120, 60.0)
        pts = rewrite_stts(path, [1 / 60] * 60 + [1 / 30] * 60)
        idx = end_window_indices(path)
        start = 3.0 - config.WINDOW_SECONDS - 0.05
        self.assertAlmostEqual(pts[idx[0]], start, delta=1 / 30 + 1e-6,
                               msg=f"end window {idx[0]}..{idx[-1]}")
        self.assertGreaterEqual(idx[-1], 116, f"end window {idx[0]}..{idx[-1]}")

    def test_jittered_30_fps_keeps_every_frame(self):
        # phone clocks: 30 fps frames alternate 19 and 21 ticks of 1/600 s
        path = self.tmp / "jitter.mp4"
        write_coded(path, 90, 30.0)
        rewrite_stts(path, [19 / 600, 21 / 600] * 45)
        idx = [decode_index(f) for f in reader.read_window(path, 0.0)[0]]
        self.assertEqual(idx, list(range(36)))
        got = reader.read_pair(path, 1.0)
        self.assertIsNotNone(got)
        self.assertLess(got[2], 1 / 30 + 0.005, "pair skipped a frame")

    def test_jittered_30_fps_keeps_its_motion_labels(self):
        for kind, px in (("whip_left", 14), ("handheld", 3)):
            with self.subTest(kind=kind):
                path = self.tmp / f"jitter_{kind}.mp4"
                synth.write_clip(path, synth.frames_for(kind, 90, px=px), 30.0)
                rewrite_stts(path, [19 / 600, 21 / 600] * 45)
                s = analysis.analyze_clip(path, is_log=False).summary
                self.assertEqual(s["start_class"], kind, s["start"])

    def test_constant_pan_reads_true_speed_at_both_ends(self):
        # 5 s at 24 fps, then 5 s at 12 fps, one camera speed throughout:
        # 3 px per 1/24 s frame, then 6 px per 1/12 s frame
        tex = synth.texture(7, block=8)
        steps = [3] * 120 + [6] * 60
        shifts = np.concatenate([[0], np.cumsum(steps)[:-1]])
        path = self.tmp / "vfr_pan.mp4"
        synth.write_clip(path, [np.roll(tex, -int(k), axis=1) for k in shifts])
        rewrite_stts(path, [1 / 24] * 120 + [1 / 12] * 60)
        s = analysis.analyze_clip(path, is_log=False).summary
        truth = -3 * 24.0 / synth.W  # -0.45, content flows left
        self.assertAlmostEqual(s["duration_s"], 10.0, delta=0.1)
        self.assertAlmostEqual(s["start"]["flow"][0], truth, delta=0.05,
                               msg=f"start {s['start']}")
        self.assertAlmostEqual(s["end"]["flow"][0], truth, delta=0.05,
                               msg=f"end {s['end']}")
        self.assertEqual(s["start_class"], "pan_right")
        self.assertEqual(s["end_class"], "pan_right")
        prof = s["energy_profile"]
        self.assertLess(max(prof) / min(prof), 1.2, f"profile {prof}")


class TestStaticNoiseAcrossFps(unittest.TestCase):
    """sensor noise moves pixels by the same amount per frame at any fps.
    a locked off shot must not turn into drift because it was shot at
    120 or 240 fps."""

    def _noisy_still(self, path, fps: float) -> None:
        # grain the size of a pixel or two, like codec noise on real footage
        rng = np.random.default_rng(11)
        base = (synth.texture(7, 8) // 2 + 60).astype(np.float32)
        frames = []
        for _ in range(int(2 * fps)):
            grain = cv2.GaussianBlur(
                rng.normal(0, 6.0, base.shape).astype(np.float32), (0, 0), 1.0)
            frames.append(np.clip(base + 2.0 * grain, 0, 255).astype(np.uint8))
        synth.write_clip(path, frames, fps)

    def test_noisy_locked_off_shot_is_static_at_any_fps(self):
        got = {}
        with tempfile.TemporaryDirectory() as tmp:
            for fps in (24.0, 120.0, 240.0):
                path = Path(tmp) / f"still_{fps:g}.mp4"
                self._noisy_still(path, fps)
                got[fps] = analysis.analyze_clip(path, is_log=False).summary
        for fps, s in got.items():
            with self.subTest(fps=fps):
                self.assertEqual(s["start_class"], "static", s["start"])
                self.assertEqual(s["end_class"], "static", s["end"])
        base = got[24.0]
        for fps in (120.0, 240.0):
            with self.subTest(fps=fps):
                s = got[fps]
                self.assertLess(s["start"]["energy"], 2 * base["start"]["energy"])
                self.assertLess(s["tempo_mean"], 2 * base["tempo_mean"])


class TestLogLumaKeepsExposure(unittest.TestCase):
    """log encoding turns an exposure change into an offset, and the
    normalization stretch removes offsets. luma has to survive it, or a
    night clip reads as bright as a beach."""

    def test_log_luma_orders_bright_over_dark(self):
        # the colors of test_luma_orders_bright_over_dark, on the log path
        bright = analysis.color_stats(
            synth.color_frames((80, 160, 250)), force_log=True)
        dark = analysis.color_stats(
            synth.color_frames((10, 20, 60)), force_log=True)
        self.assertGreater(bright["luma"], dark["luma"] + 60)

    def test_slog3_four_stop_drop_lowers_luma(self):
        rng = np.random.default_rng(3)
        small = np.exp(rng.uniform(np.log(0.03), np.log(0.9),
                                   (synth.H // 8, synth.W // 8)))
        refl = cv2.resize(small.astype(np.float32), (synth.W, synth.H),
                          interpolation=cv2.INTER_NEAREST)

        def stats(stops):
            lin = refl[..., None] * np.array([0.95, 1.0, 1.08]) * 2.0 ** stops
            code = (slog3(lin) * 255).round().clip(0, 255).astype(np.uint8)
            return analysis.color_stats([code] * 12, force_log=True)

        lumas = [stats(s)["luma"] for s in (0, -1, -2, -3, -4)]
        self.assertEqual(lumas, sorted(lumas, reverse=True))
        self.assertGreater(lumas[0], lumas[-1] + 40, lumas)

    def test_log_footage_still_gets_stretched(self):
        # the stretch stays for contrast and saturation, only luma is raw
        stats = analysis.color_stats(synth.gray_ramp_frames(), force_log=True)
        self.assertTrue(stats["flat"])
        self.assertGreater(stats["contrast"], 100)


if __name__ == "__main__":
    unittest.main()
