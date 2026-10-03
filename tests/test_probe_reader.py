# test_probe_reader.py
# probe and reader edge cases: mvhd fields out of range, the all ones
# unknown duration, the one fps rule both modules share, and windows
# near a tail that does not decode.

import shutil
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from clipengine import analysis, config, probe, reader
from tests import media_fixtures as mf
from tests import synth
from tests.util import TempDirsMixin

_N = 120
_SECONDS = _N / synth.FPS


class _SharedClip(TempDirsMixin, unittest.TestCase):
    """one 5 s 24 fps source clip per class, copied fresh for each edit."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._shared = tempfile.TemporaryDirectory(prefix="clipengine-pr-")
        cls.src = Path(cls._shared.name) / "src.mp4"
        synth.write_clip(cls.src, synth.frames_for("pan_right", n=_N))

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()
        super().tearDownClass()

    def copy(self, name: str) -> Path:
        path = self.tmp / name
        shutil.copyfile(self.src, path)
        return path


class TestContainerTimes(_SharedClip):
    def test_sane_v1_ctime_keeps_created_and_duration(self):
        path = self.copy("ok.mp4")
        mf.set_mvhd(path, 1, ctime=3866112000)
        meta = probe.parse_container(path)
        self.assertTrue(meta["created"].startswith("2026-07-05"))
        self.assertAlmostEqual(meta["duration_s"], _SECONDS, delta=0.05)

    def test_v1_ctime_out_of_range_drops_created(self):
        for ctime in (300_000_000_000, 0xFFFFFFFFFFFFFFFF):
            with self.subTest(ctime=ctime):
                path = self.copy(f"{ctime}.mp4")
                mf.set_mvhd(path, 1, ctime=ctime)
                meta = probe.parse_container(path)
                self.assertNotIn("created", meta)
                self.assertAlmostEqual(meta["duration_s"], _SECONDS, delta=0.05)
        path = self.copy("analyze.mp4")
        mf.set_mvhd(path, 1, ctime=300_000_000_000)
        result = analysis.analyze_clip(path, is_log=False)
        self.assertAlmostEqual(result.summary["duration_s"], _SECONDS, delta=0.1)
        self.assertIsNone(result.summary["created"])

    def test_created_stops_before_2100(self):
        epoch = datetime(1904, 1, 1, tzinfo=timezone.utc)
        edge = int((datetime(2100, 1, 1, tzinfo=timezone.utc)
                    - epoch).total_seconds())
        for ctime, kept in ((edge - 1, True), (edge, False),
                            (7_000_000_000, False)):
            with self.subTest(ctime=ctime):
                path = self.copy(f"late_{ctime}.mp4")
                mf.set_mvhd(path, 1, ctime=ctime)
                meta = probe.parse_container(path)
                self.assertEqual("created" in meta, kept, meta)
                self.assertAlmostEqual(meta["duration_s"], _SECONDS, delta=0.05)

    def test_ctime_before_1970_drops_created(self):
        path = self.copy("old.mp4")
        mf.set_mvhd(path, 0, ctime=86400)
        meta = probe.parse_container(path)
        self.assertNotIn("created", meta)
        self.assertAlmostEqual(meta["duration_s"], _SECONDS, delta=0.05)

    def test_zero_duration_falls_back_to_decoder(self):
        path = self.copy("zero.mp4")
        mf.set_mvhd(path, 0, duration=0)
        self.assertAlmostEqual(probe.probe(path)["duration_s"], _SECONDS, delta=0.05)

    def test_all_ones_duration_is_unknown(self):
        path = self.copy("v0.mp4")
        mf.set_mvhd(path, 0, duration=0xFFFFFFFF)
        self.assertAlmostEqual(probe.probe(path)["duration_s"], _SECONDS, delta=0.05)
        path = self.copy("v1.mp4")
        mf.set_mvhd(path, 1, duration=0xFFFFFFFFFFFFFFFF)
        self.assertNotIn("duration_s", probe.parse_container(path))
        self.assertAlmostEqual(probe.probe(path)["duration_s"], _SECONDS, delta=0.05)


def _cap_fps(path) -> float:
    cap = cv2.VideoCapture(str(path))
    try:
        return reader._fps(cap)
    finally:
        cap.release()


class TestFpsRule(TempDirsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._shared = tempfile.TemporaryDirectory(prefix="clipengine-fps-")
        cls.clips = {}
        for fps in (0.5, 1.0, 240.0, 250.0):
            path = Path(cls._shared.name) / f"fps_{fps}.mp4"
            n = 48 if fps == 1.0 else 12
            synth.write_clip(path, synth.frames_for("pan_right", n=n, px=3),
                             fps=fps)
            cls.clips[fps] = path

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()
        super().tearDownClass()

    def test_one_fps_rule_is_shared(self):
        path = self.clips[1.0]
        self.assertEqual(probe.probe(path)["fps"], 1.0)
        self.assertEqual(_cap_fps(path), 1.0)
        summary = analysis.analyze_clip(path, is_log=False).summary
        self.assertEqual(summary["fps"], 1.0)

    def test_out_of_range_fps_falls_back_to_30_in_both(self):
        for fps, want in ((0.5, 30.0), (250.0, 30.0), (240.0, 240.0)):
            with self.subTest(fps=fps):
                path = self.clips[fps]
                self.assertEqual(probe.probe(path)["fps"], want)
                self.assertEqual(_cap_fps(path), want)

    def test_probe_uses_the_reader_rule(self):
        # a tighter shared bound reaches probe too
        with mock.patch.object(reader, "FPS_MAX", 200.0), \
                mock.patch.object(reader, "_warned", set()):
            self.assertEqual(probe.probe(self.clips[240.0])["fps"], 30.0)

    def test_fallback_warns_once_per_clip_and_names_it(self):
        for first in ("probe", "read_window"):
            with self.subTest(first=first):
                path = self.tmp / f"{first}.mp4"
                shutil.copyfile(self.clips[250.0], path)
                with self.assertLogs("clipengine.reader", "WARNING") as logs:
                    if first == "probe":
                        probe.probe(path)
                    for _ in range(3):
                        reader.read_window(path, 0.0)
                        reader.read_pair(path, 0.1)
                    probe.probe(path)
                self.assertEqual(len(logs.records), 1, logs.output)
                self.assertIn(str(path), logs.output[0])

    def test_fallback_logs_a_warning_and_kept_rates_stay_quiet(self):
        with mock.patch.object(reader, "_warned", set()), \
                self.assertLogs("clipengine.reader", level="WARNING") as logs:
            self.assertEqual(reader.sane_fps(float("nan"), "x.mp4"), 30.0)
            self.assertEqual(reader.sane_fps(None), 30.0)
        self.assertEqual(len(logs.records), 2)
        self.assertIn("x.mp4", logs.output[0])
        with self.assertNoLogs("clipengine.reader", level="WARNING"):
            self.assertEqual(reader.sane_fps(23.976), 23.976)


class TestReaderArguments(_SharedClip):
    def test_defaults_follow_config_at_call_time(self):
        with mock.patch.object(config, "MAX_WINDOW_FRAMES", 10), \
                mock.patch.object(config, "ANALYSIS_WIDTH", 80):
            frames, times, _ = reader.read_window(self.src, 0.0)
            pair = reader.read_pair(self.src, 1.0)
        self.assertLessEqual(len(frames), 10)
        self.assertEqual(frames[0].shape[1], 80)
        self.assertGreater(times[-1], 1.0, "the window still spans 1.2 s")
        self.assertEqual(pair[0].shape[1], 80)

    def test_non_finite_start_is_a_clip_read_error(self):
        for t in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(t=t):
                with self.assertRaisesRegex(reader.ClipReadError, "finite"):
                    reader.read_window(self.src, t)
                with self.assertRaisesRegex(reader.ClipReadError, "finite"):
                    reader.read_pair(self.src, t)

    def test_one_frame_clip_says_one_frame_decodes(self):
        path = self.tmp / "one.mp4"
        synth.write_clip(path, synth.frames_for("static", n=1))
        with self.assertRaisesRegex(reader.ClipReadError, "only one frame"):
            reader.read_window(path, 0.0)
        self.assertIsNone(reader.read_pair(path, 0.0))
        self.assertIsNone(reader.read_frame(path, 0.0, 80))


_RealCapture = cv2.VideoCapture


class _OverstatedCount:
    """the real decoder with a frame count past the real end, as opencv
    reports for an mkv whose audio outlasts the video."""

    extra = 29

    def __init__(self, path):
        self._cap = _RealCapture(path)

    def get(self, prop):
        value = self._cap.get(prop)
        return value + self.extra if prop == cv2.CAP_PROP_FRAME_COUNT else value

    def __getattr__(self, name):
        return getattr(self._cap, name)


def _overstated(extra: int):
    """patch the decoder so its frame count runs extra frames long."""
    kind = type("_Over", (_OverstatedCount,), {"extra": extra})
    return mock.patch.object(reader.cv2, "VideoCapture", kind)


def _idx(frames) -> list[int]:
    return [mf.decode_index(f) for f in frames]


def _blue(frame) -> bool:
    return frame[..., 0].mean() > frame[..., 2].mean()


class TestTailFallback(TempDirsMixin, unittest.TestCase):
    """a seek near the end that decodes nothing must step back to the
    real tail or fail, never hand back the head of the clip."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._shared = tempfile.TemporaryDirectory(prefix="clipengine-tail-")
        root = Path(cls._shared.name)
        cls.red_blue = root / "red_blue.mp4"
        synth.write_clip(cls.red_blue,
                         synth.color_frames((40, 40, 200), n=60)
                         + synth.color_frames((200, 60, 40), n=60))
        cls.coded = {}
        for fps in (24.0, 60.0):
            cls.coded[fps] = root / f"coded_{fps:g}.mp4"
            mf.write_coded(cls.coded[fps], _N, fps)
        whole = root / "whole.mp4"
        synth.make_clip(whole, ["static", "pan_right"], n_each=120)
        blob = mf.faststart(whole.read_bytes())
        mdat = mf.find_box(blob, b"mdat")
        # moov first cuts of mdat: the count still says 240 frames. at 3
        # percent about 25 frames decode, less than one window
        cls.cuts = {}
        for pct in (3, 75, 90, 95):
            cls.cuts[pct] = root / f"cut_{pct}.mp4"
            cls.cuts[pct].write_bytes(
                blob[:mdat.body + (mdat.end - mdat.body) * pct // 100])

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()
        super().tearDownClass()

    def test_tail_retry_never_splices(self):
        with _overstated(29):
            frames, times, _ = reader.read_window(self.red_blue, 4.958)
        pattern = "".join("B" if _blue(f) else "R" for f in frames)
        self.assertEqual(pattern, "B" * 29, "end window reaches the head")
        self.assertGreater(times[0], 3.0)

    def test_count_past_the_end_reads_the_same_window_as_a_true_count(self):
        # a count long by any amount, less than a window or more, gives the
        # window a true count gives: the last frames, never a scrap
        for fps, path in self.coded.items():
            want = _idx(reader.read_window(path, 1e9)[0])
            self.assertGreaterEqual(want[-1], _N - 2)
            for extra in (1, 15, 27, 29, 71):
                with self.subTest(fps=fps, extra=extra), _overstated(extra):
                    got = _idx(reader.read_window(path, 1e9)[0])
                    # a stride of two may land a frame later, never shorter
                    self.assertEqual(len(got), len(want), got)
                    self.assertIn(got[0] - want[0], (0, 1), got)

    def test_truncated_moov_first_window_comes_from_the_tail(self):
        for pct in (75, 90, 95):
            with self.subTest(pct=pct):
                with synth._quiet_stderr(True):
                    frames, times, _ = reader.read_window(self.cuts[pct], 8.75)
                self.assertGreater(times[0], 5.0, "end window fell back to the head")
                # every frame is a real decode, in order, about one window long
                self.assertEqual(times, sorted(times))
                self.assertLessEqual(times[-1] - times[0], config.WINDOW_SECONDS)
                self.assertGreaterEqual(times[-1] - times[0],
                                        0.9 * config.WINDOW_SECONDS, times)

    def test_nothing_past_the_start_window_raises(self):
        stub = self.cuts[3]
        frames, _, _ = reader.read_window(stub, 0.0)
        self.assertGreaterEqual(len(frames), 2)
        with self.assertRaisesRegex(reader.ClipReadError, "truncated"):
            reader.read_window(stub, 8.75)


# -- probe ------------------------------------------------------------------------

def _box(kind: bytes, body: bytes) -> bytes:
    return struct.pack(">I", 8 + len(body)) + kind + body


def _part(blob: bytes, box) -> bytes:
    return blob[box.start:box.end]


class TestProbeBase(TempDirsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._shared = tempfile.TemporaryDirectory(prefix="clipengine-probe-")
        root = Path(cls._shared.name)
        frames = synth.frames_for("pan_right", n=48)
        cls.mp4 = root / "two.mp4"
        synth.write_clip(cls.mp4, frames)
        cls.mov = root / "two.mov"
        synth.write_clip(cls.mov, frames)
        cls.avi = root / "five.avi"
        synth.write_clip(cls.avi, synth.frames_for("pan_right", n=_N),
                         fourcc="MJPG")
        cls.blob = cls.mp4.read_bytes()

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()
        super().tearDownClass()

    def write(self, name: str, blob: bytes) -> Path:
        path = self.tmp / name
        path.write_bytes(blob)
        return path

    def test_probe_reports_frames_codec_and_decoder_duration(self):
        meta = probe.probe(self.mp4)
        self.assertEqual(meta["frames"], 48)
        self.assertAlmostEqual(meta["duration_cv2"], 2.0)
        self.assertTrue(meta["codec"])
        self.assertNotIn("created", meta)

    def test_probe_mov_uses_container_duration(self):
        self.assertAlmostEqual(probe.parse_container(self.mov)["duration_s"], 2.0)
        self.assertAlmostEqual(probe.probe(self.mov)["duration_s"], 2.0)

    def test_probe_non_container_extension_uses_decoder_math(self):
        meta = probe.probe(self.avi)
        self.assertAlmostEqual(meta["duration_s"], 5.0)
        self.assertNotIn("created", meta)

    def test_probe_unopenable_returns_zero_duration(self):
        folder = self.tmp / "x.mp4"
        folder.mkdir()
        cases = {"missing": self.tmp / "missing.mp4",
                 "zero byte": self.write("empty.mp4", b""),
                 "junk mov": self.write("junk.mov", b"not a movie " * 40),
                 "folder": folder,
                 "half cut": self.write("half.mp4",
                                        self.blob[:len(self.blob) // 2])}
        for label, path in cases.items():
            with self.subTest(label):
                self.assertEqual(probe.probe(path), {"duration_s": 0.0})

    def test_parse_container_layouts(self):
        blob = self.blob
        top = {b.kind: b for b in mf.boxes(blob)}
        moov, mdat = top[b"moov"], top[b"mdat"]
        head = blob[:mdat.start]
        self.assertLess(mdat.start, moov.start, "writer put moov last")
        kids = list(mf.boxes(blob, moov.body, moov.end))
        mvhd = [_part(blob, k) for k in kids if k.kind == b"mvhd"]
        rest = [_part(blob, k) for k in kids if k.kind != b"mvhd"]
        self.assertTrue(any(k.kind == b"trak" for k in kids))
        mdat_body = blob[mdat.body:mdat.end]
        large = (struct.pack(">I", 1) + b"mdat"
                 + struct.pack(">Q", 16 + len(mdat_body)) + mdat_body)
        to_end_moov = (struct.pack(">I", 0) + b"moov"
                       + blob[moov.body:moov.end])
        good = {"v1 mvhd": None,
                "largesize mdat": head + large + _part(blob, moov),
                "moov before mdat": mf.faststart(blob),
                "trak before mvhd": head + _part(blob, mdat)
                + _box(b"moov", b"".join(rest + mvhd)),
                "size 0 moov": head + _part(blob, mdat) + to_end_moov}
        for label, data in good.items():
            with self.subTest(label):
                if data is None:
                    path = self.write("v1.mp4", blob)
                    mf.set_mvhd(path, 1)
                else:
                    path = self.write("layout.mp4", data)
                meta = probe.parse_container(path)
                self.assertAlmostEqual(meta["duration_s"], 2.0)
        to_end_mdat = struct.pack(">I", 0) + b"mdat" + mdat_body
        bad = {"size 0 mdat before moov": head + to_end_mdat + _part(blob, moov),
               "moov without mvhd": head + _part(blob, mdat)
               + _box(b"moov", b"".join(rest))}
        for label, data in bad.items():
            with self.subTest(label):
                path = self.write("bad.mp4", data)
                self.assertEqual(probe.parse_container(path), {})


# -- reader -----------------------------------------------------------------------

class TestReaderBase(TempDirsMixin, unittest.TestCase):
    """frame coded clips, so every window names the exact frames it read."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._shared = tempfile.TemporaryDirectory(prefix="clipengine-read-")
        cls.root = Path(cls._shared.name)
        cls.coded = cls.root / "coded.mp4"
        mf.write_coded(cls.coded, _N, synth.FPS)
        cls.rates = {}
        for fps in (24.0, 25.0, 30.0, 60.0, 120.0):
            path = cls.root / f"rate_{fps:g}.mp4"
            mf.write_coded(path, int(2 * fps), fps)
            cls.rates[fps] = path

    @classmethod
    def tearDownClass(cls):
        cls._shared.cleanup()
        super().tearDownClass()

    def test_read_window_start_is_consecutive_from_zero(self):
        frames, times, fps = reader.read_window(self.coded, 0.0)
        self.assertEqual(_idx(frames), list(range(29)))
        self.assertEqual(fps, 24.0)
        for frame in frames:
            self.assertLessEqual(frame.shape[1], config.ANALYSIS_WIDTH)

    def test_read_window_length_follows_fps_and_cap(self):
        for fps, want in ((24.0, 29), (25.0, 30), (30.0, 36)):
            with self.subTest(fps=fps):
                frames, _, _ = reader.read_window(self.rates[fps], 0.0)
                self.assertEqual(len(frames), want)
        for fps in (60.0, 120.0):
            with self.subTest(fps=fps):
                frames, _, _ = reader.read_window(self.rates[fps], 0.0)
                self.assertLessEqual(len(frames), config.MAX_WINDOW_FRAMES)

    def test_read_window_tail_clamp_and_negative_start(self):
        frames, _, _ = reader.read_window(self.coded, 1e9)
        self.assertEqual(_idx(frames), list(range(91, 120)))
        frames, _, _ = reader.read_window(self.coded, -3.0)
        self.assertEqual(_idx(frames), list(range(29)))

    def test_read_pair_and_read_frame_positions(self):
        for t, want in ((1.0, [24, 25]), (99.0, [118, 119]), (-1.0, [0, 1])):
            with self.subTest(t=t):
                pair = reader.read_pair(self.coded, t)
                self.assertEqual(_idx(pair[:2]), want)
        frame = reader.read_frame(self.coded, 2.0, 64)
        self.assertEqual(frame.shape, (48, 64, 3))
        self.assertEqual(mf.decode_index(frame), 48)

    def test_read_window_seeks_to_exact_frame(self):
        # five frames one frame apart, so the seek itself is what is seen
        for fourcc, ext in (("mp4v", "mp4"), ("MJPG", "avi"),
                            ("avc1", "mp4"), ("hvc1", "mp4")):
            with self.subTest(fourcc=fourcc):
                path = self.tmp / f"seek_{fourcc}.{ext}"
                try:
                    mf.write_coded(path, 72, synth.FPS, fourcc=fourcc)
                except synth.WriterUnavailable as exc:
                    self.skipTest(str(exc))
                frames, _, _ = reader.read_window(path, 2.0, 5 / synth.FPS,
                                                  max_frames=5)
                self.assertEqual(_idx(frames), list(range(48, 53)))

    def test_downscale_geometry(self):
        cases = (((2160, 3840), (180, 320)), ((1920, 1080), (569, 320)))
        for (h, w), want in cases:
            with self.subTest(size=(w, h)):
                out = reader._downscale(np.zeros((h, w, 3), np.uint8), 320)
                self.assertEqual(out.shape[:2], want)
        for w in (320, 160):
            with self.subTest(width=w):
                frame = np.zeros((90, w, 3), np.uint8)
                self.assertIs(reader._downscale(frame, 320), frame)

    def test_reader_raises_clipreaderror_on_unopenable(self):
        junk = self.tmp / "junk.mp4"
        junk.write_bytes(b"not a movie " * 40)
        empty = self.tmp / "empty.mp4"
        empty.write_bytes(b"")
        for path in (self.tmp / "missing.mp4", junk, empty):
            for call in (reader.read_window, reader.read_pair):
                with self.subTest(path=path.name, call=call.__name__):
                    with self.assertRaises(reader.ClipReadError) as cm:
                        call(path, 0.0)
                    self.assertIn(str(path), str(cm.exception))

    def test_short_clips_window_is_whole_clip(self):
        for n in (2, 10):
            path = self.tmp / f"short_{n}.mp4"
            mf.write_coded(path, n, synth.FPS)
            for start in (0.0, 1e9):
                with self.subTest(n=n, start=start):
                    frames, _, _ = reader.read_window(path, start)
                    self.assertEqual(_idx(frames), list(range(n)))

    def test_rotation_reports_display_orientation(self):
        # a white block in the top left of the stored frame
        frame = np.zeros((synth.H, synth.W, 3), np.uint8)
        frame[:30, :40] = 255
        path = self.tmp / "portrait.mp4"
        synth.write_clip(path, [frame] * 48)
        mf.set_rotation(path, 90)
        meta = probe.probe(path)
        self.assertEqual((meta["width"], meta["height"]), (synth.H, synth.W))
        frames, _, _ = reader.read_window(path, 0.0)
        self.assertEqual(frames[0].shape, (synth.W, synth.H, 3))
        self.assertGreater(frames[0][:8, -8:].mean(), 128, "top right is lit")
        self.assertLess(frames[0][:8, :8].mean(), 128, "top left is dark")

    def test_awkward_paths_probe_and_read(self):
        src = self.rates[24.0]
        names = ("has spaces.mp4", "100% sunset.mp4", "clip%03d.mp4",
                 "take #2.mp4", "why?.mp4", "beach \U0001f30a.mp4",
                 "cafe\u0301.mp4")
        for name in names:
            with self.subTest(name=name):
                path = self.tmp / name
                shutil.copyfile(src, path)
                self.assertEqual(probe.probe(path)["frames"], 48)
                frames, _, _ = reader.read_window(path, 0.0)
                self.assertEqual(len(frames), 29)


# -- held frames and broken boxes -------------------------------------------------

class TestEdgeTiming(TempDirsMixin, unittest.TestCase):
    def test_vfr_held_frames_stay_within_one_frame(self):
        # phone jitter: 30 fps where every 10th frame is held for two ticks
        path = self.tmp / "held.mp4"
        mf.write_coded(path, 90, 30.0)
        deltas = [2 / 30 if i % 10 == 9 else 1 / 30 for i in range(90)]
        pts = mf.rewrite_stts(path, deltas)
        for t in (0.5, 1.0, 1.7, 2.4):
            want = int(np.searchsorted(pts, t + 1e-6, side="right")) - 1
            with self.subTest(t=t):
                pair = reader.read_pair(path, t)
                self.assertIsNotNone(pair)
                self.assertEqual(mf.decode_index(pair[0]), want)
                frames, _, _ = reader.read_window(path, t, 5 / 30,
                                                  max_frames=5)
                # the window is timed, so a held frame shortens it by one
                got = _idx(frames)
                self.assertGreaterEqual(len(got), 4)
                self.assertEqual(got, list(range(want, want + len(got))))


class TestCorruptBoxes(_SharedClip):
    def test_corrupt_box_sizes_return_empty(self):
        blob = self.src.read_bytes()
        top = {b.kind: b for b in mf.boxes(blob)}
        moov, mdat = top[b"moov"], top[b"mdat"]
        head = blob[:mdat.start]
        past_eof = (head + _part(blob, mdat)
                    + struct.pack(">I", moov.end - moov.start + 4096)
                    + b"moov" + blob[moov.body:moov.end])
        tiny_free = struct.pack(">I", 4) + b"free" + blob
        short_mvhd = (head + _part(blob, mdat)
                      + _box(b"moov", _box(b"mvhd", b"\0" * 2)))
        cases = {"moov past eof": past_eof, "4 byte free": tiny_free,
                 "10 byte mvhd": short_mvhd, "empty file": b""}
        for label, data in cases.items():
            with self.subTest(label):
                path = self.tmp / "corrupt.mp4"
                path.write_bytes(data)
                self.assertEqual(probe.parse_container(path), {})
                meta = probe.probe(path)
                if label in ("10 byte mvhd", "empty file"):
                    self.assertEqual(meta, {"duration_s": 0.0})
                    continue
                # ffmpeg still opens these, so only decoder math is left
                self.assertNotIn("created", meta)
                self.assertGreaterEqual(meta["frames"], 0, meta)
                self.assertAlmostEqual(meta["duration_s"],
                                       meta.get("duration_cv2", 0.0))


if __name__ == "__main__":
    unittest.main()
