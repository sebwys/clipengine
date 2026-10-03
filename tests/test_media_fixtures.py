# test_media_fixtures.py
# the shared fixtures have to be right before any test leans on them:
# frame codes read back through real codecs, box edits leave files the
# decoder still plays in order, and the values written read back
# through the parser the app uses. also pins that TempDirsMixin keeps
# the real footage out of reach.

import shutil
import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from clipengine import catalog, config, probe
from tests import media_fixtures as mf
from tests import synth
from tests.util import TempDirsMixin

_QT_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)
SHOT = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
SHOT_QT = int((SHOT - _QT_EPOCH).total_seconds())


def decoded(path) -> tuple[list[int], list[float], int]:
    """every frame's code and CAP_PROP_POS_MSEC in decode order, and the
    frame count the decoder reports."""
    cap = cv2.VideoCapture(str(path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx, msec = [], []
        while True:
            ok, frame = cap.read()
            if not ok:
                return idx, msec, count
            idx.append(mf.decode_index(frame))
            msec.append(cap.get(cv2.CAP_PROP_POS_MSEC))
    finally:
        cap.release()


def mvhd_fields(path) -> tuple[int, int, int, int]:
    """version, ctime, timescale and duration straight from the bytes."""
    blob = Path(path).read_bytes()
    body = mf.payload(blob, mf.find_box(blob, b"mvhd"))
    if body[0] == 1:
        ctime, _, scale, dur = struct.unpack_from(">QQIQ", body, 4)
    else:
        ctime, _, scale, dur = struct.unpack_from(">IIII", body, 4)
    return body[0], ctime, scale, dur


class FixtureCase(unittest.TestCase):
    """one 48 frame 24 fps coded clip, copied fresh for each test."""

    @classmethod
    def setUpClass(cls):
        cls._dir = tempfile.TemporaryDirectory(prefix="clipengine-fixtures-")
        cls.dir = Path(cls._dir.name)
        cls.base = cls.dir / "base.mp4"
        mf.write_coded(cls.base, 48)

    @classmethod
    def tearDownClass(cls):
        cls._dir.cleanup()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="clipengine-test-")
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def fresh(self, name: str = "clip.mp4") -> Path:
        path = self.tmp / name
        shutil.copyfile(self.base, path)
        return path


class TestFrameCodes(FixtureCase):
    def test_codes_round_trip_through_mp4v_at_24_60_and_120_fps(self):
        for fps in (24.0, 60.0, 120.0):
            with self.subTest(fps=fps):
                n = int(5 * fps)
                path = self.tmp / f"count_{fps:g}.mp4"
                mf.write_coded(path, n, fps)
                idx, _, count = decoded(path)
                self.assertEqual(count, n)
                self.assertEqual(idx, list(range(n)))

    def test_codes_survive_mjpg_avc1_and_hvc1(self):
        for fourcc, ext in (("MJPG", ".avi"), ("avc1", ".mp4"),
                            ("hvc1", ".mp4")):
            with self.subTest(fourcc=fourcc):
                path = self.tmp / f"count_{fourcc}{ext}"
                try:
                    mf.write_coded(path, 48, fourcc=fourcc)
                except synth.WriterUnavailable as exc:
                    self.skipTest(str(exc))
                self.assertEqual(decoded(path)[0], list(range(48)))

    def test_codes_read_back_at_any_frame_size(self):
        # every bit on and off, drawn big and read at window and thumb sizes
        for i in (0, 1, 0x5555, 0xAAAA, 0xFFFF):
            frame = mf.coded_frame(i, 1280, 720)
            for w, h in ((1280, 720), (320, 180), (160, 120), (64, 36)):
                with self.subTest(i=i, width=w):
                    small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    self.assertEqual(mf.decode_index(small), i)

    def test_frames_without_a_code_raise(self):
        turned = cv2.rotate(mf.coded_frame(77), cv2.ROTATE_90_CLOCKWISE)
        for name, frame in (("texture", synth.texture()),
                            ("black", np.zeros((120, 160, 3), np.uint8)),
                            ("gray", np.full((120, 160, 3), 128, np.uint8)),
                            ("turned", turned)):
            with self.subTest(frame=name), self.assertRaises(ValueError):
                mf.decode_index(frame)
        for i in (-1, 1 << mf.CODE_BITS):
            with self.subTest(i=i), self.assertRaises(ValueError):
                mf.coded_frame(i)


class TestRewriteStts(FixtureCase):
    def test_frame_times_follow_the_new_deltas(self):
        # 1 s at 48 fps, then 2 s at 24 fps
        path = self.tmp / "vfr.mp4"
        mf.write_coded(path, 96, 48.0)
        pts = mf.rewrite_stts(path, [1 / 48] * 48 + [1 / 24] * 48)
        np.testing.assert_allclose(np.diff(pts), [1 / 48] * 48 + [1 / 24] * 47,
                                   atol=1e-9)
        idx, msec, count = decoded(path)
        self.assertEqual((count, idx), (96, list(range(96))))
        np.testing.assert_allclose(msec, pts * 1000, atol=0.5)
        self.assertAlmostEqual(probe.parse_container(path)["duration_s"], 3.0)

    def test_finer_timescale_keeps_phone_jitter_exact(self):
        path = self.tmp / "jitter.mp4"
        mf.write_coded(path, 90, 30.0)
        pts = mf.rewrite_stts(path, [19 / 600, 21 / 600] * 45, timescale=600)
        steps = np.diff(decoded(path)[1])
        np.testing.assert_allclose(steps, ([19 / 0.6, 21 / 0.6] * 45)[:89],
                                   atol=1e-3)
        self.assertAlmostEqual(pts[-1], 3.0 - 21 / 600)
        self.assertAlmostEqual(probe.parse_container(path)["duration_s"], 3.0)

    def test_uniform_deltas_leave_the_file_as_written(self):
        path = self.fresh()
        mf.rewrite_stts(path, [1 / 24] * 48)
        self.assertEqual(path.read_bytes(), self.base.read_bytes())

    def test_wrong_frame_count_or_b_frames_are_refused(self):
        with self.assertRaises(ValueError):
            mf.rewrite_stts(self.fresh(), [1 / 24] * 47)
        avc = self.tmp / "avc.mp4"
        try:
            mf.write_coded(avc, 48, fourcc="avc1")
        except synth.WriterUnavailable as exc:
            self.skipTest(str(exc))
        if mf.find_box(avc.read_bytes(), b"ctts") is None:
            self.skipTest("this avc1 writer made no b frames")
        with self.assertRaises(ValueError):
            mf.rewrite_stts(avc, [1 / 24] * 48)


class TestSetMvhd(FixtureCase):
    def test_values_read_back_through_parse_container(self):
        for version in (0, 1):
            with self.subTest(version=version):
                path = self.fresh(f"v{version}.mp4")
                scale = mf.set_mvhd(path, version, SHOT_QT, 4321)
                self.assertEqual(scale, 1000)
                self.assertEqual(mvhd_fields(path), (version, SHOT_QT, 1000, 4321))
                meta = probe.parse_container(path)
                self.assertAlmostEqual(meta["duration_s"], 4.321)
                self.assertEqual(meta["created"], SHOT.isoformat())
                self.assertEqual(decoded(path)[0], list(range(48)))

    def test_none_keeps_the_written_values(self):
        path = self.fresh()
        before = mvhd_fields(path)
        mf.set_mvhd(path, 1)
        self.assertEqual(mvhd_fields(path), (1,) + before[1:])
        self.assertEqual(probe.parse_container(path),
                         probe.parse_container(self.base))

    def test_all_ones_fit_each_version(self):
        path = self.fresh()
        mf.set_mvhd(path, 0, 0xFFFFFFFF, 0xFFFFFFFF)
        self.assertEqual(mvhd_fields(path), (0, 0xFFFFFFFF, 1000, 0xFFFFFFFF))
        ones = (1 << 64) - 1
        mf.set_mvhd(path, 1, ones, ones)
        self.assertEqual(mvhd_fields(path), (1, ones, 1000, ones))
        with self.assertRaises(struct.error):
            mf.set_mvhd(path, 0, 1 << 32, 1000)


class TestLayoutAndDisplay(FixtureCase):
    def test_faststart_puts_moov_first_and_keeps_every_frame(self):
        path = self.fresh()
        blob = mf.faststart(path.read_bytes())
        kinds = [b.kind for b in mf.boxes(blob)]
        self.assertLess(kinds.index(b"moov"), kinds.index(b"mdat"))
        self.assertEqual(mf.faststart(blob), blob)
        path.write_bytes(blob)
        self.assertEqual(decoded(path)[0], list(range(48)))

    def test_edits_on_a_moov_first_file_move_chunk_offsets(self):
        # a v1 mvhd and a longer stts both grow moov ahead of the media
        path = self.fresh()
        path.write_bytes(mf.faststart(path.read_bytes()))
        mf.set_mvhd(path, 1, SHOT_QT)
        pts = mf.rewrite_stts(path, [1 / 48] * 24 + [1 / 24] * 24)
        idx, msec, _ = decoded(path)
        self.assertEqual(idx, list(range(48)))
        np.testing.assert_allclose(msec, pts * 1000, atol=0.5)
        self.assertAlmostEqual(probe.parse_container(path)["duration_s"], 1.5)

    def test_rotation_turns_the_frame_the_decoder_shows(self):
        # a white block in the top left corner of a black frame
        frame = np.zeros((synth.H, synth.W, 3), np.uint8)
        frame[:30, :40] = 255
        path = self.tmp / "turn.mp4"
        synth.write_clip(path, [frame] * 12)
        for degrees, shape, corner in ((90, (160, 120, 3), "top right"),
                                       (180, (120, 160, 3), "bottom right"),
                                       (270, (160, 120, 3), "bottom left"),
                                       (0, (120, 160, 3), "top left")):
            with self.subTest(degrees=degrees):
                mf.set_rotation(path, degrees)
                cap = cv2.VideoCapture(str(path))
                ok, shown = cap.read()
                cap.release()
                self.assertTrue(ok)
                self.assertEqual(shown.shape, shape)
                corners = {"top left": shown[:8, :8], "top right": shown[:8, -8:],
                           "bottom left": shown[-8:, :8],
                           "bottom right": shown[-8:, -8:]}
                lit = [k for k, v in corners.items() if v.mean() > 128]
                self.assertEqual(lit, [corner])


class TestSlog3(unittest.TestCase):
    def test_slog3_hits_the_published_code_values(self):
        # 10 bit codes: black 95, 18 percent gray 420, 90 percent white 598
        got = mf.slog3([0.0, 0.18, 0.9]) * 1023
        np.testing.assert_allclose(got, [95, 420, 598], atol=0.5)
        # the linear toe meets the log curve
        edge = 0.01125
        self.assertAlmostEqual(float(mf.slog3(edge - 1e-9)),
                               float(mf.slog3(edge)), places=4)


class TestTempDirsMediaRoot(unittest.TestCase):
    def test_media_root_is_an_empty_temp_folder_and_comes_back(self):
        real = config.MEDIA_ROOT
        seen = {}

        class Probe(TempDirsMixin, unittest.TestCase):
            def runTest(self):
                seen["tmp"] = self.tmp
                seen["root"] = config.MEDIA_ROOT
                seen["listing"] = list(config.MEDIA_ROOT.iterdir())
                conn = catalog.connect()
                try:
                    seen["scan"] = catalog.scan(conn)["seen"]
                finally:
                    conn.close()

        result = unittest.TestResult()
        Probe().run(result)
        self.assertEqual(result.errors + result.failures, [])
        self.assertEqual(seen["root"], seen["tmp"] / "media")
        self.assertEqual((seen["listing"], seen["scan"]), ([], 0))
        self.assertEqual(config.MEDIA_ROOT, real)
        self.assertFalse(seen["tmp"].exists())


if __name__ == "__main__":
    unittest.main()
