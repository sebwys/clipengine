# test_probe.py
# the mp4 box parser is exercised two ways: against a hand built binary
# fixture with known values (proving we read the spec correctly) and
# against a real file written by opencv (proving we survive real output).

import struct
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from clipengine import probe
from tests import synth


def _box(btype: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + btype + payload


def build_minimal_mp4(ctime_utc: datetime, timescale: int,
                      duration_units: int) -> bytes:
    """smallest structure parse_container understands: ftyp + moov/mvhd."""
    qt_epoch = datetime(1904, 1, 1, tzinfo=timezone.utc)
    ctime = int((ctime_utc - qt_epoch).total_seconds())
    mvhd_payload = (bytes([0, 0, 0, 0])  # version 0 + flags
                    + struct.pack(">IIII", ctime, ctime,
                                  timescale, duration_units)
                    + b"\x00" * 80)
    ftyp = _box(b"ftyp", b"isom\x00\x00\x02\x00isomiso2")
    moov = _box(b"moov", _box(b"mvhd", mvhd_payload))
    return ftyp + moov


class TestContainerParser(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_duration_and_created(self):
        shot = datetime(2026, 7, 4, 12, 0, 0, tzinfo=timezone.utc)
        blob = build_minimal_mp4(shot, timescale=600, duration_units=2400)
        path = self.tmp / "fixture.mp4"
        path.write_bytes(blob)
        meta = probe.parse_container(path)
        self.assertAlmostEqual(meta["duration_s"], 4.0, places=3)
        self.assertTrue(meta["created"].startswith("2026-07-04T12:00:00"))

    def test_garbage_returns_empty(self):
        path = self.tmp / "junk.mp4"
        path.write_bytes(b"this is not an mp4 file at all, sorry")
        self.assertEqual(probe.parse_container(path), {})

    def test_truncated_returns_empty(self):
        shot = datetime(2026, 1, 1, tzinfo=timezone.utc)
        blob = build_minimal_mp4(shot, 600, 600)
        path = self.tmp / "cut.mp4"
        path.write_bytes(blob[:20])
        self.assertEqual(probe.parse_container(path), {})


class TestRealFileProbe(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.path = self.tmp / "pan.mp4"
        synth.write_clip(self.path, synth.frames_for("pan_right", n=48))

    def test_probe_merges_sane_metadata(self):
        meta = probe.probe(self.path)
        self.assertAlmostEqual(meta["fps"], synth.FPS, delta=0.5)
        self.assertEqual(meta["width"], synth.W)
        self.assertEqual(meta["height"], synth.H)
        # 48 frames at 24 fps -> 2 s, allow container rounding
        self.assertAlmostEqual(meta["duration_s"], 2.0, delta=0.25)


if __name__ == "__main__":
    unittest.main()
