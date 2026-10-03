# test_fcpxml.py
# the fcpxml exporter must produce frame accurate rational times and
# properly escaped file urls, or fcp/resolve will misalign or fail to
# relink the import. parsing the output back is the proof.

import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from clipengine import fcpxml

META = [
    {"id": 1, "name": "clip & one.mp4", "country": "Japan",
     "path": "/x/The Footage/clip & one.mp4", "duration_s": 2.002,
     "fps": 23.976, "width": 3840, "height": 2160},
    {"id": 2, "name": "two.mp4", "country": "Iceland",
     "path": "/x/two.mp4", "duration_s": 4.0,
     "fps": 25.0, "width": 1920, "height": 1080},
]


class TestFrameDuration(unittest.TestCase):
    def test_ntsc_rates_map_to_1001(self):
        self.assertEqual(fcpxml.frame_duration(23.976), (1001, 24000))
        self.assertEqual(fcpxml.frame_duration(29.97), (1001, 30000))
        self.assertEqual(fcpxml.frame_duration(59.94), (1001, 60000))

    def test_integer_rates(self):
        self.assertEqual(fcpxml.frame_duration(25.0), (100, 2500))
        self.assertEqual(fcpxml.frame_duration(30.0), (100, 3000))

    def test_missing_fps_defaults_to_24(self):
        self.assertEqual(fcpxml.frame_duration(0), (100, 2400))

    def test_oddball_rate_still_rational(self):
        self.assertEqual(fcpxml.frame_duration(48.0), (100, 4800))


class TestExport(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_structure_and_frame_accurate_times(self):
        path = fcpxml.export_fcpxml(META, "momentum",
                                    out_dir=self.tmp, stamp="t1")
        text = path.read_text()
        self.assertIn("<!DOCTYPE fcpxml>", text)
        root = ET.parse(path).getroot()
        self.assertEqual(root.tag, "fcpxml")

        formats = root.findall("./resources/format")
        self.assertEqual({f.get("frameDuration") for f in formats},
                         {"1001/24000s", "100/2500s"})

        assets = root.findall("./resources/asset")
        self.assertEqual(len(assets), 2)
        # 2.002 s at 23.976 fps is exactly 48 frames of 1001/24000 s
        self.assertEqual(assets[0].get("duration"), "48048/24000s")
        # 4.0 s at 25 fps is 100 frames of 100/2500 s
        self.assertEqual(assets[1].get("duration"), "10000/2500s")

        rep = assets[0].find("media-rep")
        self.assertEqual(rep.get("kind"), "original-media")
        self.assertTrue(rep.get("src").startswith("file:///"))
        self.assertIn("%20", rep.get("src"))  # spaces url encoded

        clips = root.findall(".//spine/asset-clip")
        self.assertEqual([c.get("name") for c in clips],
                         ["clip & one.mp4", "two.mp4"])
        self.assertEqual(clips[0].get("ref"), assets[0].get("id"))
        self.assertEqual(clips[0].get("duration"),
                         assets[0].get("duration"))

    def test_defaults_for_missing_metadata(self):
        meta = [{"id": 1, "name": "x.mp4", "country": "X",
                 "path": "/x/x.mp4", "duration_s": None,
                 "fps": None, "width": None, "height": None}]
        path = fcpxml.export_fcpxml(meta, "calm",
                                    out_dir=self.tmp, stamp="t2")
        root = ET.parse(path).getroot()
        asset = root.find("./resources/asset")
        # 1 s fallback at 24 fps fallback: 24 frames of 100/2400 s
        self.assertEqual(asset.get("duration"), "2400/2400s")
        fmt = root.find("./resources/format")
        self.assertIsNone(fmt.get("width"))  # unknown geometry omitted


if __name__ == "__main__":
    unittest.main()
