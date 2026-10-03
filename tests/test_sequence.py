# test_sequence.py
# beam search on hand built matrices. the key fixture is a greedy trap:
# the single best first edge leads into a dead end, so only a searcher
# that keeps alternatives alive finds the chain with the higher total.

import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from clipengine import sequence


def matrix(n: int, base: float = 0.1) -> np.ndarray:
    m = np.full((n, n), base, dtype=np.float32)
    np.fill_diagonal(m, -1.0)
    return m


class TestBeamSearch(unittest.TestCase):
    def test_beats_greedy_trap(self):
        m = matrix(5, base=0.05)
        m[0, 1] = 0.95   # tempting first hop into a dead end
        m[0, 2] = 0.90
        m[2, 4] = 0.80
        m[4, 3] = 0.70
        rows, edges = sequence.build_chain(m, seed_row=0, length=4)
        self.assertEqual(rows, [0, 2, 4, 3])
        self.assertAlmostEqual(sum(edges), 0.90 + 0.80 + 0.70, places=5)

    def test_no_repeats_and_seed_first(self):
        m = matrix(6, base=0.5)
        rows, _ = sequence.build_chain(m, seed_row=3, length=6)
        self.assertEqual(rows[0], 3)
        self.assertEqual(len(rows), len(set(rows)))

    def test_length_caps_at_library_size(self):
        m = matrix(4, base=0.5)
        rows, _ = sequence.build_chain(m, seed_row=0, length=99)
        self.assertLessEqual(len(rows), 4)

    def test_zero_edges_stop_the_chain(self):
        m = matrix(3, base=-1.0)  # nothing is a valid cut
        rows, edges = sequence.build_chain(m, seed_row=1, length=3)
        self.assertEqual(rows, [1])
        self.assertEqual(edges, [])

    def test_same_country_mode_stays_home(self):
        m = matrix(5, base=0.5)
        countries = ["A", "A", "B", "B", "C"]
        rows, _ = sequence.build_chain(m, 0, 5, countries=countries,
                                       country_mode="same")
        self.assertTrue(all(countries[r] == "A" for r in rows))

    def test_travel_mode_prefers_variety(self):
        m = matrix(5, base=0.5)
        countries = ["A", "A", "B", "B", "C"]
        rows, _ = sequence.build_chain(m, 0, 3, countries=countries,
                                       country_mode="travel")
        self.assertNotEqual(countries[rows[1]], "A")

    def test_bad_seed_raises(self):
        with self.assertRaises(IndexError):
            sequence.build_chain(matrix(3), seed_row=7, length=2)


class TestExport(unittest.TestCase):
    def test_writes_json_and_m3u8(self):
        meta = [{"id": 1, "name": "a.mp4", "country": "Japan",
                 "path": "/x/a.mp4", "duration_s": 4.2},
                {"id": 2, "name": "b.mp4", "country": "Iceland",
                 "path": "/x/b.mp4", "duration_s": 2.0}]
        with tempfile.TemporaryDirectory() as tmp:
            json_path, m3u_path, xml_path = sequence.export_chain(
                meta, edges=[0.77], mode="momentum", out_dir=Path(tmp))
            plan = json.loads(json_path.read_text())
            self.assertEqual(plan["mode"], "momentum")
            self.assertEqual(len(plan["clips"]), 2)
            self.assertIsNone(plan["clips"][0]["cut_score_from_prev"])
            self.assertAlmostEqual(
                plan["clips"][1]["cut_score_from_prev"], 0.77)
            lines = m3u_path.read_text().strip().splitlines()
            self.assertEqual(lines[0], "#EXTM3U")
            self.assertIn("/x/a.mp4", lines)
            self.assertIn("/x/b.mp4", lines)
            root = ET.parse(xml_path).getroot()
            self.assertEqual(
                len(root.findall(".//spine/asset-clip")), 2)


if __name__ == "__main__":
    unittest.main()
