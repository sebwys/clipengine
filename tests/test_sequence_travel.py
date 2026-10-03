# test_sequence_travel.py
# two sequence guards. travel mode must apply its same country penalty
# before the beam keeps its top b, or a big country folder buries every
# foreign clip. and exports made in the same second must each keep their
# own intact files under one shared stem.

import json
import threading
import unittest
import xml.etree.ElementTree as ET
from datetime import datetime
from unittest import mock

import numpy as np

from clipengine import config, sequence
from tests.util import TempDirsMixin


def matrix(n: int, base: float) -> np.ndarray:
    m = np.full((n, n), base, dtype=np.float32)
    np.fill_diagonal(m, -1.0)
    return m


class TestTravelBeam(unittest.TestCase):
    def test_foreign_clip_ranked_below_beam_width_still_wins_the_cut(self):
        # 13 home clips all beat the one foreign clip raw, but each loses
        # to it once the 0.6 penalty applies
        n = 15
        m = matrix(n, base=0.01)
        countries = ["A"] * 14 + ["B"]
        for j in range(1, 14):
            m[0, j] = 0.60 + 0.01 * j
        m[0, 14] = 0.50
        self.assertGreater(m[0, 14], 0.6 * m[0, 1:14].max())
        rows, edges = sequence.build_chain(m, 0, 2, countries=countries,
                                           country_mode="travel")
        self.assertEqual(rows, [0, 14])
        self.assertAlmostEqual(edges[0], 0.50, places=5)

    def test_travel_alternates_when_home_cuts_score_higher_raw(self):
        # a home folder of 30 clips that cut well into each other, plus 3
        # foreign clips that cut a little worse raw in and out of home
        rng = np.random.default_rng(3)
        home, away = 30, 3
        n = home + away
        m = matrix(n, base=0.01)
        m[:home, :home] = rng.uniform(0.80, 0.95, (home, home))
        m[:home, home:] = 0.70
        m[home:, :home] = 0.70
        np.fill_diagonal(m, -1.0)
        countries = ["Japan"] * home + ["Iceland"] * away
        rows, edges = sequence.build_chain(m, 0, 5, countries=countries,
                                           country_mode="travel")
        played = [countries[r] for r in rows]
        self.assertEqual(played, ["Japan", "Iceland", "Japan",
                                  "Iceland", "Japan"])
        self.assertAlmostEqual(sum(edges), 4 * 0.70, places=5)

    def test_travel_still_reports_raw_cut_scores(self):
        m = matrix(4, base=0.5)
        rows, edges = sequence.build_chain(m, 0, 4, countries=["A"] * 4,
                                           country_mode="travel")
        self.assertEqual(len(rows), 4)
        for e in edges:
            self.assertAlmostEqual(e, 0.5, places=5)

    def test_same_mode_never_leaves_home_in_a_wide_library(self):
        m = matrix(40, base=0.2)
        m[:, 20:] = 0.9
        np.fill_diagonal(m, -1.0)
        countries = ["A"] * 20 + ["B"] * 20
        rows, _ = sequence.build_chain(m, 0, 8, countries=countries,
                                       country_mode="same")
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(countries[r] == "A" for r in rows))


def meta(i: int) -> dict:
    return {"id": i, "name": f"c{i}.mp4", "country": "Japan",
            "path": f"/x/c{i}.mp4", "duration_s": 2.0, "fps": 24.0,
            "width": 160, "height": 120}


class FixedClock:
    # every call lands in the same wall clock second
    @staticmethod
    def now():
        return datetime(2026, 10, 2, 12, 0, 0, 0)


def stem_of(path) -> str:
    return path.name.rsplit(".", 1)[0]


class TestExportNames(TempDirsMixin, unittest.TestCase):
    def check_export(self, paths, ids):
        json_path, m3u_path, xml_path = paths
        self.assertEqual(json_path.suffix, ".json")
        self.assertEqual(m3u_path.suffix, ".m3u8")
        self.assertEqual(xml_path.suffix, ".fcpxml")
        self.assertEqual({stem_of(p) for p in paths}, {stem_of(json_path)},
                         "one export split across stems")
        plan = json.loads(json_path.read_text())
        self.assertEqual([c["id"] for c in plan["clips"]], ids)
        listed = [ln for ln in m3u_path.read_text().splitlines()
                  if not ln.startswith("#")]
        self.assertEqual(listed, [f"/x/c{i}.mp4" for i in ids])
        root = ET.parse(xml_path).getroot()
        names = [c.get("name") for c in root.findall(".//spine/asset-clip")]
        self.assertEqual(names, [f"c{i}.mp4" for i in ids])

    def test_two_exports_in_one_second_keep_both(self):
        with mock.patch.object(sequence, "datetime", FixedClock):
            first = sequence.export_chain([meta(1), meta(2), meta(3)],
                                          [0.5, 0.4], "momentum")
            second = sequence.export_chain([meta(7), meta(8)],
                                           [0.6], "calm")
        for a, b in zip(first, second):
            self.assertNotEqual(a, b, "second export reused the first's path")
        self.assertEqual(stem_of(first[0]), "sequence_20261002_120000")
        self.assertTrue(stem_of(second[0]).startswith(
            "sequence_20261002_120000"))
        self.check_export(first, [1, 2, 3])
        self.check_export(second, [7, 8])
        self.assertEqual(len(list(config.EXPORT_DIR.iterdir())), 6)

    def test_concurrent_exports_in_one_second_stay_intact(self):
        workers = 8
        gate = threading.Barrier(workers)
        results, errors = {}, []

        def run(k):
            ids = list(range(100 * k, 100 * k + k + 2))
            try:
                gate.wait()
                results[k] = (ids, sequence.export_chain(
                    [meta(i) for i in ids], [0.5] * (len(ids) - 1),
                    "momentum"))
            except Exception as exc:
                errors.append(exc)

        with mock.patch.object(sequence, "datetime", FixedClock):
            threads = [threading.Thread(target=run, args=(k,))
                       for k in range(workers)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(errors, [])
        stems = {stem_of(paths[0]) for _, paths in results.values()}
        self.assertEqual(len(stems), workers)
        for ids, paths in results.values():
            self.check_export(paths, ids)
        self.assertEqual(len(list(config.EXPORT_DIR.iterdir())), 3 * workers)

    def test_leftover_file_with_the_same_stem_is_not_overwritten(self):
        stale = config.EXPORT_DIR / "sequence_20261002_120000.fcpxml"
        stale.write_text("keep me")
        with mock.patch.object(sequence, "datetime", FixedClock):
            paths = sequence.export_chain([meta(1), meta(2)], [0.5],
                                          "momentum")
        self.assertEqual(stale.read_text(), "keep me")
        self.assertNotEqual(paths[2], stale)
        self.check_export(paths, [1, 2])

    def test_plan_that_cannot_be_written_leaves_no_empty_json(self):
        bad = meta(2)
        bad["path"] = config.EXPORT_DIR / "c2.mp4"
        with mock.patch.object(sequence, "datetime", FixedClock):
            with self.assertRaises(TypeError):
                sequence.export_chain([meta(1), bad], [0.5], "momentum")
        self.assertEqual(sorted(p.name for p in config.EXPORT_DIR.iterdir()),
                         [], "failed export left a claimed file behind")

    def test_failed_timeline_leaves_no_partial_files(self):
        # the json and m3u8 go out before the fcpxml, so a timeline that
        # dies half written has to take all three with it
        def half_written(meta_rows, mode, out_dir, stamp):
            (out_dir / f"sequence_{stamp}.fcpxml").write_text("<fcpx")
            raise OSError(28, "No space left on device")

        with mock.patch.object(sequence, "datetime", FixedClock):
            with mock.patch("clipengine.fcpxml.export_fcpxml",
                            side_effect=half_written):
                with self.assertRaises(OSError):
                    sequence.export_chain([meta(1), meta(2)], [0.5],
                                          "momentum")
            self.assertEqual(
                sorted(p.name for p in config.EXPORT_DIR.iterdir()), [],
                "failed export left partial files behind")
            paths = sequence.export_chain([meta(1), meta(2)], [0.5],
                                          "momentum")
        # nothing was left to push the next export onto a suffix
        self.assertEqual(stem_of(paths[0]), "sequence_20261002_120000")
        self.check_export(paths, [1, 2])


if __name__ == "__main__":
    unittest.main()
