# test_catalog.py
# the catalog is the engine's memory. these tests build a fake footage
# tree on disk and verify: path semantics (profile/country), stat only
# eviction awareness, idempotent rescans, change detection, and the
# pending/analyzed lifecycle.

import os
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from clipengine import catalog, config, features
from tests.util import TempDirsMixin


def touch_video(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v" * size)


class CatalogBase(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "Sony FX300001.MP4")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "Sony FX300002.MP4")
        touch_video(self.root / "DJI DLOG-M" / "Italy" / "DJI_0042.mp4")
        touch_video(self.root / "Unknown Cam" / "loose.mov")
        # junk that must be ignored
        (self.root / "Sony SLOG-3" / "Japan" / "Sony FX300001M01.XML"
         ).write_text("<x/>")
        (self.root / ".DS_Store").write_bytes(b"\x00")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "._ghost.mp4")
        # dotfiles are skipped: ._ghost starts with '.'
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)


class TestScan(CatalogBase):
    def test_first_scan_counts_and_classification(self):
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["seen"], 4)
        self.assertEqual(stats["new"], 4)
        self.assertEqual(stats["evicted"], 0)
        rows = {r["name"]: r for r in
                self.conn.execute("SELECT * FROM clips")}
        self.assertEqual(len(rows), 4)
        jp = rows["Sony FX300001.MP4"]
        self.assertEqual(jp["profile"], "sony_slog3")
        self.assertEqual(jp["country"], "Japan")
        self.assertEqual(jp["is_log"], 1)
        dji = rows["DJI_0042.mp4"]
        self.assertEqual(dji["profile"], "dji_dlogm")
        self.assertEqual(dji["country"], "Italy")
        loose = rows["loose.mov"]
        self.assertEqual(loose["profile"], "Unknown Cam")
        self.assertEqual(loose["country"], "Unsorted")
        self.assertEqual(loose["is_log"], 0)

    def test_rescan_is_idempotent(self):
        catalog.scan(self.conn, self.root)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["new"], 0)
        self.assertEqual(stats["changed"], 0)
        count = self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        self.assertEqual(count, 4)

    def test_modified_file_changes_content_key(self):
        catalog.scan(self.conn, self.root)
        old = self.conn.execute(
            "SELECT content_key FROM clips WHERE name='DJI_0042.mp4'"
        ).fetchone()[0]
        target = self.root / "DJI DLOG-M" / "Italy" / "DJI_0042.mp4"
        target.write_bytes(b"v" * 9999)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["changed"], 1)
        new = self.conn.execute(
            "SELECT content_key FROM clips WHERE name='DJI_0042.mp4'"
        ).fetchone()[0]
        self.assertNotEqual(old, new)

    def test_deleted_file_goes_missing(self):
        catalog.scan(self.conn, self.root)
        (self.root / "Unknown Cam" / "loose.mov").unlink()
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["missing"], 1)
        row = self.conn.execute(
            "SELECT missing FROM clips WHERE name='loose.mov'").fetchone()
        self.assertEqual(row["missing"], 1)

    def test_evicted_stub_is_cataloged_not_pending(self):
        with mock.patch.object(catalog, "is_materialized",
                               return_value=False):
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["evicted"], 4)
        self.assertEqual(len(catalog.pending(self.conn)), 0)

    def test_oversize_is_cataloged_not_pending(self):
        with mock.patch.object(config, "MAX_ANALYZE_BYTES", 100):
            stats = catalog.scan(self.conn, self.root)
            self.assertEqual(stats["oversize"], 4)
            self.assertEqual(len(catalog.pending(self.conn)), 0)


class TestLifecycle(CatalogBase):
    def _vector_blob(self) -> bytes:
        scalars = {n: 0.5 for n in features.FIELDS
                   if not n.startswith(("start_hue_", "end_hue_"))}
        hue = np.full(features.HUE_BINS, 1.0 / features.HUE_BINS)
        return features.to_bytes(features.pack(scalars, hue, hue))

    def test_pending_then_analyzed(self):
        catalog.scan(self.conn, self.root)
        rows = catalog.pending(self.conn)
        self.assertEqual(len(rows), 4)
        first = rows[0]
        catalog.save_features(self.conn, first["id"], first["content_key"],
                              self._vector_blob(), '{"duration_s": 2}')
        self.assertEqual(len(catalog.pending(self.conn)), 3)
        analyzed = catalog.analyzed(self.conn)
        self.assertEqual(len(analyzed), 1)
        self.assertEqual(analyzed[0]["id"], first["id"])

    def test_stale_features_reenter_pending(self):
        catalog.scan(self.conn, self.root)
        row = catalog.pending(self.conn)[0]
        catalog.save_features(self.conn, row["id"], row["content_key"],
                              self._vector_blob(), "{}")
        target = Path(row["path"])
        target.write_bytes(b"v" * 12345)  # file changed on disk
        catalog.scan(self.conn, self.root)
        pending_ids = {r["id"] for r in catalog.pending(self.conn)}
        self.assertIn(row["id"], pending_ids)

    def test_error_rows_do_not_count_as_analyzed(self):
        catalog.scan(self.conn, self.root)
        row = catalog.pending(self.conn)[0]
        catalog.save_features(self.conn, row["id"], row["content_key"],
                              None, None, error="decode blew up")
        self.assertEqual(len(catalog.analyzed(self.conn)), 0)
        ov = catalog.overview(self.conn)
        self.assertEqual(ov["errors"], 1)

    def test_country_filter_and_find_clip(self):
        catalog.scan(self.conn, self.root)
        japan = catalog.pending(self.conn, country="Japan")
        self.assertEqual(len(japan), 2)
        hit = catalog.find_clip(self.conn, "FX300002")
        self.assertIsNotNone(hit)
        self.assertEqual(hit["country"], "Japan")
        by_id = catalog.find_clip(self.conn, str(hit["id"]))
        self.assertEqual(by_id["id"], hit["id"])


if __name__ == "__main__":
    unittest.main()
