# test_catalog_scan.py
# scan and analyze must follow the disk as it is now. these tests cover
# failed clips under --limit, scans of a subfolder or another spelling of
# the root, folders we cannot read, moved and renamed clips, clips evicted
# between scan and analyze, profile edits in config, and opening a
# catalog written before the features table tracked is_log.

import contextlib
import io
import logging
import os
import shutil
import sqlite3
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from clipengine import analysis, catalog, cli, config, features
from tests import synth
from tests.util import TempDirsMixin


def touch_video(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v" * size)


def vector_blob() -> bytes:
    scalars = {n: 0.5 for n in features.FIELDS
               if not n.startswith(("start_hue_", "end_hue_"))}
    hue = np.full(features.HUE_BINS, 1.0 / features.HUE_BINS)
    return features.to_bytes(features.pack(scalars, hue, hue))


def quietly(fn, args) -> str:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        fn(args)
    return out.getvalue()


def analyze_all(conn) -> None:
    """fake analysis: a fresh vector for every pending clip."""
    for r in catalog.pending(conn):
        catalog.save_features(conn, r["id"], r["content_key"],
                              vector_blob(), "{}")


class TestFailedClipsDoNotStarveAnalyze(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        self.japan = self.root / "Sony SLOG-3" / "Japan"
        self.japan.mkdir(parents=True)
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)

    def _fail(self, name: str) -> None:
        row = catalog.find_clip(self.conn, name)
        catalog.save_features(self.conn, row["id"], row["content_key"],
                              None, None, error="decoder could not open")

    def test_failed_clip_does_not_block_limited_analyze(self):
        (self.japan / "A_broken.MP4").write_bytes(b"\x00" * 20000)
        synth.make_clip(self.japan / "B_good.mp4", ["static", "static"],
                        n_each=36)
        catalog.scan(self.conn, self.root)
        quietly(cli.cmd_analyze, ["--limit=1"])
        quietly(cli.cmd_analyze, ["--limit=1"])
        self.assertEqual([r["name"] for r in catalog.analyzed(self.conn)],
                         ["B_good.mp4"])

    def test_unchanged_failure_waits_behind_new_clips(self):
        touch_video(self.japan / "A_broken.MP4")
        touch_video(self.japan / "B_new.MP4")
        catalog.scan(self.conn, self.root)
        self._fail("A_broken")
        self.assertEqual(
            [r["name"] for r in catalog.pending(self.conn, limit=1)],
            ["B_new.MP4"])
        self.assertEqual([r["name"] for r in catalog.pending(self.conn)],
                         ["B_new.MP4", "A_broken.MP4"])

    def test_skip_failed_leaves_unchanged_failures_out(self):
        touch_video(self.japan / "A_broken.MP4")
        touch_video(self.japan / "B_new.MP4")
        catalog.scan(self.conn, self.root)
        self._fail("A_broken")
        names = [r["name"] for r in
                 catalog.pending(self.conn, skip_failed=True)]
        self.assertEqual(names, ["B_new.MP4"])

    def test_failures_retry_oldest_attempt_first(self):
        touch_video(self.japan / "A_broken.MP4")
        touch_video(self.japan / "B_broken.MP4")
        catalog.scan(self.conn, self.root)
        self._fail("A_broken")
        self._fail("B_broken")
        self.assertEqual(
            [r["name"] for r in catalog.pending(self.conn, limit=1)],
            ["A_broken.MP4"])
        self._fail("A_broken")
        self.assertEqual(
            [r["name"] for r in catalog.pending(self.conn, limit=1)],
            ["B_broken.MP4"])

    def test_failed_file_that_changes_is_tried_again(self):
        target = self.japan / "A_broken.MP4"
        touch_video(target)
        catalog.scan(self.conn, self.root)
        self._fail("A_broken")
        target.write_bytes(b"v" * 9999)
        catalog.scan(self.conn, self.root)
        self.assertEqual([r["name"] for r in catalog.pending(self.conn)],
                         ["A_broken.MP4"])

    def test_skip_failed_says_what_it_skipped(self):
        touch_video(self.japan / "A_broken.MP4")
        touch_video(self.japan / "B_done.MP4")
        catalog.scan(self.conn, self.root)
        self._fail("A_broken")
        done = catalog.find_clip(self.conn, "B_done")
        catalog.save_features(self.conn, done["id"], done["content_key"],
                              vector_blob(), "{}")
        text = quietly(cli.cmd_analyze, ["--skip-failed"])
        self.assertIn("skipping 1 clip(s) that failed before", text)
        self.assertIn("nothing else to analyze", text)
        self.assertNotIn("FAILED", text)

    def test_clips_gone_or_changed_since_scan_do_not_use_up_the_limit(self):
        touch_video(self.japan / "A_gone.MP4")
        touch_video(self.japan / "B_changed.MP4")
        synth.make_clip(self.japan / "C_good.mp4", ["static", "static"],
                        n_each=36)
        catalog.scan(self.conn, self.root)
        (self.japan / "A_gone.MP4").unlink()
        with open(self.japan / "B_changed.MP4", "ab") as f:
            f.write(b"\x00" * 16)
        with mock.patch.object(analysis, "analyze_clip",
                               wraps=analysis.analyze_clip) as spy:
            text = quietly(cli.cmd_analyze, ["--limit=2"])
        self.assertEqual([r["name"] for r in catalog.analyzed(self.conn)],
                         ["C_good.mp4"])
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(text.count("run scan"), 2)


class TestScanRootScope(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        for i in range(3):
            touch_video(self.root / "Sony SLOG-3" / "Japan" / f"C{i:04d}.MP4")
        for i in range(2):
            touch_video(self.root / "DJI DLOG-M" / "Italy" / f"DJI_{i:04d}.mp4")
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)
        catalog.scan(self.conn, self.root)

    def live(self, country=None) -> int:
        sql = "SELECT COUNT(*) FROM clips WHERE missing=0"
        args = ()
        if country:
            sql += " AND country=?"
            args = (country,)
        return self.conn.execute(sql, args).fetchone()[0]

    def test_subfolder_scan_keeps_other_folders_live(self):
        stats = catalog.scan(self.conn, self.root / "Sony SLOG-3" / "Japan")
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(self.live("Italy"), 2)

    def test_subfolder_scan_still_marks_deleted_clip_under_it_missing(self):
        (self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4").unlink()
        stats = catalog.scan(self.conn, self.root / "Sony SLOG-3" / "Japan")
        self.assertEqual(stats["missing"], 1)
        self.assertEqual(self.live("Japan"), 2)

    def test_clip_outside_the_root_goes_missing_only_when_gone(self):
        (self.root / "DJI DLOG-M" / "Italy" / "DJI_0000.mp4").unlink()
        catalog.scan(self.conn, self.root / "Sony SLOG-3" / "Japan")
        flags = dict(self.conn.execute(
            "SELECT name, missing FROM clips WHERE country='Italy'"))
        self.assertEqual(flags, {"DJI_0000.mp4": 1, "DJI_0001.mp4": 0})

    def test_file_root_is_rejected(self):
        target = self.root / "Sony SLOG-3" / "Japan" / "C0000.MP4"
        with self.assertRaises(NotADirectoryError):
            catalog.scan(self.conn, target)
        self.assertEqual(self.live(), 5)

    def test_cli_file_root_prints_an_error_and_exits(self):
        target = self.root / "Sony SLOG-3" / "Japan" / "C0000.MP4"
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                self.assertRaises(SystemExit) as ctx:
            cli.cmd_scan([f"--root={target}"])
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("not a folder", out.getvalue())
        self.assertEqual(self.live(), 5)

    def test_empty_root_does_not_hide_catalog(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        stats = catalog.scan(self.conn, empty)
        self.assertEqual(stats["missing"], 0)
        self.assertEqual(self.live(), 5)

    def test_subfolder_scan_classifies_new_clip_against_media_root(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "A999.MP4")
        with mock.patch.object(config, "MEDIA_ROOT", self.root):
            catalog.scan(self.conn, self.root / "Sony SLOG-3" / "Japan")
        row = self.conn.execute(
            "SELECT rel_path, profile, is_log, country FROM clips"
            " WHERE name='A999.MP4'").fetchone()
        self.assertEqual(tuple(row), ("Sony SLOG-3/Japan/A999.MP4",
                                      "sony_slog3", 1, "Japan"))

    def _a999(self):
        return tuple(self.conn.execute(
            "SELECT rel_path, profile, is_log, country FROM clips"
            " WHERE name='A999.MP4'").fetchone())

    def test_library_root_below_media_root_classifies_against_itself(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "A999.MP4")
        with mock.patch.object(config, "MEDIA_ROOT", self.tmp):
            catalog.scan(self.conn, self.root)
        self.assertEqual(self._a999(), ("Sony SLOG-3/Japan/A999.MP4",
                                        "sony_slog3", 1, "Japan"))

    def test_full_scan_repairs_clip_filed_by_a_subfolder_scan(self):
        # what an older release stored for a clip first seen by a scan
        # of its country folder
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "A999.MP4")
        with mock.patch.object(config, "MEDIA_ROOT", self.tmp / "elsewhere"):
            catalog.scan(self.conn, japan)
        # a file straight in the scanned root has no profile folder, so
        # it files as unknown now instead of under its own name
        self.assertEqual(self._a999(), ("A999.MP4", "unknown", 0,
                                        "Unsorted"))
        catalog.scan(self.conn, self.root)
        self.assertEqual(self._a999(), ("Sony SLOG-3/Japan/A999.MP4",
                                        "sony_slog3", 1, "Japan"))

    def test_clips_in_one_folder_classify_alike_whichever_scan_came_first(self):
        trip = self.tmp / "media" / "Trip2024"
        japan = trip / "Sony SLOG-3" / "Japan"
        touch_video(japan / "a.mp4", 5000)
        catalog.scan(self.conn, trip)
        touch_video(japan / "b.mp4", 6000)
        catalog.scan(self.conn)
        rows = dict((r[0], tuple(r[1:])) for r in self.conn.execute(
            "SELECT name, rel_path, profile, is_log, country FROM clips"
            " WHERE name IN ('a.mp4', 'b.mp4')"))
        self.assertEqual(rows, {
            "a.mp4": ("Sony SLOG-3/Japan/a.mp4", "sony_slog3", 1, "Japan"),
            "b.mp4": ("Sony SLOG-3/Japan/b.mp4", "sony_slog3", 1, "Japan")})

    def test_copy_scanned_while_media_root_is_unmounted_takes_no_rows(self):
        primary = self.tmp / "primary"
        for name, size in (("a.mp4", 5000), ("b.mp4", 6000)):
            touch_video(primary / "Sony SLOG-3" / "Japan" / name, size)
        with mock.patch.object(config, "MEDIA_ROOT", primary):
            catalog.scan(self.conn)
            ids = dict(self.conn.execute(
                "SELECT path, id FROM clips WHERE name IN ('a.mp4', 'b.mp4')"))
            backup = self.tmp / "backup"
            shutil.copytree(primary, backup)
            os.rename(primary, self.tmp / "unmounted")
            stats = catalog.scan(self.conn, backup)
            self.assertEqual((stats["new"], stats["moved"]), (2, 0))
            os.rename(self.tmp / "unmounted", primary)
            catalog.scan(self.conn)
        live = dict(self.conn.execute(
            "SELECT path, id FROM clips WHERE missing=0"
            " AND name IN ('a.mp4', 'b.mp4') AND path LIKE ?",
            (str(primary) + os.sep + "%",)))
        self.assertEqual(live, ids)

    def test_empty_root_says_so_once(self):
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertNoLogs("clipengine", level="WARNING"):
            text = quietly(cli.cmd_scan, [f"--root={empty}"])
        self.assertEqual(text.count("no videos found"), 1)


class TestScanRootSpelling(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        # macos temp dirs sit behind the /var symlink; resolve so a
        # relative root and an absolute one name the same folder
        self.tmp = self.tmp.resolve()
        self.root = self.tmp / "The Footage"
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "a.mp4")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "b.mp4", 5000)
        self.addCleanup(os.chdir, os.getcwd())

    def _scan(self, root_arg: str) -> str:
        return quietly(cli.cmd_scan, [f"--root={root_arg}"])

    def _rows(self):
        conn = catalog.connect()
        rows = conn.execute(
            "SELECT id, path, missing FROM clips ORDER BY id").fetchall()
        conn.close()
        return [tuple(r) for r in rows]

    def test_relative_root_keeps_ids_and_absolute_paths(self):
        self._scan(str(self.root))
        before = self._rows()
        os.chdir(self.tmp)
        self._scan("The Footage")
        after = self._rows()
        self.assertEqual(after, before)
        self.assertTrue(all(os.path.isabs(p) for _, p, _ in after))

    def test_dotdot_root_keeps_ids(self):
        self._scan(str(self.root))
        before = self._rows()
        self._scan(f"{self.root}/Sony SLOG-3/../")
        self.assertEqual(self._rows(), before)

    def test_tilde_root_is_expanded(self):
        with mock.patch.dict(os.environ, {"HOME": str(self.tmp)}):
            text = self._scan("~/The Footage")
        self.assertIn(f"scan of {self.root}", text)
        self.assertEqual(len(self._rows()), 2)

    def test_symlinked_root_keeps_ids(self):
        self._scan(str(self.root))
        ids = [i for i, _, _ in self._rows()]
        alias = self.tmp / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        self._scan(str(alias))
        rows = self._rows()
        self.assertEqual([i for i, _, _ in rows], ids)
        self.assertEqual([m for _, _, m in rows], [0, 0])


@unittest.skipIf(os.geteuid() == 0, "root ignores directory permissions")
class TestUnreadableFolder(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        self.italy = self.root / "DJI DLOG-M" / "Italy"
        for i in range(3):
            touch_video(self.italy / f"DJI_000{i}.MP4")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4")
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)
        self.assertEqual(catalog.scan(self.conn, self.root)["new"], 4)

    @contextlib.contextmanager
    def _locked(self, folder: Path):
        # unlock before teardown so the temp dir can be removed
        os.chmod(folder, 0)
        try:
            yield
        finally:
            os.chmod(folder, 0o755)

    def _scan_logged(self, folder: Path):
        records = []
        handler = logging.Handler(level=logging.WARNING)
        handler.emit = records.append
        logger = logging.getLogger("clipengine")
        logger.addHandler(handler)
        try:
            with self._locked(folder):
                stats = catalog.scan(self.conn, self.root)
        finally:
            logger.removeHandler(handler)
        return stats, [r.getMessage() for r in records]

    def test_unreadable_folder_warns_and_keeps_rows(self):
        stats, messages = self._scan_logged(self.italy)
        flags = [r[0] for r in self.conn.execute(
            "SELECT missing FROM clips WHERE country='Italy'")]
        self.assertEqual((stats["missing"], flags), (0, [0, 0, 0]),
                         "unreadable folder flagged its clips missing")
        self.assertTrue(any(str(self.italy) in m for m in messages),
                        "no warning names the unreadable folder")
        self.assertEqual(stats["unreadable"], 1)

    def test_unreadable_root_keeps_every_row(self):
        stats, _ = self._scan_logged(self.root)
        live = self.conn.execute(
            "SELECT COUNT(*) FROM clips WHERE missing=0").fetchone()[0]
        self.assertEqual((stats["missing"], live), (0, 4))

    def test_cli_scan_reports_unreadable_folders(self):
        with self._locked(self.italy), \
                self.assertLogs("clipengine", level="WARNING"):
            text = quietly(cli.cmd_scan, [f"--root={self.root}"])
        self.assertIn("gone missing 0", text)
        self.assertIn("unreadable 1", text)

    def test_cli_scan_of_unreadable_root_does_not_call_it_empty(self):
        with self._locked(self.root), \
                self.assertLogs("clipengine", level="WARNING"):
            text = quietly(cli.cmd_scan, [f"--root={self.root}"])
        self.assertIn("unreadable 1", text)
        self.assertNotIn("no videos found", text)


class TestMoveKeepsAnalysis(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        touch_video(self.root / "Sony SLOG-3" / "C0001.MP4", 4096)
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4", 5000)
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        self.ids = dict(self.conn.execute("SELECT name, id FROM clips"))

    def _analyzed_paths(self):
        return sorted(r["rel_path"] for r in catalog.analyzed(self.conn))

    def test_move_within_profile_keeps_features(self):
        src = self.root / "Sony SLOG-3" / "C0001.MP4"
        dst = self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4"
        os.rename(src, dst)
        stats = catalog.scan(self.conn, self.root)
        row = self.conn.execute("SELECT id, country FROM clips WHERE path=?",
                                (str(dst),)).fetchone()
        self.assertEqual(row["country"], "Japan")
        self.assertEqual(row["id"], self.ids["C0001.MP4"])
        self.assertEqual((stats["new"], stats["moved"], stats["missing"]),
                         (0, 1, 0))
        self.assertEqual([r["rel_path"] for r in catalog.pending(self.conn)],
                         [])
        self.assertEqual(self._analyzed_paths(),
                         ["Sony SLOG-3/Japan/C0001.MP4",
                          "Sony SLOG-3/Japan/C0002.MP4"])

    def test_case_only_rename_keeps_features(self):
        src = self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4"
        os.rename(src, src.with_name("c0002.mp4"))
        catalog.scan(self.conn, self.root)
        self.assertEqual([r["rel_path"] for r in catalog.pending(self.conn)],
                         [])
        self.assertEqual(catalog.overview(self.conn)["totals"]["analyzed"], 2)
        names = [r[0] for r in self.conn.execute(
            "SELECT name FROM clips WHERE missing=0 ORDER BY id")]
        self.assertEqual(names, ["C0001.MP4", "c0002.mp4"])

    def test_renamed_country_folder_keeps_evicted_clip_in_grid(self):
        os.rename(self.root / "Sony SLOG-3" / "Japan",
                  self.root / "Sony SLOG-3" / "Nippon")
        with mock.patch.object(catalog, "is_materialized",
                               return_value=False):
            catalog.scan(self.conn, self.root)
        self.assertIn("Sony SLOG-3/Nippon/C0002.MP4", self._analyzed_paths())

    def test_moved_library_keeps_ids_and_features(self):
        moved = self.tmp / "New Drive" / "Footage"
        moved.parent.mkdir()
        os.rename(self.root, moved)
        stats = catalog.scan(self.conn, moved)
        rows = dict(self.conn.execute(
            "SELECT name, id FROM clips WHERE missing=0"))
        self.assertEqual(rows, self.ids)
        self.assertEqual(len(catalog.analyzed(self.conn)), 2)
        total = self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0]
        self.assertEqual(total, 2)
        self.assertEqual((stats["new"], stats["moved"]), (0, 2))

    def test_move_across_log_boundary_needs_reanalysis(self):
        src = self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4"
        dst = self.root / "Unknown Cam" / "Japan" / "C0002.MP4"
        dst.parent.mkdir(parents=True)
        os.rename(src, dst)
        catalog.scan(self.conn, self.root)
        pending = [(r["name"], r["is_log"]) for r in
                   catalog.pending(self.conn)]
        self.assertEqual(pending, [("C0002.MP4", 0)])

    def test_copy_does_not_take_over_the_original_row(self):
        src = self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4"
        dst = self.root / "Sony SLOG-3" / "Iceland" / "C0002.MP4"
        dst.parent.mkdir()
        shutil.copy2(src, dst)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["new"], 1)
        row = self.conn.execute("SELECT path, missing FROM clips WHERE id=?",
                                (self.ids["C0002.MP4"],)).fetchone()
        self.assertEqual(tuple(row), (str(src), 0))

    def _link_in_selects(self, make_link):
        original = self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4"
        selects = self.root / "Sony SLOG-3" / "Selects"
        selects.mkdir()
        make_link(original, selects / "best.MP4")
        with mock.patch.object(config, "MEDIA_ROOT", self.root):
            catalog.scan(self.conn, selects)
            row = self.conn.execute("SELECT path, missing FROM clips WHERE id=?",
                                    (self.ids["C0002.MP4"],)).fetchone()
            self.assertEqual(tuple(row), (str(original), 0))
            link = self.conn.execute("SELECT id FROM clips WHERE path=?",
                                     (str(selects / "best.MP4"),)).fetchone()
            self.assertIsNotNone(link)
            self.assertNotIn(link["id"], self.ids.values())
            catalog.scan(self.conn, self.root)
        self.assertIn("Sony SLOG-3/Japan/C0002.MP4", self._analyzed_paths())

    def test_symlink_in_scanned_subfolder_leaves_the_original_row(self):
        self._link_in_selects(lambda src, dst: dst.symlink_to(src))

    def test_hard_link_in_scanned_subfolder_leaves_the_original_row(self):
        self._link_in_selects(lambda src, dst: os.link(src, dst))

    def test_move_onto_the_path_of_a_deleted_clip_keeps_analysis(self):
        # card names repeat: an older Japan/C0001.MP4 was deleted and
        # scanned away, then another card's C0001 is sorted into Japan
        old = self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4"
        touch_video(old, 6000)
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        dead = self.conn.execute("SELECT id FROM clips WHERE path=?",
                                 (str(old),)).fetchone()["id"]
        old.unlink()
        catalog.scan(self.conn, self.root)
        os.rename(self.root / "Sony SLOG-3" / "C0001.MP4", old)
        stats = catalog.scan(self.conn, self.root)
        row = self.conn.execute("SELECT id FROM clips WHERE path=?",
                                (str(old),)).fetchone()
        self.assertEqual(row["id"], self.ids["C0001.MP4"])
        self.assertEqual((stats["moved"], stats["changed"], stats["new"]),
                         (1, 0, 0))
        self.assertEqual(catalog.pending(self.conn), [])
        self.assertEqual(self._analyzed_paths(),
                         ["Sony SLOG-3/Japan/C0001.MP4",
                          "Sony SLOG-3/Japan/C0002.MP4"])
        flag = self.conn.execute("SELECT missing FROM clips WHERE id=?",
                                 (dead,)).fetchone()["missing"]
        self.assertEqual(flag, 1)

    def test_clip_moved_over_another_keeps_its_own_analysis(self):
        # finder replace: the moved clip overwrites one already cataloged
        src = self.root / "Sony SLOG-3" / "C0001.MP4"
        dst = self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4"
        os.rename(src, dst)
        stats = catalog.scan(self.conn, self.root)
        row = self.conn.execute("SELECT id FROM clips WHERE path=?",
                                (str(dst),)).fetchone()
        self.assertEqual(row["id"], self.ids["C0001.MP4"])
        self.assertEqual((stats["moved"], stats["missing"]), (1, 1))
        self.assertEqual(catalog.pending(self.conn), [])
        self.assertEqual(self._analyzed_paths(),
                         ["Sony SLOG-3/Japan/C0002.MP4"])

    def test_rename_chain_keeps_both_analyses(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        os.rename(japan / "C0002.MP4", japan / "C0002_old.MP4")
        os.rename(self.root / "Sony SLOG-3" / "C0001.MP4",
                  japan / "C0002.MP4")
        stats = catalog.scan(self.conn, self.root)
        rows = dict(self.conn.execute(
            "SELECT rel_path, id FROM clips WHERE missing=0"))
        self.assertEqual(rows, {
            "Sony SLOG-3/Japan/C0002.MP4": self.ids["C0001.MP4"],
            "Sony SLOG-3/Japan/C0002_old.MP4": self.ids["C0002.MP4"]})
        self.assertEqual((stats["moved"], stats["new"]), (2, 0))
        self.assertEqual(catalog.pending(self.conn), [])

    def test_two_vanished_rows_with_one_key_are_not_guessed(self):
        a = self.root / "Sony SLOG-3" / "Japan" / "twin_a.MP4"
        b = self.root / "Sony SLOG-3" / "Japan" / "twin_b.MP4"
        touch_video(a, 7000)
        touch_video(b, 7000)
        st = a.stat()
        os.utime(b, ns=(st.st_atime_ns, st.st_mtime_ns))
        catalog.scan(self.conn, self.root)
        twins = {r[0] for r in self.conn.execute(
            "SELECT id FROM clips WHERE name LIKE 'twin_%'")}
        b.unlink()
        dst = self.root / "Sony SLOG-3" / "Peru" / "twin.MP4"
        dst.parent.mkdir()
        os.rename(a, dst)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual((stats["new"], stats["missing"]), (1, 2))
        row = self.conn.execute("SELECT id FROM clips WHERE path=?",
                                (str(dst),)).fetchone()
        self.assertNotIn(row["id"], twins)


class TestAnalyzeRechecksDisk(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        self.clip = self.root / "Sony SLOG-3" / "Iceland" / "still.mp4"
        self.clip.parent.mkdir(parents=True)
        synth.make_clip(self.clip, ["static", "static"], n_each=36)
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)
        catalog.scan(self.conn, self.root)
        self.assertEqual(len(catalog.pending(self.conn)), 1)

    def _analyze_spied(self):
        with mock.patch.object(analysis, "analyze_clip",
                               wraps=analysis.analyze_clip) as spy:
            text = quietly(cli.cmd_analyze, [])
        return spy, text

    def _errors(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM features WHERE error IS NOT NULL"
        ).fetchone()[0]

    def test_clip_evicted_after_scan_is_not_opened(self):
        # what optimize mac storage leaves behind: same size and mtime,
        # zero allocated blocks, so the content key does not change
        st = self.clip.stat()
        self.clip.unlink()
        with open(self.clip, "wb") as f:
            f.truncate(st.st_size)
        os.utime(self.clip, ns=(st.st_atime_ns, st.st_mtime_ns))
        if self.clip.stat().st_blocks != 0:
            self.skipTest("filesystem does not make sparse files")
        spy, text = self._analyze_spied()
        spy.assert_not_called()
        self.assertEqual(self._errors(), 0)
        self.assertIn("evicted", text)
        row = self.conn.execute("SELECT available FROM clips").fetchone()
        self.assertEqual(row["available"], 0)

    def test_eviction_seen_by_a_fresh_stat_skips_the_clip(self):
        with mock.patch.object(catalog, "is_materialized",
                               return_value=False):
            spy, _ = self._analyze_spied()
        spy.assert_not_called()
        self.assertEqual(self._errors(), 0)

    def test_clip_deleted_after_scan_is_skipped(self):
        self.clip.unlink()
        spy, text = self._analyze_spied()
        spy.assert_not_called()
        self.assertEqual(self._errors(), 0)
        self.assertIn("skipped", text)

    def test_clip_changed_after_scan_waits_for_a_rescan(self):
        with open(self.clip, "ab") as f:
            f.write(b"\x00" * 16)
        spy, text = self._analyze_spied()
        spy.assert_not_called()
        self.assertEqual(self._errors(), 0)
        self.assertIn("changed on disk since the last scan", text)

    def test_plain_analyze_tries_a_known_failure_again(self):
        row = catalog.pending(self.conn)[0]
        catalog.save_features(self.conn, row["id"], row["content_key"],
                              None, None, error="decoder could not open")
        with mock.patch.object(analysis, "analyze_clip",
                               wraps=analysis.analyze_clip) as spy:
            quietly(cli.cmd_analyze, ["--skip-failed"])
        spy.assert_not_called()
        spy, _ = self._analyze_spied()
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(catalog.analyzed(self.conn)), 1)

    def test_clip_still_on_disk_is_analyzed(self):
        spy, _ = self._analyze_spied()
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(len(catalog.analyzed(self.conn)), 1)


class TestRescanReclassifies(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        touch_video(self.root / "Canon CLOG3" / "Peru" / "A001.MP4")
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)

    def _row(self, name: str):
        return self.conn.execute(
            "SELECT id, profile, is_log, country FROM clips WHERE name=?",
            (name,)).fetchone()

    def test_rescan_applies_new_log_profile(self):
        catalog.scan(self.conn, self.root)
        with mock.patch.dict(config.CAMERA_PROFILES, {
                "Canon CLOG3": {"key": "canon_clog3", "log": True}}):
            catalog.scan(self.conn, self.root)
        row = self._row("A001.MP4")
        self.assertEqual((row["profile"], row["is_log"], row["country"]),
                         ("canon_clog3", 1, "Peru"))

    def test_rescan_drops_log_flag_when_profile_changes(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4")
        catalog.scan(self.conn, self.root)
        with mock.patch.dict(config.CAMERA_PROFILES, {
                "Sony SLOG-3": {"key": "sony_slog3", "log": False}}):
            catalog.scan(self.conn, self.root)
        self.assertEqual(self._row("C0001.MP4")["is_log"], 0)

    def test_log_flag_change_puts_clip_back_in_pending(self):
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        with mock.patch.dict(config.CAMERA_PROFILES, {
                "Canon CLOG3": {"key": "canon_clog3", "log": True}}):
            catalog.scan(self.conn, self.root)
            pending = [(r["name"], r["is_log"]) for r in
                       catalog.pending(self.conn)]
        self.assertEqual(pending, [("A001.MP4", 1)])

    def test_profile_change_back_restores_features(self):
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        with mock.patch.dict(config.CAMERA_PROFILES, {
                "Canon CLOG3": {"key": "canon_clog3", "log": True}}):
            catalog.scan(self.conn, self.root)
        catalog.scan(self.conn, self.root)
        self.assertEqual(len(catalog.analyzed(self.conn)), 1)
        self.assertEqual(catalog.pending(self.conn), [])


# the catalog layout before features tracked is_log, copied verbatim so
# the migration test opens a real old database
OLD_SCHEMA = """
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    rel_path TEXT NOT NULL,
    name TEXT NOT NULL,
    profile TEXT NOT NULL,
    is_log INTEGER NOT NULL,
    country TEXT NOT NULL,
    ext TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL,
    content_key TEXT NOT NULL,
    available INTEGER NOT NULL,
    oversize INTEGER NOT NULL DEFAULT 0,
    missing INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_country ON clips(country);
CREATE INDEX IF NOT EXISTS idx_clips_available ON clips(available);

CREATE TABLE IF NOT EXISTS features (
    clip_id INTEGER PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
    version INTEGER NOT NULL,
    content_key TEXT NOT NULL,
    vector BLOB,
    summary TEXT,
    error TEXT,
    analyzed_at TEXT NOT NULL
);
"""


class TestOldCatalogMigrates(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.db = config.DATA_DIR / "old.db"
        self.blob = vector_blob()
        old = sqlite3.connect(str(self.db))
        old.executescript(OLD_SCHEMA)
        for i, (name, is_log) in enumerate((("a.mp4", 1), ("b.mp4", 0)), 1):
            old.execute(
                "INSERT INTO clips (id, path, rel_path, name, profile,"
                " is_log, country, ext, size_bytes, mtime_ns, content_key,"
                " available, oversize, missing, first_seen, last_seen)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,0,?,?)",
                (i, f"/x/{name}", name, name, "t", is_log, "Japan", ".mp4",
                 1, 1, "1-1", 1, "t", "t"))
            old.execute(
                "INSERT INTO features (clip_id, version, content_key,"
                " vector, summary, error, analyzed_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (i, config.FEATURE_VERSION, "1-1", self.blob, "{}", None, "t"))
        old.commit()
        old.close()

    def test_old_catalog_gains_is_log_and_keeps_features(self):
        conn = catalog.connect(self.db)
        self.addCleanup(conn.close)
        cols = [r["name"] for r in conn.execute("PRAGMA table_info(features)")]
        self.assertIn("is_log", cols)
        rows = conn.execute("SELECT clip_id, is_log, vector FROM features"
                            " ORDER BY clip_id").fetchall()
        self.assertEqual([(r["clip_id"], r["is_log"]) for r in rows],
                         [(1, 1), (2, 0)])
        self.assertTrue(all(r["vector"] == self.blob for r in rows))
        self.assertEqual([r["name"] for r in catalog.analyzed(conn)],
                         ["a.mp4", "b.mp4"])

    def test_second_open_leaves_migrated_catalog_alone(self):
        catalog.connect(self.db).close()
        conn = catalog.connect(self.db)
        self.addCleanup(conn.close)
        self.assertEqual(len(catalog.analyzed(conn)), 2)


if __name__ == "__main__":
    unittest.main()
