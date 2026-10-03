# test_catalog_cli.py
# the catalog as status, analyze and match see it: schema, stat only
# scans, clip identity across edits, deletes and evictions, pending
# order and limits, the overview counts, and how a typed clip name or
# country resolves. media here is plain bytes: nothing gets decoded.

import contextlib
import io
import os
import shutil
import stat
import subprocess
import unicodedata
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from clipengine import catalog, cli, config, features
from tests.util import TempDirsMixin


def touch_video(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v" * size)


def vector_blob() -> bytes:
    scalars = {n: 0.5 for n in features.FIELDS
               if not n.startswith(("start_hue_", "end_hue_"))}
    hue = np.full(features.HUE_BINS, 1.0 / features.HUE_BINS)
    return features.to_bytes(features.pack(scalars, hue, hue))


def analyze_all(conn) -> None:
    """fake analysis: a fresh vector for every pending clip."""
    for r in catalog.pending(conn):
        catalog.save_features(conn, r["id"], r["content_key"],
                              vector_blob(), "{}")


def fail(conn, name: str) -> None:
    row = catalog.find_clip(conn, name)
    catalog.save_features(conn, row["id"], row["content_key"], None, None,
                          error="decoder could not open")


def succeed(conn, name: str) -> None:
    row = catalog.find_clip(conn, name)
    catalog.save_features(conn, row["id"], row["content_key"],
                          vector_blob(), "{}")


def run_cli(argv: list) -> str:
    """stdout of one cli call. exit codes are the cli tests' business."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out), \
            contextlib.redirect_stderr(io.StringIO()):
        try:
            cli.main(argv)
        except SystemExit:
            pass
    return out.getvalue()


def evict(path: Path) -> bytes:
    """what optimize mac storage leaves: same size and mtime, no blocks.
    returns the original bytes so the clip can be downloaded again."""
    data = path.read_bytes()
    st = path.stat()
    path.unlink()
    with open(path, "wb") as f:
        f.truncate(st.st_size)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    return data


def restore(path: Path, data: bytes, mtime_ns: int) -> None:
    path.write_bytes(data)
    os.utime(path, ns=(mtime_ns, mtime_ns))


class CatalogCase(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.root = self.tmp / "The Footage"
        self.root.mkdir()
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)

    def names(self, rows) -> list:
        return [r["name"] for r in rows]

    def row(self, name: str):
        return self.conn.execute("SELECT * FROM clips WHERE name=?",
                                 (name,)).fetchone()


class TestCatalogBasics(CatalogCase):
    def test_connect_creates_schema_and_pragmas(self):
        db = self.tmp / "deep" / "er" / "catalog.db"
        conn = catalog.connect(db)
        self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0],
                         "wal")
        self.assertEqual(conn.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue({"clips", "features"} <= tables)
        conn.execute(
            "INSERT INTO clips (path, rel_path, name, profile, is_log,"
            " country, ext, size_bytes, mtime_ns, content_key, available,"
            " first_seen, last_seen) VALUES"
            " ('/x/a.mp4','a.mp4','a.mp4','t',0,'X','.mp4',1,1,'1-1',1,'t','t')")
        conn.commit()
        conn.close()
        again = catalog.connect(db)
        self.addCleanup(again.close)
        self.assertEqual(again.execute("PRAGMA foreign_keys").fetchone()[0], 1)
        self.assertEqual(
            again.execute("SELECT COUNT(*) FROM clips").fetchone()[0], 1)

    def test_scan_missing_root_raises(self):
        with self.assertRaisesRegex(FileNotFoundError, "media root not found"):
            catalog.scan(self.conn, self.tmp / "nowhere")
        self.assertEqual(
            self.conn.execute("SELECT COUNT(*) FROM clips").fetchone()[0], 0)

    @unittest.skipIf(os.geteuid() == 0, "root reads any file")
    def test_scan_is_stat_only(self):
        clip = self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4"
        touch_video(clip)
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "C0002.MP4")
        opened = []
        real_open, real_os_open = open, os.open

        def spy_open(file, *args, **kwargs):
            opened.append(file)
            return real_open(file, *args, **kwargs)

        def spy_os_open(file, *args, **kwargs):
            opened.append(file)
            return real_os_open(file, *args, **kwargs)

        os.chmod(clip, 0)
        try:
            with mock.patch("builtins.open", spy_open), \
                    mock.patch("io.open", spy_open), \
                    mock.patch.object(os, "open", spy_os_open):
                stats = catalog.scan(self.conn, self.root)
        finally:
            os.chmod(clip, 0o644)
        under = [f for f in opened
                 if str(os.fspath(f)).startswith(str(self.root))]
        self.assertEqual(under, [])
        self.assertEqual(stats["seen"], 2)
        self.assertEqual(self.row("C0001.MP4")["available"], 1)

    def test_real_sparse_stub_counts_as_evicted(self):
        stub = self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4"
        stub.parent.mkdir(parents=True)
        with open(stub, "wb") as f:
            f.truncate(50 * 1024 * 1024)
        if stub.stat().st_blocks != 0:
            self.skipTest("filesystem allocated blocks for a sparse file")
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual((stats["available"], stats["evicted"]), (0, 1))
        self.assertEqual(self.row("C0001.MP4")["available"], 0)
        self.assertEqual(catalog.pending(self.conn), [])

    def test_apfs_compressed_file_is_materialized(self):
        ditto = shutil.which("ditto")
        if not ditto:
            self.skipTest("needs ditto")
        src = self.tmp / "plain.mp4"
        src.write_bytes(b"v" * 200_000)
        dst = self.root / "Sony SLOG-3" / "Japan" / "packed.mp4"
        dst.parent.mkdir(parents=True)
        subprocess.run([ditto, "--hfsCompression", str(src), str(dst)],
                       check=True, capture_output=True)
        if not dst.stat().st_flags & stat.UF_COMPRESSED:
            self.skipTest("ditto left the copy uncompressed")
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["available"], 1)
        self.assertEqual(self.names(catalog.pending(self.conn)), ["packed.mp4"])

    def test_scan_skip_rules_and_classification(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / ".cache" / "proxy.mp4")
        touch_video(self.root / ".Trashes" / "501" / "old.mp4")
        touch_video(japan / "C0001.LRF")
        touch_video(japan / "C0002.mts")
        touch_video(japan / "C0003.m4v")
        touch_video(self.root / "DJI DLOG-M" / "Italy" / "Rome" / "deep.Mov")
        touch_video(japan / "C0004.MXF")
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["seen"], 2)
        rows = {r["name"]: tuple(r) for r in self.conn.execute(
            "SELECT name, profile, is_log, country, ext FROM clips")}
        self.assertEqual(rows, {
            "deep.Mov": ("deep.Mov", "dji_dlogm", 1, "Italy", ".mov"),
            "C0004.MXF": ("C0004.MXF", "sony_slog3", 1, "Japan", ".mxf")})

    def test_unicode_and_spaces_path(self):
        country = unicodedata.normalize("NFC", "C\u00f4te d'Ivoire")
        touch_video(self.root / "Apple ProRes-Log" / country / "My Clip 01.mov")
        catalog.scan(self.conn, self.root)
        row = self.row("My Clip 01.mov")
        self.assertEqual((row["profile"], row["is_log"], row["country"]),
                         ("apple_prores_log", 1, country))
        self.assertEqual(catalog.find_clip(self.conn, "Clip 01")["id"],
                         row["id"])

    def test_oversize_boundary(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "at_limit.MP4", 4096)
        touch_video(japan / "over_limit.MP4", 5000)
        with mock.patch.object(config, "MAX_ANALYZE_BYTES", 4096):
            stats = catalog.scan(self.conn, self.root)
            self.assertEqual(stats["oversize"], 1)
            self.assertEqual(self.names(catalog.pending(self.conn)),
                             ["at_limit.MP4"])
        self.assertEqual(self.row("over_limit.MP4")["oversize"], 1)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["oversize"], 0)
        self.assertEqual(self.row("over_limit.MP4")["oversize"], 0)
        self.assertEqual(len(catalog.pending(self.conn)), 2)

    def test_numeric_token_resolves_id_first(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        for name in ("3.MP4", "a.MP4", "b.MP4", "c.MP4"):
            touch_video(japan / name)
        catalog.scan(self.conn, self.root)
        self.assertEqual(self.row("3.MP4")["id"], 1)
        hit = catalog.find_clip(self.conn, "3")
        self.assertEqual((hit["id"], hit["name"]), (3, "b.MP4"))
        self.assertEqual(catalog.find_clip(self.conn, "3.MP")["name"], "3.MP4")

    def test_odd_numeric_tokens_do_not_crash(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "a.MP4")
        catalog.scan(self.conn, self.root)
        # a superscript passes isdigit but not int, and twenty digits
        # overflow sqlite's integer: both are just names that match nothing
        self.assertIsNone(catalog.find_clip(self.conn, "\u00b2"))
        self.assertIsNone(catalog.find_clip(self.conn, "9" * 20))
        self.assertEqual(catalog.find_clip(self.conn, " 1 ")["name"], "a.MP4")

    def test_symlinked_file_cataloged_and_broken_link_skipped(self):
        real = self.tmp / "Elsewhere" / "real.mp4"
        touch_video(real)
        japan = self.root / "Sony SLOG-3" / "Japan"
        japan.mkdir(parents=True)
        (japan / "link.mp4").symlink_to(real)
        (japan / "dead.mp4").symlink_to(self.tmp / "Elsewhere" / "gone.mp4")
        with self.assertLogs("clipengine", level="WARNING") as logs:
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["seen"], 1)
        rows = [tuple(r) for r in self.conn.execute(
            "SELECT name, path, available FROM clips")]
        self.assertEqual(rows, [("link.mp4", str(japan / "link.mp4"), 1)])
        self.assertTrue(any("dead.mp4" in m for m in logs.output))

    def test_cataloged_link_goes_missing_when_its_target_does(self):
        real = self.tmp / "Elsewhere" / "real.mp4"
        touch_video(real)
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "b.mp4", 5000)
        (japan / "link.mp4").symlink_to(real)
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        real.unlink()
        with self.assertLogs("clipengine", level="WARNING"):
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual((stats["missing"], stats["unreadable"]), (1, 0))
        self.assertEqual(self.row("link.mp4")["missing"], 1)
        self.assertEqual(self.names(catalog.analyzed(self.conn)), ["b.mp4"])


class TestClipIdentity(CatalogCase):
    def setUp(self):
        super().setUp()
        self.japan = self.root / "Sony SLOG-3" / "Japan"
        self.clip = self.japan / "C0001.MP4"
        touch_video(self.clip, 50_000)
        touch_video(self.japan / "C0002.MP4", 6000)
        catalog.scan(self.conn, self.root)
        analyze_all(self.conn)
        self.first = self.row("C0001.MP4")

    def analyzed_names(self) -> list:
        return self.names(catalog.analyzed(self.conn))

    def test_mtime_only_change_keeps_id_and_first_seen(self):
        st = self.clip.stat()
        later = st.st_mtime_ns + 10 * 10**9
        os.utime(self.clip, ns=(later, later))
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["changed"], 1)
        row = self.row("C0001.MP4")
        self.assertEqual((row["id"], row["first_seen"]),
                         (self.first["id"], self.first["first_seen"]))
        self.assertNotEqual(row["content_key"], self.first["content_key"])
        self.assertEqual(self.names(catalog.pending(self.conn)), ["C0001.MP4"])
        self.assertEqual(self.analyzed_names(), ["C0002.MP4"])

    def test_deleted_then_restored_keeps_id_and_features(self):
        data, mtime = self.clip.read_bytes(), self.clip.stat().st_mtime_ns
        self.clip.unlink()
        self.assertEqual(catalog.scan(self.conn, self.root)["missing"], 1)
        self.assertEqual(self.analyzed_names(), ["C0002.MP4"])
        restore(self.clip, data, mtime)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["new"], 0)
        row = self.row("C0001.MP4")
        self.assertEqual((row["id"], row["missing"]), (self.first["id"], 0))
        self.assertEqual(self.analyzed_names(), ["C0001.MP4", "C0002.MP4"])
        self.assertEqual(catalog.pending(self.conn), [])

    def test_evicted_then_redownloaded_keeps_features(self):
        mtime = self.clip.stat().st_mtime_ns
        data = evict(self.clip)
        if self.clip.stat().st_blocks != 0:
            self.skipTest("filesystem allocated blocks for a sparse file")
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["evicted"], 1)
        self.assertEqual(self.row("C0001.MP4")["available"], 0)
        self.assertEqual(self.analyzed_names(), ["C0001.MP4", "C0002.MP4"])
        self.assertEqual(catalog.pending(self.conn), [])
        restore(self.clip, data, mtime)
        catalog.scan(self.conn, self.root)
        row = self.row("C0001.MP4")
        self.assertEqual((row["id"], row["available"]), (self.first["id"], 1))
        self.assertEqual(catalog.pending(self.conn), [])
        self.assertEqual(self.analyzed_names(), ["C0001.MP4", "C0002.MP4"])

    def test_feature_version_bump_makes_everything_stale(self):
        with mock.patch.object(config, "FEATURE_VERSION",
                               config.FEATURE_VERSION + 1):
            self.assertEqual(self.names(catalog.pending(self.conn)),
                             ["C0001.MP4", "C0002.MP4"])
            self.assertEqual(catalog.analyzed(self.conn), [])
            ov = catalog.overview(self.conn)
            self.assertEqual(ov["totals"]["analyzed"], 0)
            self.assertEqual([c["analyzed"] for c in ov["countries"]], [0])

    def test_success_replaces_error_row(self):
        fail(self.conn, "C0002")
        self.assertEqual(catalog.overview(self.conn)["errors"], 1)
        row = self.row("C0002.MP4")
        catalog.save_features(self.conn, row["id"], row["content_key"],
                              vector_blob(), "{}")
        rows = self.conn.execute("SELECT error FROM features WHERE clip_id=?",
                                 (row["id"],)).fetchall()
        self.assertEqual([r["error"] for r in rows], [None])
        self.assertEqual(catalog.overview(self.conn)["errors"], 0)
        self.assertIn("C0002.MP4", self.analyzed_names())


class TestPendingAndOverview(CatalogCase):
    def setUp(self):
        super().setUp()
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "b.MP4")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "a.MP4")
        touch_video(self.root / "DJI DLOG-M" / "Italy" / "c.mp4")
        touch_video(self.root / "Unknown Cam" / "Peru" / "d.mov")
        catalog.scan(self.conn, self.root)

    def test_pending_order_limit_force_country(self):
        pending = lambda **kw: self.names(catalog.pending(self.conn, **kw))
        self.assertEqual(pending(), ["c.mp4", "a.MP4", "b.MP4", "d.mov"])
        self.assertEqual(pending(limit=2), ["c.mp4", "a.MP4"])
        self.assertEqual(pending(country="Japan"), ["a.MP4", "b.MP4"])
        self.assertEqual(pending(country="Japan", limit=1), ["a.MP4"])
        analyze_all(self.conn)
        self.assertEqual(pending(), [])
        self.assertEqual(sorted(pending(force=True)),
                         ["a.MP4", "b.MP4", "c.mp4", "d.mov"])
        self.assertEqual(pending(force=True, country="Peru"), ["d.mov"])

    def test_limit_zero_is_empty_and_negative_is_refused(self):
        self.assertEqual(catalog.pending(self.conn, limit=0), [])
        with self.assertRaises(ValueError):
            catalog.pending(self.conn, limit=-1)
        self.assertEqual(len(catalog.pending(self.conn, limit=1)), 1)

    def test_overview_excludes_missing_rows(self):
        (self.root / "DJI DLOG-M" / "Italy" / "c.mp4").unlink()
        catalog.scan(self.conn, self.root)
        ov = catalog.overview(self.conn)
        self.assertEqual(ov["totals"]["total"], 3)
        self.assertEqual([c["country"] for c in ov["countries"]],
                         ["Japan", "Peru"])
        self.assertEqual([p["profile"] for p in ov["profiles"]],
                         ["Unknown Cam", "sony_slog3"])

    def test_overview_counts_missing_clips(self):
        target = self.root / "DJI DLOG-M" / "Italy" / "c.mp4"
        data, mtime = target.read_bytes(), target.stat().st_mtime_ns
        self.assertEqual(catalog.overview(self.conn)["missing"], 0)
        target.unlink()
        catalog.scan(self.conn, self.root)
        self.assertEqual(catalog.overview(self.conn)["missing"], 1)
        restore(target, data, mtime)
        catalog.scan(self.conn, self.root)
        self.assertEqual(catalog.overview(self.conn)["missing"], 0)


class TestEdgeCases(CatalogCase):
    def test_error_count_ignores_gone_clips(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "bad.MP4", 5000)
        touch_video(japan / "good.mp4")
        catalog.scan(self.conn, self.root)
        fail(self.conn, "bad")
        succeed(self.conn, "good")
        self.assertEqual(catalog.overview(self.conn)["errors"], 1)
        (japan / "bad.MP4").unlink()
        catalog.scan(self.conn, self.root)
        ov = catalog.overview(self.conn)
        self.assertEqual((ov["totals"]["total"], ov["errors"]), (1, 0))
        # a failure from an older extractor or an older copy of the file
        # is stale: the clip is pending again, not broken
        fail(self.conn, "good")
        self.assertEqual(catalog.overview(self.conn)["errors"], 1)
        with mock.patch.object(config, "FEATURE_VERSION",
                               config.FEATURE_VERSION + 1):
            self.assertEqual(catalog.overview(self.conn)["errors"], 0)
        touch_video(japan / "good.mp4", 6000)
        catalog.scan(self.conn, self.root)
        self.assertEqual(catalog.overview(self.conn)["errors"], 0)

    def test_zero_byte_file_is_not_evicted(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "C0001.MP4")
        touch_video(japan / "empty.MP4", 0)
        stats = catalog.scan(self.conn, self.root)
        self.assertEqual((stats["seen"], stats["evicted"]), (2, 0))
        self.assertEqual(catalog.overview(self.conn)["totals"]["evicted"], 0)
        self.assertIn("empty.MP4", self.names(catalog.pending(self.conn)))

    def test_empty_or_wildcard_token_resolves_nothing(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "A001.MP4")
        touch_video(self.root / "DJI DLOG-M" / "Italy" / "DJI_0042.mp4")
        catalog.scan(self.conn, self.root)
        for token in ("", "   ", "%", "%%", "A_01"):
            self.assertIsNone(catalog.find_clip(self.conn, token), repr(token))
        # wildcards are literal characters, not patterns
        self.assertEqual(catalog.find_clip(self.conn, "I_0")["name"],
                         "DJI_0042.mp4")
        self.assertNotIn("#1", run_cli(["match", ""]))

    def test_country_filter_ignores_case_and_normalization(self):
        nfc = unicodedata.normalize("NFC", "T\u00fcrkiye")
        nfd = unicodedata.normalize("NFD", nfc)
        place = unicodedata.normalize("NFD", "G\u00f6reme.MP4")
        touch_video(self.root / "Sony SLOG-3" / nfd / place)
        if os.listdir(self.root / "Sony SLOG-3") != [nfd]:
            self.skipTest("filesystem normalizes file names")
        touch_video(self.root / "DJI DLOG-M" / "Italy" / "DJI_0042.mp4")
        catalog.scan(self.conn, self.root)
        count = lambda c: len(catalog.pending(self.conn, country=c))
        # case folds past ascii too, the way apfs compares names
        self.assertEqual([count(c) for c in ("italy", "ITALY", nfc, nfd,
                                             nfc.upper())],
                         [1, 1, 1, 1, 1])
        self.assertEqual(count("Itlay"), 0)
        stored = self.row(unicodedata.normalize("NFC", place))
        self.assertIsNotNone(stored, "name not stored as nfc")
        self.assertEqual(stored["country"], nfc)
        typed = "G\u00f6reme"
        for token in (typed, unicodedata.normalize("NFD", typed)):
            self.assertEqual(catalog.find_clip(self.conn, token)["id"],
                             stored["id"], ascii(token))
        fail(self.conn, "DJI_0042")
        self.assertEqual(catalog.known_failures(self.conn, "italy"), 1)
        analyze_all(self.conn)
        self.assertEqual(len(catalog.analyzed(self.conn, "italy")), 1)
        self.assertEqual(len(catalog.analyzed(self.conn, nfd)), 1)

    def test_root_level_file_profile_is_unknown(self):
        touch_video(self.root / "rootlevel.mp4")
        touch_video(self.root / "loose.mov")
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4")
        self.assertEqual(catalog.classify_path(Path("rootlevel.mp4")),
                         ("unknown", 0, "Unsorted"))
        catalog.scan(self.conn, self.root)
        for name in ("rootlevel.mp4", "loose.mov"):
            row = self.row(name)
            self.assertEqual((row["profile"], row["country"]),
                             ("unknown", "Unsorted"))
        profiles = [p["profile"] for p in catalog.overview(self.conn)["profiles"]]
        self.assertEqual(profiles, ["sony_slog3", "unknown"])
        status = run_cli(["status"])
        self.assertNotIn("rootlevel.mp4", status)
        self.assertNotIn("loose.mov", status)

    def test_overlapping_scans_do_not_crash(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "A0001.MP4")
        catalog.scan(self.conn, self.root)
        touch_video(japan / "N0001.MP4", 5000)
        other = catalog.connect()
        self.addCleanup(other.close)
        real_walk = os.walk
        fired = []

        def walk(top, *args, **kwargs):
            # the second scan catalogs the new file and commits while
            # the first one is under way
            if not fired:
                fired.append(top)
                catalog.scan(other, self.root)
            return real_walk(top, *args, **kwargs)

        with mock.patch.object(catalog.os, "walk", walk):
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual(fired, [str(self.root)])
        self.assertEqual((stats["seen"], stats["new"]), (2, 0))
        rows = [tuple(r) for r in self.conn.execute(
            "SELECT name, missing FROM clips ORDER BY name")]
        self.assertEqual(rows, [("A0001.MP4", 0), ("N0001.MP4", 0)])

    def test_clip_another_scan_adds_after_our_listing_stays_live(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "A0001.MP4")
        catalog.scan(self.conn, self.root)
        other = catalog.connect()
        self.addCleanup(other.close)
        real_walk = os.walk
        fired = []

        def walk(top, *args, **kwargs):
            # our listing is done, then a file lands and the second scan
            # catalogs it before we write a thing
            yield from real_walk(top, *args, **kwargs)
            if not fired:
                fired.append(top)
                touch_video(japan / "N0001.MP4", 5000)
                catalog.scan(other, self.root)

        with mock.patch.object(catalog.os, "walk", walk):
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual(fired, [str(self.root)])
        self.assertEqual((stats["seen"], stats["missing"]), (1, 0))
        rows = [tuple(r) for r in self.conn.execute(
            "SELECT name, missing FROM clips ORDER BY name")]
        self.assertEqual(rows, [("A0001.MP4", 0), ("N0001.MP4", 0)])
        # the next scan of our own sees it as it is
        self.assertEqual(catalog.scan(self.conn, self.root)["seen"], 2)

    def test_failed_scan_rolls_back_and_frees_the_catalog(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "A0001.MP4")
        touch_video(japan / "A0002.MP4")
        catalog.scan(self.conn, self.root)
        touch_video(japan / "A0003.MP4")
        real = catalog.classify_path
        calls = []

        def flaky(rel):
            calls.append(rel)
            if len(calls) == 2:
                raise RuntimeError("disk went away")
            return real(rel)

        with mock.patch.object(catalog, "classify_path", flaky), \
                self.assertRaises(RuntimeError):
            catalog.scan(self.conn, self.root)
        self.assertFalse(self.conn.in_transaction)
        other = catalog.connect()
        self.addCleanup(other.close)
        other.execute("PRAGMA busy_timeout=0")
        other.execute("BEGIN IMMEDIATE")
        other.rollback()
        self.assertIsNone(self.row("A0003.MP4"))

    def test_symlinked_folder_is_skipped_with_a_warning(self):
        outside = self.tmp / "Elsewhere" / "Japan"
        touch_video(outside / "C0001.MP4")
        touch_video(self.root / "Sony SLOG-3" / "Peru" / "P0001.MP4")
        link = self.root / "Sony SLOG-3" / "Japan"
        link.symlink_to(outside, target_is_directory=True)
        with self.assertLogs("clipengine", level="WARNING") as logs:
            stats = catalog.scan(self.conn, self.root)
        self.assertEqual(stats["seen"], 1)
        self.assertTrue(any(str(link) in m and "symlink" in m
                            for m in logs.output), logs.output)

    def test_root_spellings_keep_ids_through_scan_itself(self):
        # the cli tests pin this through scan --root; here scan alone
        # gets relative, dot dot and tilde spellings of one tree
        tmp = self.tmp.resolve()
        root = tmp / "The Footage"
        touch_video(root / "Sony SLOG-3" / "Japan" / "a.mp4")
        touch_video(root / "Sony SLOG-3" / "Japan" / "b.mp4", 5000)
        rows = lambda: [tuple(r) for r in self.conn.execute(
            "SELECT id, path, missing FROM clips ORDER BY id")]
        catalog.scan(self.conn, root)
        before = rows()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(tmp)
        spellings = ["The Footage", f"{root}/Sony SLOG-3/../",
                     "./The Footage/./"]
        for spelling in spellings:
            with self.subTest(spelling=spelling):
                self.assertEqual(catalog.scan(self.conn, spelling)["new"], 0)
                self.assertEqual(rows(), before)
        with mock.patch.dict(os.environ, {"HOME": str(tmp)}):
            catalog.scan(self.conn, "~/The Footage")
        self.assertEqual(rows(), before)
        self.assertTrue(all(os.path.isabs(p) and os.path.normpath(p) == p
                            for _, p, _ in before))


class TestScanDetails(CatalogCase):
    # small pins on scan and lookup details no other test reaches

    def test_scan_holds_the_write_lock_while_it_reconciles(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "A0001.MP4")
        other = catalog.connect()
        self.addCleanup(other.close)
        other.execute("PRAGMA busy_timeout=0")
        real = catalog._reconcile
        outcome = []

        def spy(*args, **kwargs):
            # another writer must wait from the first read of known rows
            try:
                other.execute("BEGIN IMMEDIATE")
            except Exception as exc:
                outcome.append(type(exc).__name__)
            else:
                other.rollback()
                outcome.append("got the lock")
            return real(*args, **kwargs)

        with mock.patch.object(catalog, "_reconcile", spy):
            catalog.scan(self.conn, self.root)
        self.assertEqual(outcome, ["OperationalError"])

    def test_scan_commits_work_the_caller_left_open(self):
        touch_video(self.root / "Sony SLOG-3" / "Japan" / "A0001.MP4")
        catalog.scan(self.conn, self.root)
        # first_seen is a column scan never rewrites for a known clip
        self.conn.execute("UPDATE clips SET first_seen='kept'")
        self.assertTrue(self.conn.in_transaction)
        catalog.scan(self.conn, self.root)
        self.assertFalse(self.conn.in_transaction)
        other = catalog.connect()
        self.addCleanup(other.close)
        self.assertEqual(other.execute(
            "SELECT first_seen FROM clips").fetchone()[0], "kept")

    def test_country_filter_matches_full_case_folding(self):
        touch_video(self.root / "Sony SLOG-3" / "Stra\u00dfe" / "S0001.MP4")
        catalog.scan(self.conn, self.root)
        # casefold maps sharp s to ss; lower leaves it
        for typed in ("STRASSE", "strasse", "Stra\u00dfe"):
            self.assertEqual(
                len(catalog.pending(self.conn, country=typed)), 1, typed)

    def test_rescan_stores_an_old_nfd_name_as_nfc(self):
        name = "G\u00f6reme.MP4"
        nfd = unicodedata.normalize("NFD", name)
        folder = self.root / "Sony SLOG-3" / "Turkey"
        touch_video(folder / nfd)
        if os.listdir(folder) != [nfd]:
            self.skipTest("filesystem normalizes file names")
        catalog.scan(self.conn, self.root)
        # a row cataloged before names were normalized kept the nfd
        # bytes the disk holds
        self.conn.execute("UPDATE clips SET name=?", (nfd,))
        self.conn.commit()
        catalog.scan(self.conn, self.root)
        self.assertEqual([r["name"] for r in self.conn.execute(
            "SELECT name FROM clips")], [name])

    def test_backslash_in_a_name_matches_only_itself(self):
        japan = self.root / "Sony SLOG-3" / "Japan"
        touch_video(japan / "ab.MP4")
        touch_video(japan / "a\\b.MP4", 5000)
        catalog.scan(self.conn, self.root)
        self.assertEqual(catalog.find_clip(self.conn, "a\\b")["name"],
                         "a\\b.MP4")
        self.assertEqual(catalog.find_clip(self.conn, "ab")["name"],
                         "ab.MP4")

    def test_nfd_profile_folder_found_in_a_library_below_media_root(self):
        key = "Cam\u00e9ra Caf\u00e9"
        folder = unicodedata.normalize("NFD", key)
        library = config.MEDIA_ROOT / "Archive"
        touch_video(library / folder / "Japan" / "K0001.MP4")
        if os.listdir(library) != [folder]:
            self.skipTest("filesystem normalizes file names")
        profile = {key: {"key": "cafe_cam", "log": False}}
        with mock.patch.dict(config.CAMERA_PROFILES, profile):
            catalog.scan(self.conn, library)
        row = self.row("K0001.MP4")
        self.assertEqual((row["profile"], row["country"]),
                         ("cafe_cam", "Japan"))


class TestRecheckBeforeDecode(CatalogCase):
    # the analyze tests drive this through the cli; these pin the
    # catalog half alone: one fresh stat, no reads, eviction recorded
    def setUp(self):
        super().setUp()
        self.clip = self.root / "Sony SLOG-3" / "Japan" / "C0001.MP4"
        touch_video(self.clip, 50_000)
        catalog.scan(self.conn, self.root)
        self.pend = catalog.pending(self.conn)[0]

    def test_clip_unchanged_since_scan_is_fine(self):
        self.assertIsNone(catalog.recheck(self.conn, self.pend))

    def test_clip_evicted_since_scan_leaves_pending(self):
        evict(self.clip)
        if self.clip.stat().st_blocks != 0:
            self.skipTest("filesystem allocated blocks for a sparse file")
        why = catalog.recheck(self.conn, self.pend)
        self.assertIn("evicted", why)
        self.assertEqual(self.row("C0001.MP4")["available"], 0)
        self.assertEqual(catalog.pending(self.conn), [])
        self.assertEqual(catalog.overview(self.conn)["errors"], 0)

    def test_clip_deleted_or_changed_since_scan_says_run_scan(self):
        st = self.clip.stat()
        later = st.st_mtime_ns + 10**9
        os.utime(self.clip, ns=(later, later))
        self.assertIn("changed", catalog.recheck(self.conn, self.pend))
        self.clip.unlink()
        self.assertIn("gone", catalog.recheck(self.conn, self.pend))
        self.assertEqual(self.row("C0001.MP4")["missing"], 0)


if __name__ == "__main__":
    unittest.main()
