# test_fcpxml_export.py
# how the three export files get written: exact rates for unlisted
# ntsc and fractional fps, utf8 whatever the locale, names that cannot
# break the xml or the playlist, and writes that never leave a half
# finished file under a real export name.

import json
import os
import re
import shutil
import resource
import signal
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from unittest import mock

import clipengine
from clipengine import config, fcpxml, sequence
from tests.util import TempDirsMixin

REPO = Path(clipengine.__file__).resolve().parents[1]


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 3, 9, 30, 0)


def row(i, name=None, path=None, fps=24.0, duration_s=2.0):
    name = name if name is not None else f"c{i}.mp4"
    return {"id": i, "name": name, "country": "X",
            "path": path if path is not None else f"/x/{name}",
            "duration_s": duration_s, "fps": fps,
            "width": 160, "height": 120}


@contextmanager
def file_size_limit(nbytes):
    """writes past nbytes fail with EFBIG, like a disk that fills up"""
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    old = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (nbytes, hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, old)


class TestUnlistedRates(TempDirsMixin, unittest.TestCase):
    def test_unlisted_ntsc_and_fractional_rates(self):
        self.assertEqual(fcpxml.frame_duration(47.952), (1001, 48000))
        self.assertEqual(fcpxml.frame_duration(239.76), (1001, 240000))
        num, den = fcpxml.frame_duration(12.5)
        self.assertEqual(Fraction(den, num), Fraction(25, 2))
        self.assertEqual(fcpxml.frame_duration(48.0), (100, 4800))

    def test_600_frames_at_47_952_export_as_600_frames(self):
        # fps and duration rounded the way analysis stores them
        meta = [row(1, fps=round(48000 / 1001, 3),
                    duration_s=round(600 * 1001 / 48000, 3))]
        path = fcpxml.export_fcpxml(meta, "momentum", out_dir=self.tmp,
                                    stamp="t")
        root = ET.parse(path).getroot()
        fd = Fraction(root.find("./resources/format")
                      .get("frameDuration")[:-1])
        ad = Fraction(root.find("./resources/asset").get("duration")[:-1])
        self.assertEqual(fd, Fraction(1001, 48000))
        self.assertEqual(ad / fd, 600)

    def test_odd_rates_still_land_on_whole_frames(self):
        for fps in (12.5, 7.3, 0.5, 1.0, 100.0):
            with self.subTest(fps=fps):
                num, den = fcpxml.frame_duration(fps)
                self.assertIsInstance(num, int)
                self.assertIsInstance(den, int)
                self.assertAlmostEqual(den / num, fps, places=2)

    def test_jittery_measured_rates_snap_to_the_nearest_standard(self):
        self.assertEqual(fcpxml.frame_duration(29.96), (1001, 30000))
        self.assertEqual(fcpxml.frame_duration(24.02), (100, 2400))
        self.assertEqual(fcpxml.frame_duration(119.9), (1001, 120000))

    def test_measured_pal_rates_stay_whole(self):
        # no camera shoots 25000/1001, so a jittery 24.98 is still 25
        for fps, want in ((24.98, 2500), (49.96, 5000), (49.97, 5000),
                          (99.92, 10000), (14.99, 1500)):
            with self.subTest(fps=fps):
                self.assertEqual(fcpxml.frame_duration(fps), (100, want))

    def test_slow_rates_keep_their_own_fraction(self):
        # a 1.05 fps timelapse is 5 percent off 1 fps, not jitter
        num, den = fcpxml.frame_duration(1.05)
        self.assertEqual(Fraction(den, num), Fraction(21, 20))

    def test_unusable_duration_exports_one_second(self):
        for dur in (float("nan"), float("inf")):
            with self.subTest(duration_s=dur):
                path = fcpxml.export_fcpxml([row(1, duration_s=dur)],
                                            "calm", out_dir=self.tmp)
                asset = ET.parse(path).getroot().find("./resources/asset")
                self.assertEqual(asset.get("duration"), "2400/2400s")

    def test_unusable_duration_plays_as_unknown_in_m3u8(self):
        rows = [row(1, duration_s=float("nan")),
                row(2, duration_s=float("inf"))]
        _, m3u, _ = sequence.export_chain(rows, [0.5], "calm")
        lines = m3u.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[1], "#EXTINF:0,c1.mp4")
        self.assertEqual(lines[3], "#EXTINF:0,c2.mp4")

    def test_unusable_fps_falls_back_to_24(self):
        for fps in (None, -5.0, 0.004, float("nan"), float("inf")):
            with self.subTest(fps=fps):
                self.assertEqual(fcpxml.frame_duration(fps), (100, 2400))


# the child repoints config into the temp dir before it imports anything
# that writes, then reports its locale encoding so a utf8 host can skip.
# names travel as ascii json so the latin1 argv decoding cannot mangle them
CHILD = r'''
import json, locale, sys
from pathlib import Path
from clipengine import config
out = Path(sys.argv[1])
config.DATA_DIR = out / "data"
config.DB_PATH = config.DATA_DIR / "catalog.db"
config.THUMB_DIR = config.DATA_DIR / "thumbs"
config.EXPORT_DIR = out / "exports"
config.MEDIA_ROOT = out
from clipengine import sequence
print(locale.getpreferredencoding(False))
meta = [{"id": i, "name": n, "country": "X", "path": "/m/" + n,
         "duration_s": 1.0, "fps": 24.0, "width": 160, "height": 120}
        for i, n in enumerate(json.loads(sys.argv[2]), 1)]
sequence.export_chain(meta, [0.5] * (len(meta) - 1), "momentum")
'''


class TestLocale(unittest.TestCase):
    def test_exports_are_utf8_under_latin1_locale(self):
        names = ["caf\u00e9.mov", "Ky\u014dto \u65e5\u672c.mov"]
        with tempfile.TemporaryDirectory() as d:
            tmp = Path(d)
            env = {k: v for k, v in os.environ.items()
                   if k not in ("PYTHONUTF8", "PYTHONIOENCODING")
                   and not k.startswith("LC_")}
            env.update(LC_ALL="en_US.ISO8859-1", LANG="en_US.ISO8859-1",
                       CLIPENGINE_MEDIA_ROOT=str(tmp),
                       PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
            proc = subprocess.run(
                [sys.executable, "-X", "utf8=0", "-c", CHILD, str(tmp),
                 json.dumps(names)],
                cwd=tmp, env=env, capture_output=True,
                encoding="iso8859-1", timeout=60)
            if proc.stdout.strip().lower().replace("-", "") in ("utf8", ""):
                if proc.returncode == 0:
                    self.skipTest("no latin1 locale on this host")
            self.assertEqual(proc.returncode, 0, proc.stderr[-400:])
            out = tmp / "exports"
            self.assertEqual(sorted(p.suffix for p in out.iterdir()),
                             [".fcpxml", ".json", ".m3u8"])
            xml_path = next(out.glob("*.fcpxml"))
            got = [c.get("name") for c in
                   ET.parse(xml_path).getroot().iter("asset-clip")]
            self.assertEqual(got, names)
            m3u = next(out.glob("*.m3u8")).read_bytes().decode("utf-8")
            for n in names:
                self.assertIn(n, m3u)


class TestUnsafeNames(TempDirsMixin, unittest.TestCase):
    def test_control_chars_in_names(self):
        rows = [row(1, "esc\x1bctl.mp4"), row(2, "plain.mp4")]
        _, _, xml_path = sequence.export_chain(rows, [0.5], "calm")
        root = ET.parse(xml_path).getroot()
        names = [c.get("name") for c in root.iter("asset-clip")]
        self.assertEqual(names, ["escctl.mp4", "plain.mp4"])
        srcs = [r.get("src") for r in root.iter("media-rep")]
        self.assertEqual(srcs[0], "file:///x/esc%1Bctl.mp4")

    def test_newlines_in_names_keep_one_m3u8_entry_per_clip(self):
        rows = [row(1, "line1\nline2.mp4"), row(2, "cr\rret.mp4"),
                row(3, "plain.mp4")]
        _, m3u, _ = sequence.export_chain(rows, [0.5, 0.5], "calm")
        raw = m3u.read_text(encoding="utf-8")
        self.assertNotIn("\r", raw)
        lines = raw.split("\n")
        self.assertEqual(lines[-1], "")
        self.assertEqual(lines[:-1], [
            "#EXTM3U",
            "#EXTINF:2.0,line1 line2.mp4", "file:///x/line1%0Aline2.mp4",
            "#EXTINF:2.0,cr ret.mp4", "file:///x/cr%0Dret.mp4",
            "#EXTINF:2.0,plain.mp4", "/x/plain.mp4"])

    def test_undecodable_filename_bytes_still_export(self):
        # a name that is not valid utf8 reaches python as surrogates
        bad = os.fsdecode(b"caf\xe9.mp4")
        rows = [row(1, bad), row(2)]
        paths = sequence.export_chain(rows, [0.5], "calm")
        root = ET.parse(paths[2]).getroot()
        srcs = [r.get("src") for r in root.iter("media-rep")]
        self.assertEqual(srcs[0], "file:///x/caf%E9.mp4")
        lines = paths[1].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 5, lines)
        self.assertEqual(lines[2], "file:///x/caf%E9.mp4")


class TestDirectExport(TempDirsMixin, unittest.TestCase):
    def test_direct_exports_in_one_second_never_overwrite(self):
        stale = config.EXPORT_DIR / "sequence_20261003_093000.fcpxml"
        stale.write_text("keep me")
        with mock.patch.object(fcpxml, "datetime", FixedClock):
            first = fcpxml.export_fcpxml([row(1)], "calm")
            second = fcpxml.export_fcpxml([row(2)], "calm")
        self.assertEqual(stale.read_text(), "keep me")
        self.assertEqual(len({stale, first, second}), 3)
        for path, want in ((first, "c1.mp4"), (second, "c2.mp4")):
            clip = ET.parse(path).getroot().find(".//spine/asset-clip")
            self.assertEqual(clip.get("name"), want)

    def test_direct_export_skips_a_stem_an_export_chain_holds(self):
        with mock.patch.object(sequence, "datetime", FixedClock):
            chain = sequence.export_chain([row(1), row(2)], [0.5], "calm")
        with mock.patch.object(fcpxml, "datetime", FixedClock):
            alone = fcpxml.export_fcpxml([row(3)], "calm")
        self.assertNotIn(alone, chain)
        self.assertNotEqual(alone.stem, chain[0].stem)

    def test_leftover_json_or_m3u8_pushes_the_name_on(self):
        # an empty json was how the old scheme claimed a name
        stem = config.EXPORT_DIR / "sequence_20261003_093000"
        for ext in (".json", ".m3u8"):
            with self.subTest(leftover=ext):
                leftover = stem.with_suffix(ext)
                leftover.write_text("")
                with mock.patch.object(fcpxml, "datetime", FixedClock):
                    alone = fcpxml.export_fcpxml([row(1)], "calm")
                with mock.patch.object(sequence, "datetime", FixedClock):
                    chain = sequence.export_chain([row(1), row(2)], [0.5],
                                                  "calm")
                self.assertEqual(alone.name,
                                 "sequence_20261003_093000_2.fcpxml")
                self.assertEqual({p.stem for p in chain},
                                 {"sequence_20261003_093000_3"})
                self.assertEqual(leftover.read_text(), "")
                for p in config.EXPORT_DIR.iterdir():
                    p.unlink()

    def test_timeline_that_dies_half_written_keeps_the_old_file(self):
        meta = [row(i) for i in range(1, 30)]
        good = fcpxml.export_fcpxml(meta[:1], "calm", stamp="t")
        before = good.read_bytes()
        self.assertLess(len(before), 4096)
        with file_size_limit(4096):
            with self.assertRaises(OSError):
                fcpxml.export_fcpxml(meta, "calm", stamp="t")
        self.assertEqual(good.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in config.EXPORT_DIR.iterdir()),
                         [good.name])

    def test_unnamed_timeline_that_dies_leaves_nothing(self):
        meta = [row(i) for i in range(1, 30)]
        with file_size_limit(4096):
            with self.assertRaises(OSError):
                fcpxml.export_fcpxml(meta, "calm")
        self.assertEqual(list(config.EXPORT_DIR.iterdir()), [])

    def test_chain_that_dies_mid_write_leaves_nothing(self):
        meta = [row(i) for i in range(1, 30)]
        with file_size_limit(4096):
            with self.assertRaises(OSError):
                sequence.export_chain(meta, [0.5] * 28, "calm")
        self.assertEqual(list(config.EXPORT_DIR.iterdir()), [])


def parse(path):
    """root, formats by id, assets by id, spine clips"""
    root = ET.parse(path).getroot()
    formats = {f.get("id"): f for f in root.iter("format")}
    assets = {a.get("id"): a for a in root.iter("asset")}
    return root, formats, assets, root.findall(".//spine/asset-clip")


def secs(t):
    return Fraction(t[:-1])


class TestTimelineBasics(TempDirsMixin, unittest.TestCase):
    def export(self, meta, stamp="t", **kw):
        kw.setdefault("out_dir", self.tmp)
        return fcpxml.export_fcpxml(meta, "momentum", stamp=stamp, **kw)

    def test_frame_duration_full_table(self):
        exact = {24: (100, 2400), 50: (100, 5000), 60: (100, 6000),
                 119.88: (1001, 120000), 120: (100, 12000)}
        snapped = {23.98: (1001, 24000), 29.92: (1001, 30000),
                   30.02: (100, 3000)}
        for fps, want in {**exact, **snapped}.items():
            with self.subTest(fps=fps):
                self.assertEqual(fcpxml.frame_duration(fps), want)

    def test_ntsc_durations_frame_exact(self):
        cases = [(23.976, 10.01, 240), (29.97, 3.003, 90),
                 (59.94, 2.002, 120), (59.94, 7200.0, 431568)]
        for k, (fps, dur, frames) in enumerate(cases):
            with self.subTest(fps=fps, duration=dur):
                meta = [row(1, fps=fps, duration_s=dur)]
                _, formats, assets, clips = parse(self.export(meta, f"n{k}"))
                fd = secs(formats["r1"].get("frameDuration"))
                own = secs(assets["a1"].get("duration"))
                self.assertEqual(own, frames * fd)
                self.assertEqual(clips[0].get("duration"),
                                 assets["a1"].get("duration"))

    def test_formats_deduplicated_and_sequence_uses_first(self):
        meta = [row(1, fps=25.0), row(2, fps=23.976), row(3, fps=25.0),
                row(4, fps=23.976)]
        root, formats, assets, _ = parse(self.export(meta))
        self.assertEqual(len(formats), 2)
        refs = [assets[f"a{i}"].get("format") for i in range(1, 5)]
        self.assertEqual(refs, ["r1", "r2", "r1", "r2"])
        self.assertEqual(formats["r1"].get("frameDuration"), "100/2500s")
        self.assertEqual(root.find(".//sequence").get("format"), "r1")

    def test_file_url_and_name_escaping(self):
        meta = [row(1, name="tab\there.mp4",
                    path="/x/The Footage/a b#c?d%e&f+g;h.mp4"),
                row(2, name="two\nlines <&> \"q\".mp4",
                    path="/x/caf\u00e9 \u65e5.mp4")]
        _, _, assets, clips = parse(self.export(meta))
        srcs = [assets[f"a{i}"].find("media-rep").get("src")
                for i in (1, 2)]
        self.assertEqual(srcs, [
            "file:///x/The%20Footage/a%20b%23c%3Fd%25e%26f%2Bg%3Bh.mp4",
            "file:///x/caf%C3%A9%20%E6%97%A5.mp4"])
        want = [m["name"] for m in meta]
        self.assertEqual([c.get("name") for c in clips], want)
        self.assertEqual([assets[f"a{i}"].get("name") for i in (1, 2)],
                         want)

    def test_project_event_and_output_path(self):
        out = self.tmp / "deep" / "er"
        path = fcpxml.export_fcpxml([row(1), row(2)], "calm", out_dir=out,
                                    stamp="20260101_000000")
        self.assertEqual(path, out / "sequence_20260101_000000.fcpxml")
        root, _, _, clips = parse(path)
        self.assertEqual(root.get("version"), "1.9")
        event = root.find("./library/event")
        self.assertEqual(event.get("name"), "ClipEngine")
        self.assertEqual(event.find("project").get("name"),
                         "clipengine calm 20260101_000000")
        self.assertEqual(len(clips), 2)
        for clip in clips:
            self.assertIsNone(clip.get("offset"))
        self.assertTrue(path.read_text(encoding="utf-8").startswith(
            '<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>'))

    def test_default_out_dir_is_config_export_dir(self):
        name = re.compile(r"sequence_\d{8}_\d{6}(_\d+)?\.fcpxml")
        first = fcpxml.export_fcpxml([row(1)], "calm")
        second = fcpxml.export_fcpxml([row(2)], "calm")
        for path in (first, second):
            self.assertEqual(path.parent, config.EXPORT_DIR)
            self.assertRegex(path.name, name)
        chain = sequence.export_chain([row(1), row(2)], [0.5], "calm")
        for path in chain:
            self.assertEqual(path.parent, config.EXPORT_DIR)
            self.assertRegex(path.with_suffix(".fcpxml").name, name)

    def test_export_validates_against_fcp_dtd(self):
        dtd = Path("/Applications/Final Cut Pro.app/Contents/Frameworks/"
                   "Interchange.framework/Versions/A/Resources/"
                   "FCPXMLv1_9.dtd")
        xmllint = shutil.which("xmllint")
        if not dtd.is_file() or not xmllint:
            self.skipTest("needs final cut pro and xmllint")
        # xmllint chokes on a dtd path with spaces, so copy it
        local = self.tmp / "fcpxml19.dtd"
        shutil.copyfile(dtd, local)
        meta = [row(1, name="a & b.mp4", path="/x/The Footage/a & b.mp4",
                    fps=23.976, duration_s=10.01),
                row(2, fps=25.0, duration_s=3.0),
                row(3, fps=None, duration_s=None)]
        path = self.export(meta)
        proc = subprocess.run([xmllint, "--noout", "--dtdvalid", str(local),
                               str(path)], capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])

    def test_m3u8_unknown_duration(self):
        meta = [row(1, duration_s=None), row(2, duration_s=0)]
        _, m3u, _ = sequence.export_chain(meta, [0.5], "calm")
        lines = m3u.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines, ["#EXTM3U", "#EXTINF:0,c1.mp4",
                                 "/x/c1.mp4", "#EXTINF:0,c2.mp4",
                                 "/x/c2.mp4"])

    def test_chain_with_relative_path_writes_absolute_urls(self):
        work = self.tmp / "work"
        work.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(work)
        meta = [row(1, path="footage/a b.mp4"), row(2)]
        paths = sequence.export_chain(meta, [0.5], "calm")
        _, _, assets, _ = parse(paths[2])
        self.assertEqual(assets["a1"].find("media-rep").get("src"),
                         (Path.cwd() / "footage/a b.mp4").as_uri())
        self.assertEqual(len(list(config.EXPORT_DIR.iterdir())), 3)


    def test_relative_paths_play_from_the_exports_folder(self):
        # a player reads relative m3u8 entries against the playlist folder,
        # and a line that starts with # is a comment
        work = self.tmp / "work"
        work.mkdir()
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(work)
        meta = [row(1, path="footage/a.mp4"), row(2, path="#2.mp4")]
        _, m3u, _ = sequence.export_chain(meta, [0.5], "calm")
        lines = m3u.read_text(encoding="utf-8").splitlines()
        self.assertEqual(lines[2], str(Path.cwd() / "footage/a.mp4"))
        self.assertEqual(lines[4], str(Path.cwd() / "#2.mp4"))

if __name__ == "__main__":
    unittest.main()
