# test_fcpxml_timing.py
# fcp checks every edit against the sequence frame grid, so a timeline
# that mixes camera rates must still place each clip on whole sequence
# frames. and a file url has to be absolute, so a relative media root
# must never reach the exporter or the preview route as a relative path.

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from fractions import Fraction
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from clipengine import catalog, config, fcpxml, features
from clipengine.web import server
from tests import synth
from tests.test_matching import make_vec
from tests.util import TempDirsMixin

REPO = Path(__file__).resolve().parent.parent

# the rates the library actually shoots, with lengths that do not divide
# evenly into each other's frames
RATES = [23.976, 24.0, 25.0, 29.97, 30.0, 50.0, 59.94, 60.0]
LENGTHS = [10.01, 4.0, 3.1, 2.5, 1.9, 2.0, 151 / 59.94, 0.73]

SUMMARY = json.dumps({"duration_s": 1.0, "fps": 24.0, "width": 160,
                      "height": 120, "codec": "mp4v", "flat": False,
                      "start_class": "pan_right", "end_class": "pan_right",
                      "energy_profile": [0.1]})


def _t(value: str) -> Fraction:
    return Fraction(value.rstrip("s"))


def _meta(rates, lengths):
    return [{"id": i + 1, "name": f"clip_{i}_{fps}.mp4", "country": "X",
             "path": f"/x/clip_{i}_{fps}.mp4", "duration_s": dur,
             "fps": fps, "width": 3840, "height": 2160}
            for i, (fps, dur) in enumerate(zip(rates, lengths))]


def _parse(path: Path):
    root = ET.parse(path).getroot()
    frames = {f.get("id"): _t(f.get("frameDuration"))
              for f in root.iter("format")}
    assets = {a.get("id"): a for a in root.iter("asset")}
    seq = root.find(".//sequence")
    return root, frames, assets, frames[seq.get("format")]


class TestMixedRateTimeline(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def export(self, meta, stamp):
        return fcpxml.export_fcpxml(meta, "momentum",
                                    out_dir=self.tmp, stamp=stamp)

    def test_spine_clips_land_on_sequence_frames(self):
        # each rate takes a turn as the first clip, so as the sequence rate
        for k, first in enumerate(RATES):
            rates = RATES[k:] + RATES[:k]
            with self.subTest(sequence_fps=first):
                root, _, _, frame = _parse(
                    self.export(_meta(rates, LENGTHS), f"g{k}"))
                self.assertEqual(frame, Fraction(
                    *fcpxml.frame_duration(first)))
                offset = Fraction(0)
                for clip in root.findall(".//spine/asset-clip"):
                    dur = _t(clip.get("duration"))
                    self.assertEqual(
                        (offset / frame).denominator, 1,
                        f"{clip.get('name')} starts at "
                        f"{float(offset / frame):.4f} sequence frames")
                    self.assertEqual(
                        (dur / frame).denominator, 1,
                        f"{clip.get('name')} spans "
                        f"{float(dur / frame):.4f} sequence frames")
                    for attr in ("offset", "start"):
                        if clip.get(attr) is not None:
                            self.assertEqual(
                                (_t(clip.get(attr)) / frame).denominator, 1)
                    offset += dur

    def test_each_clip_keeps_its_own_length_to_the_frame(self):
        for k, first in enumerate(RATES):
            rates = RATES[k:] + RATES[:k]
            with self.subTest(sequence_fps=first):
                root, frames, assets, frame = _parse(
                    self.export(_meta(rates, LENGTHS), f"l{k}"))
                for clip in root.findall(".//spine/asset-clip"):
                    asset = assets[clip.get("ref")]
                    own = _t(asset.get("duration"))
                    # the asset stays in whole frames of its own rate
                    self.assertEqual(
                        (own / frames[asset.get("format")]).denominator, 1)
                    # the cut never runs past the last media frame and
                    # loses less than one sequence frame of it
                    dur = _t(clip.get("duration"))
                    self.assertLessEqual(dur, own, clip.get("name"))
                    self.assertLess(own - dur, frame, clip.get("name"))

    def test_single_rate_timeline_keeps_asset_lengths(self):
        meta = _meta([25.0] * 3, [4.0, 3.1, 0.73])
        root, _, assets, _ = _parse(self.export(meta, "s"))
        for clip in root.findall(".//spine/asset-clip"):
            self.assertEqual(clip.get("duration"),
                             assets[clip.get("ref")].get("duration"))

    def test_tiny_clip_still_gets_one_sequence_frame(self):
        meta = _meta([23.976, 60.0], [2.0, 1 / 60])
        root, _, _, frame = _parse(self.export(meta, "tiny"))
        clips = root.findall(".//spine/asset-clip")
        self.assertEqual(_t(clips[1].get("duration")), frame)


class TestRelativeMediaRoot(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.work = self.tmp / "work"
        clip_dir = self.work / "footage" / "Sony SLOG-3" / "Japan"
        clip_dir.mkdir(parents=True)
        for name in ("a.mp4", "b.mp4"):
            synth.write_clip(clip_dir / name,
                             synth.frames_for("pan_right", n=24))
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(self.work)

    def config_root(self, value: str) -> Path:
        """MEDIA_ROOT as a fresh process computes it from the env var"""
        env = dict(os.environ, CLIPENGINE_MEDIA_ROOT=value,
                   PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE="1")
        out = subprocess.run(
            [sys.executable, "-c",
             "from clipengine import config; print(config.MEDIA_ROOT)"],
            cwd=self.work, env=env, capture_output=True, text=True,
            check=True)
        return Path(out.stdout.strip())

    def test_relative_env_root_becomes_absolute(self):
        root = self.config_root("./footage")
        self.assertTrue(root.is_absolute(), str(root))
        self.assertEqual(root.resolve(), (self.work / "footage").resolve())

    def test_relative_clip_path_exports_absolute_file_url(self):
        meta = [{"id": 1, "name": "a.mp4", "country": "Japan",
                 "path": "footage/Sony SLOG-3/Japan/a.mp4",
                 "duration_s": 1.0, "fps": 24.0,
                 "width": 160, "height": 120}]
        try:
            path = fcpxml.export_fcpxml(meta, "momentum", stamp="rel")
        except ValueError as exc:
            self.fail(f"export_fcpxml raised {exc!r}")
        src = ET.parse(path).getroot().find(".//media-rep").get("src")
        self.assertEqual(
            src, (Path.cwd() / "footage/Sony SLOG-3/Japan/a.mp4").as_uri())

    def test_relative_env_root_serves_and_exports_from_any_cwd(self):
        root = self.config_root("footage")
        with mock.patch.object(config, "MEDIA_ROOT", root):
            conn = catalog.connect()
            catalog.scan(conn)
            ids = []
            for r in conn.execute("SELECT id, content_key FROM clips"
                                  " ORDER BY id"):
                vec = make_vec(end_flow_x=-0.5, end_energy=0.5,
                               start_flow_x=-0.5, start_energy=0.5)
                catalog.save_features(conn, r["id"], r["content_key"],
                                      features.to_bytes(vec), SUMMARY)
                ids.append(r["id"])
            conn.close()
        self.assertEqual(len(ids), 2)

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, args=(0.02,),
                         daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)

        # the ui gets started from some other directory
        os.chdir("/")
        size = (self.work / "footage" / "Sony SLOG-3" / "Japan"
                / "a.mp4").stat().st_size
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/media/{ids[0]}",
                    timeout=10) as r:
                self.assertEqual(r.status, 200)
                self.assertEqual(len(r.read()), size)
        except urllib.error.HTTPError as exc:
            self.fail(f"/media returned {exc.code} {exc.read()!r}")

        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/export",
            data=json.dumps({"ids": ids, "mode": "momentum"}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as exc:
            self.fail(f"/api/export returned {exc.code} {exc.read()!r}")
        srcs = [rep.get("src") for rep in
                ET.parse(data["fcpxml"]).getroot().iter("media-rep")]
        self.assertEqual(len(srcs), 2)
        for src in srcs:
            self.assertTrue(src.startswith("file:///"), src)
            local = urllib.request.url2pathname(src[len("file://"):])
            self.assertTrue(Path(local).exists(), src)


if __name__ == "__main__":
    unittest.main()
