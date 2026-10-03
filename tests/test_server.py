# test_server.py
# the web layer against a live server on an ephemeral port: json routes,
# byte range streaming (the thing that makes <video> scrubbing work),
# thumbnail serving, and the refusal of clips evicted to icloud.

import json
import threading
import types
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from clipengine import catalog, config, features
from clipengine.web import server
from tests import synth
from tests.test_matching import make_vec
from tests.util import TempDirsMixin

SUMMARY = json.dumps({"duration_s": 1.25, "fps": 24.0, "width": 160,
                      "height": 120, "codec": "mp4v", "flat": False,
                      "start_class": "pan_right", "end_class": "pan_right",
                      "energy_profile": [0.1, 0.2, 0.15]})


class ServerBase(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        root = self.tmp / "The Footage"
        self.clip_a = root / "Sony SLOG-3" / "Japan" / "a.mp4"
        self.clip_b = root / "Sony SLOG-3" / "Japan" / "b.mp4"
        self.clip_a.parent.mkdir(parents=True)
        synth.write_clip(self.clip_a, synth.frames_for("pan_right", n=30))
        synth.write_clip(self.clip_b, synth.frames_for("pan_right", n=30))
        conn = catalog.connect()
        catalog.scan(conn, root)
        self.ids = {}
        for row in conn.execute("SELECT id, name, content_key FROM clips"):
            vec = make_vec(end_flow_x=-0.5, end_energy=0.5,
                           start_flow_x=-0.5, start_energy=0.5)
            catalog.save_features(conn, row["id"], row["content_key"],
                                  features.to_bytes(vec), SUMMARY)
            self.ids[row["name"]] = row["id"]
            for pos in ("start", "mid", "end"):
                (config.THUMB_DIR / f"{row['id']}_{pos}.jpg"
                 ).write_bytes(b"\xff\xd8\xff fake jpeg body")
        conn.close()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        # a short poll keeps shutdown fast; the default waits half a second
        threading.Thread(target=self.httpd.serve_forever, args=(0.02,),
                         daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def get(self, path, headers=None):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", headers=headers or {})
        return urllib.request.urlopen(req, timeout=10)

    def get_json(self, path):
        with self.get(path) as r:
            return json.loads(r.read())

    def post_json(self, path, obj):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())


class TestRoutes(ServerBase):
    def test_index_serves_ui(self):
        with self.get("/") as r:
            self.assertEqual(r.status, 200)
            self.assertIn(b"CLIPENGINE", r.read())

    def test_overview(self):
        ov = self.get_json("/api/overview")
        self.assertEqual(ov["totals"]["total"], 2)
        self.assertEqual(ov["totals"]["analyzed"], 2)
        self.assertEqual(set(ov["modes"]),
                         {"momentum", "whip", "calm", "contrast"})

    def test_clips_listing(self):
        data = self.get_json("/api/clips")
        self.assertEqual(len(data["clips"]), 2)
        names = {c["name"] for c in data["clips"]}
        self.assertEqual(names, {"a.mp4", "b.mp4"})

    def test_clip_detail(self):
        cid = self.ids["a.mp4"]
        d = self.get_json(f"/api/clip/{cid}")
        self.assertEqual(d["country"], "Japan")
        self.assertEqual(d["summary"]["end_class"], "pan_right")

    def test_matches(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        data = self.get_json(f"/api/matches?clip={a}&mode=momentum&n=5")
        self.assertEqual([m["id"] for m in data["matches"]], [b])
        self.assertGreater(data["matches"][0]["breakdown"]["motion"], 0.9)

    def test_sequence_route(self):
        a = self.ids["a.mp4"]
        data = self.get_json(f"/api/sequence?seed={a}&length=2&mode=momentum")
        self.assertEqual(len(data["clips"]), 2)
        self.assertEqual(data["clips"][0]["id"], a)

    def test_export_roundtrip(self):
        ids = [self.ids["a.mp4"], self.ids["b.mp4"]]
        data = self.post_json("/api/export", {"ids": ids, "mode": "momentum"})
        json_path, m3u_path = Path(data["json"]), Path(data["m3u8"])
        self.assertTrue(json_path.exists())
        self.assertTrue(m3u_path.exists())
        self.assertTrue(Path(data["fcpxml"]).exists())
        plan = json.loads(json_path.read_text())
        self.assertEqual([c["id"] for c in plan["clips"]], ids)
        self.assertIn(str(self.clip_a), m3u_path.read_text())

    def test_thumb(self):
        cid = self.ids["a.mp4"]
        with self.get(f"/thumb/{cid}/start.jpg") as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers["Content-Type"], "image/jpeg")

    def test_unknown_routes_404(self):
        for path in ("/nope", "/api/clip/9999", "/thumb/9999/start.jpg"):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(path)
            self.assertEqual(ctx.exception.code, 404)


class TestMediaStreaming(ServerBase):
    def test_full_body(self):
        cid = self.ids["a.mp4"]
        size = self.clip_a.stat().st_size
        with self.get(f"/media/{cid}") as r:
            self.assertEqual(r.status, 200)
            self.assertEqual(r.headers["Accept-Ranges"], "bytes")
            self.assertEqual(len(r.read()), size)

    def test_range_start(self):
        cid = self.ids["a.mp4"]
        size = self.clip_a.stat().st_size
        with self.get(f"/media/{cid}", {"Range": "bytes=0-99"}) as r:
            self.assertEqual(r.status, 206)
            body = r.read()
        self.assertEqual(len(body), 100)
        with open(self.clip_a, "rb") as f:
            self.assertEqual(body, f.read(100))

    def test_range_suffix(self):
        cid = self.ids["a.mp4"]
        size = self.clip_a.stat().st_size
        with self.get(f"/media/{cid}", {"Range": "bytes=-50"}) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(
                r.headers["Content-Range"],
                f"bytes {size - 50}-{size - 1}/{size}")
            self.assertEqual(len(r.read()), 50)

    def test_range_beyond_eof_416(self):
        cid = self.ids["a.mp4"]
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get(f"/media/{cid}", {"Range": "bytes=999999999-"})
        self.assertEqual(ctx.exception.code, 416)

    def test_evicted_clip_refused_not_downloaded(self):
        cid = self.ids["a.mp4"]
        fake = types.SimpleNamespace(st_blocks=0, st_size=12345)
        real_stat = server.os.stat

        def selective(path, *args, **kwargs):
            # fake an evicted stub only for the clip under test; a global
            # fake would poison sqlite's own path checks in the handler
            if str(path) == str(self.clip_a):
                return fake
            return real_stat(path, *args, **kwargs)

        with mock.patch.object(server.os, "stat", side_effect=selective):
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                self.get(f"/media/{cid}")
        self.assertEqual(ctx.exception.code, 409)


if __name__ == "__main__":
    unittest.main()
