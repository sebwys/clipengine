# test_server_web.py
# the web layer as a browser and a hostile page see it: which hosts and
# origins get answered, head and range edge cases, and a clip that cannot
# be opened, bad input, search, strict json and thumbnail revalidation.
# each test runs a live server on an ephemeral loopback port.

import contextlib
import email.utils
import io
import json
import os
import socket
import tempfile
import threading
import unicodedata
import unittest
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
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


class WebBase(TempDirsMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._clipdir = tempfile.TemporaryDirectory(prefix="clipengine-web-")
        cls.root = Path(cls._clipdir.name) / "footage"
        folder = cls.root / "Sony SLOG-3" / "Japan"
        folder.mkdir(parents=True)
        for name in ("a.mp4", "b.mp4"):
            synth.write_clip(folder / name,
                             synth.frames_for("pan_right", n=24))
        cls.clip_a = folder / "a.mp4"

    @classmethod
    def tearDownClass(cls):
        cls._clipdir.cleanup()
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        conn = catalog.connect()
        catalog.scan(conn, self.root)
        self.ids = {}
        for row in conn.execute("SELECT id, name, content_key FROM clips"):
            vec = make_vec(end_flow_x=-0.5, end_energy=0.5,
                           start_flow_x=-0.5, start_energy=0.5)
            catalog.save_features(conn, row["id"], row["content_key"],
                                  features.to_bytes(vec), SUMMARY)
            self.ids[row["name"]] = row["id"]
        conn.close()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        # a short poll keeps shutdown fast; the default waits half a second
        threading.Thread(target=self.httpd.serve_forever, args=(0.02,),
                         daemon=True).start()
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)

    def raw(self, method, path, headers=(), body=b"", version="1.1"):
        """send exactly these header lines and return the whole reply as
        (status, header dict, body, raw bytes). host is added only when
        the caller names it, so a missing host can be tested too."""
        lines = [f"{method} {path} HTTP/{version}"]
        lines += [f"{k}: {v}" for k, v in headers]
        if body:
            lines.append(f"Content-Length: {len(body)}")
        lines.append("Connection: close")
        req = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
        with socket.create_connection(("127.0.0.1", self.port),
                                      timeout=10) as s:
            s.sendall(req)
            buf = b""
            while chunk := s.recv(65536):
                buf += chunk
        head, _, rest = buf.partition(b"\r\n\r\n")
        status_line, *hlines = head.decode("latin-1").split("\r\n")
        hdrs = {}
        for line in hlines:
            k, _, v = line.partition(":")
            hdrs[k.strip().lower()] = v.strip()
        return int(status_line.split()[1]), hdrs, rest, buf

    def host(self, name="127.0.0.1"):
        return ("Host", f"{name}:{self.port}")

    def export_body(self):
        return json.dumps({"ids": [self.ids["a.mp4"], self.ids["b.mp4"]],
                           "mode": config.DEFAULT_MODE}).encode()

    def exported(self):
        return sorted(os.listdir(config.EXPORT_DIR))


class TestHostAndOrigin(WebBase):
    def test_loopback_hosts_are_served(self):
        for name in ("127.0.0.1", "localhost", "LocalHost"):
            with self.subTest(host=name):
                status, _, _, _ = self.raw("GET", "/api/overview",
                                           [self.host(name)])
                self.assertEqual(status, 200)

    def test_request_without_host_is_served(self):
        # raw http/1.0 clients send no host at all
        status, _, body, _ = self.raw("GET", "/api/overview", version="1.0")
        self.assertEqual(status, 200)
        self.assertIn("totals", json.loads(body))

    def test_page_export_post_still_works(self):
        # the shape fetch sends from the page itself, from either name
        for name in ("127.0.0.1", "localhost"):
            with self.subTest(origin=name):
                status, _, body, _ = self.raw(
                    "POST", "/api/export",
                    [self.host(name),
                     ("Origin", f"http://{name}:{self.port}"),
                     ("Content-Type", "application/json")],
                    self.export_body())
                self.assertEqual(status, 200, body)
                self.assertIn("fcpxml", json.loads(body))

    def test_json_post_with_charset_and_no_origin_works(self):
        status, _, body, _ = self.raw(
            "POST", "/api/export",
            [self.host(), ("Content-Type", "application/json; charset=utf-8")],
            self.export_body())
        self.assertEqual(status, 200, body)

    def test_foreign_host_and_cross_site_post_refused(self):
        for path in ("/", "/api/overview", f"/media/{self.ids['a.mp4']}"):
            with self.subTest(path=path):
                status, _, _, _ = self.raw(
                    "GET", path, [("Host", f"evil.example:{self.port}")])
                self.assertIn(status, (421, 403))
        with self.subTest(host="wrong port"):
            status, _, _, _ = self.raw("GET", "/api/overview",
                                       [("Host", "127.0.0.1:1")])
            self.assertIn(status, (421, 403))
        cases = [
            ("text/plain", "http://evil.example"),
            ("application/json", "http://evil.example"),
            ("text/plain", None),
            ("application/x-www-form-urlencoded", None),
        ]
        for ctype, origin in cases:
            with self.subTest(ctype=ctype, origin=origin):
                hdrs = [self.host(), ("Content-Type", ctype)]
                if origin:
                    hdrs.append(("Origin", origin))
                status, _, _, _ = self.raw("POST", "/api/export", hdrs,
                                           self.export_body())
                self.assertIn(status, (403, 415))
                self.assertEqual(self.exported(), [])


class TestHeadAndRange(WebBase):
    def setUp(self):
        super().setUp()
        self.media = f"/media/{self.ids['a.mp4']}"
        self.size = self.clip_a.stat().st_size

    def get_media(self, rng):
        return self.raw("GET", self.media, [self.host(), ("Range", rng)])

    def test_plain_ranges_still_work(self):
        status, hdrs, body, _ = self.get_media("bytes=10-19")
        self.assertEqual(status, 206)
        self.assertEqual(hdrs["content-range"], f"bytes 10-19/{self.size}")
        self.assertEqual(body, self.clip_a.read_bytes()[10:20])

    def test_head_and_range_forms(self):
        with self.subTest(head="media"):
            status, hdrs, body, _ = self.raw("HEAD", self.media, [self.host()])
            self.assertEqual(status, 200)
            self.assertEqual(hdrs["content-length"], str(self.size))
            self.assertEqual(hdrs["accept-ranges"], "bytes")
            self.assertEqual(body, b"")
        with self.subTest(head="media range"):
            status, hdrs, body, _ = self.raw(
                "HEAD", self.media, [self.host(), ("Range", "bytes=0-1")])
            self.assertEqual(status, 206)
            self.assertEqual(hdrs["content-length"], "2")
            self.assertEqual(body, b"")
        for path in ("/", "/api/overview", "/nowhere"):
            with self.subTest(head=path):
                get_status, get_hdrs, _, _ = self.raw("GET", path,
                                                      [self.host()])
                status, hdrs, body, _ = self.raw("HEAD", path, [self.host()])
                self.assertEqual(status, get_status)
                self.assertEqual(hdrs["content-type"],
                                 get_hdrs["content-type"])
                self.assertEqual(body, b"")
        with self.subTest(rng="items=0-1"):
            status, _, body, _ = self.get_media("items=0-1")
            self.assertEqual(status, 200)
            self.assertEqual(len(body), self.size)
        with self.subTest(rng="Bytes=0-1"):
            status, _, body, _ = self.get_media("Bytes=0-1")
            self.assertEqual(status, 206)
            self.assertEqual(body, self.clip_a.read_bytes()[:2])
        with self.subTest(rng="multi"):
            status, _, _, _ = self.get_media("bytes=0-1,5-6")
            self.assertIn(status, (200, 206))
        huge = "9" * 5000
        with self.subTest(rng="huge end"):
            status, _, body, _ = self.get_media(f"bytes=0-{huge}")
            self.assertEqual(status, 206)
            self.assertEqual(len(body), self.size)
        with self.subTest(rng="huge suffix"):
            status, _, body, _ = self.get_media(f"bytes=-{huge}")
            self.assertEqual(status, 206)
            self.assertEqual(len(body), self.size)
        for rng in ("bytes=abc", "bytes=-", "bytes=99999999-", "bytes=-0",
                    f"bytes={huge}-"):
            with self.subTest(rng=rng[:30]):
                status, hdrs, _, _ = self.get_media(rng)
                self.assertEqual(status, 416)
                self.assertEqual(hdrs.get("content-range"),
                                 f"bytes */{self.size}")


@unittest.skipIf(os.geteuid() == 0, "root ignores file modes")
class TestUnreadableMedia(WebBase):
    def test_unreadable_media_gets_one_response(self):
        os.chmod(self.clip_a, 0)
        self.addCleanup(os.chmod, self.clip_a, 0o644)
        media = f"/media/{self.ids['a.mp4']}"
        for rng in (None, "bytes=0-99"):
            with self.subTest(rng=rng):
                hdrs = [self.host()] + ([("Range", rng)] if rng else [])
                status, _, _, buf = self.raw("GET", media, hdrs)
                self.assertEqual(status, 403, buf[:300])
                self.assertEqual(buf.count(b"HTTP/1."), 1, buf[:300])


class _FakeClip(io.BytesIO):
    """stands in for the clip once headers are out: it either fails the
    read like a dying disk, or comes up short like a file being cut."""

    def __init__(self, data, fail):
        super().__init__(data)
        self.fail = fail

    def read(self, n=-1):
        if self.fail:
            raise OSError(5, "input/output error")
        return super().read(n)


class TestMidStream(WebBase):
    def pipelined(self, count):
        """send count keep alive requests at once (the last one asks to
        close) and return every byte the server sends back."""
        media = f"/media/{self.ids['a.mp4']}"
        req = f"GET {media} HTTP/1.1\r\nHost: 127.0.0.1:{self.port}\r\n"
        reqs = (req + "\r\n") * (count - 1) + req + "Connection: close\r\n\r\n"
        with socket.create_connection(("127.0.0.1", self.port),
                                      timeout=10) as s:
            s.sendall(reqs.encode())
            buf = b""
            while chunk := s.recv(65536):
                buf += chunk
        return buf

    def test_read_error_after_headers_drops_the_connection(self):
        fake = _FakeClip(b"", fail=True)
        with mock.patch.object(server, "open", create=True,
                               return_value=fake):
            buf = self.pipelined(2)
        # the 200 already went out; a 500 after it would corrupt the stream
        self.assertTrue(buf.startswith(b"HTTP/1.0 200")
                        or buf.startswith(b"HTTP/1.1 200"), buf[:200])
        self.assertEqual(buf.count(b"HTTP/1."), 1, buf[:300])

    def test_file_that_shrinks_ends_the_connection(self):
        fake = _FakeClip(b"short", fail=False)
        # http/1.0 closes after every reply anyway; keep alive is where
        # a short body would bleed into the next response
        with mock.patch.object(server, "open", create=True,
                               return_value=fake), \
                mock.patch.object(server.Handler, "protocol_version",
                                  "HTTP/1.1"):
            buf = self.pipelined(2)
        # a short body followed by a second reply would be read as body
        self.assertEqual(buf.count(b"HTTP/1."), 1, buf[:300])
        self.assertTrue(buf.endswith(b"\r\n\r\nshort"), buf[-100:])


class TestBadInput(WebBase):
    def get(self, path):
        return self.raw("GET", path, [self.host()])

    def post(self, body):
        return self.raw("POST", "/api/export",
                        [self.host(), ("Content-Type", "application/json")],
                        body)

    def test_bad_params_are_4xx(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        for path in ("/api/matches?clip=abc", f"/api/matches?clip={a}&n=x",
                     "/api/sequence?seed=zz",
                     f"/api/sequence?seed={a}&length=2.5",
                     f"/api/sequence?seed={a}&length=-3"):
            with self.subTest(path=path):
                status, _, body, _ = self.get(path)
                self.assertEqual(status, 400, body)
                self.assertIn("error", json.loads(body))
        huge = "9" * 5000
        for path in (f"/api/matches?clip={huge}",
                     f"/api/matches?clip={a}&n={huge}",
                     f"/api/sequence?seed={a}&length={huge}"):
            with self.subTest(path=path[:40]):
                status, _, body, _ = self.get(path)
                self.assertEqual(status, 400, body[:200])
                self.assertIn("error", json.loads(body))
        big = "9" * 20
        for n in (big, huge):
            for path in (f"/api/clip/{n}", f"/media/{n}",
                         f"/thumb/{n}/start.jpg"):
                with self.subTest(path=path[:40]):
                    self.assertEqual(self.get(path)[0], 404)
        bodies = [b"not json", json.dumps([a, b]).encode(),
                  json.dumps({"ids": f"{a}{b}"}).encode(),
                  json.dumps({"ids": [str(a), str(b)]}).encode(),
                  json.dumps({"ids": [b, True]}).encode(),
                  b"[" * 200000 + b"]" * 200000,
                  json.dumps({"ids": [a, a]}).encode(),
                  json.dumps({"ids": [a, b], "mode": 3}).encode(),
                  json.dumps({"ids": [a, int(big)]}).encode()]
        for body in bodies:
            with self.subTest(body=body[:60]):
                status, _, reply, _ = self.post(body)
                self.assertEqual(status, 400, reply)
                self.assertIn("error", json.loads(reply))
        self.assertEqual(self.exported(), [])

    def test_body_over_a_mebibyte_is_413(self):
        # the length alone decides; the server never reads the body
        status, _, reply, _ = self.raw(
            "POST", "/api/export",
            [self.host(), ("Content-Type", "application/json"),
             ("Content-Length", str(1024 * 1024 + 1))])
        self.assertEqual(status, 413, reply)
        self.assertEqual(self.exported(), [])
        status, _, reply, _ = self.raw(
            "POST", "/api/export",
            [self.host(), ("Content-Type", "application/json"),
             ("Content-Length", "9" * 5000)])
        self.assertEqual(status, 413, reply[:200])

    def test_bad_mode_is_400_everywhere(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        for path in (f"/api/matches?clip={a}&mode=bogus",
                     f"/api/sequence?seed={a}&mode=bogus&length=2"):
            with self.subTest(path=path):
                status, _, body, _ = self.get(path)
                self.assertEqual(status, 400, body)
                self.assertIn("bogus", json.loads(body)["error"])
        with self.subTest(route="export"):
            status, _, body, _ = self.post(
                json.dumps({"ids": [a, b], "mode": "bogus"}).encode())
            self.assertEqual(status, 400, body)
            self.assertIn("bogus", json.loads(body)["error"])
        for bad in ("Same", "different", "bogus"):
            with self.subTest(country_mode=bad):
                status, _, body, _ = self.get(
                    f"/api/sequence?seed={a}&length=2&country_mode={bad}")
                self.assertEqual(status, 400, body)
                self.assertIn("country mode", json.loads(body)["error"])
        self.assertEqual(self.exported(), [])

    def test_good_params_still_answer(self):
        a = self.ids["a.mp4"]
        for path in (f"/api/matches?clip={a}&n=3&mode=momentum",
                     f"/api/sequence?seed={a}&length=2&country_mode=travel",
                     f"/api/clip/{a}"):
            with self.subTest(path=path):
                status, _, body, _ = self.get(path)
                self.assertEqual(status, 200, body)


class TestSearchAndStrictJson(WebBase):
    NAME = "Z\u00fcrich dawn.mp4"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # names made through cocoa land on disk decomposed
        nfd = unicodedata.normalize("NFD", cls.NAME)
        synth.write_clip(cls.clip_a.parent / nfd,
                         synth.frames_for("pan_right", n=24))

    def setUp(self):
        super().setUp()
        # a catalog written before scan stored nfc still holds nfd names
        nfd = unicodedata.normalize("NFD", self.NAME)
        conn = catalog.connect()
        conn.execute("UPDATE clips SET name=? WHERE name=?", (nfd, self.NAME))
        conn.commit()
        conn.close()

    def search(self, text):
        path = "/api/clips?" + urllib.parse.urlencode({"q": text})
        status, _, body, _ = self.raw("GET", path, [self.host()])
        self.assertEqual(status, 200, body)
        return [c["name"] for c in json.loads(body)["clips"]]

    def test_ascii_search_still_finds_names(self):
        self.assertEqual(len(self.search("rich")), 1)
        self.assertEqual(len(self.search("A.MP4")), 1)

    def test_nfc_search_finds_nfd_name(self):
        for text in ("z\u00fcrich", "Z\u00dcRICH",
                     unicodedata.normalize("NFD", "z\u00fcrich")):
            with self.subTest(text=text):
                self.assertEqual(len(self.search(text)), 1)

    def test_json_writer_refuses_nan(self):
        # the backstop on its own: the routes above never reach it
        h = server.Handler.__new__(server.Handler)
        h._head = False
        with self.assertRaises(ValueError):
            h._json({"x": float("nan")})

    def test_json_bodies_never_hold_nan(self):
        conn = catalog.connect()
        conn.execute("UPDATE features SET summary=? WHERE clip_id=?",
                     (SUMMARY.replace("1.25", "NaN"), self.ids["b.mp4"]))
        conn.commit()
        conn.close()

        def reject(tok):
            raise ValueError(f"non standard json token {tok}")
        a = self.ids["a.mp4"]
        for path in ("/api/clips", f"/api/matches?clip={a}",
                     f"/api/sequence?seed={a}&length=2"):
            with self.subTest(path=path):
                # one bad row must not take the whole listing down
                status, _, body, _ = self.raw("GET", path, [self.host()])
                self.assertEqual(status, 200, body)
                json.loads(body, parse_constant=reject)


class TestThumbRevalidation(WebBase):
    def setUp(self):
        super().setUp()
        cid = self.ids["a.mp4"]
        config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
        self.thumb = config.THUMB_DIR / f"{cid}_start.jpg"
        self.thumb.write_bytes(b"\xff\xd8 first \xff\xd9")
        self.url = f"/thumb/{cid}/start.jpg"

    def get(self, *extra, method="GET"):
        return self.raw(method, self.url, [self.host(), *extra])

    def test_thumb_revalidates_instead_of_max_age(self):
        status, hdrs, body, _ = self.get()
        self.assertEqual(status, 200)
        self.assertEqual(body, self.thumb.read_bytes())
        self.assertNotIn("max-age=86400", hdrs.get("cache-control", ""))
        self.assertIn("no-cache", hdrs.get("cache-control", ""))
        etag, stamp = hdrs["etag"], hdrs["last-modified"]
        for extra in (("If-None-Match", etag), ("If-None-Match", f"W/{etag}"),
                      ("If-None-Match", f'"nope", {etag}'),
                      ("If-None-Match", "*"),
                      ("If-Modified-Since", stamp)):
            with self.subTest(header=extra):
                status, hdrs2, body, _ = self.get(extra)
                self.assertEqual(status, 304)
                self.assertEqual(body, b"")
                self.assertEqual(hdrs2["etag"], etag)
        with self.subTest(method="HEAD"):
            status, hdrs2, body, _ = self.get(method="HEAD")
            self.assertEqual((status, body), (200, b""))
            self.assertEqual(hdrs2["etag"], etag)

    def test_rewritten_thumb_is_served_fresh(self):
        # reanalysis rewrites the jpeg under the same url
        _, hdrs, _, _ = self.get()
        etag, stamp = hdrs["etag"], hdrs["last-modified"]
        st = self.thumb.stat()
        self.thumb.write_bytes(b"\xff\xd8 second one \xff\xd9")
        os.utime(self.thumb, ns=(st.st_atime_ns, st.st_mtime_ns + 2 * 10**9))
        for extra in (("If-None-Match", etag), ("If-Modified-Since", stamp)):
            with self.subTest(header=extra):
                status, hdrs2, body, _ = self.get(extra)
                self.assertEqual(status, 200)
                self.assertEqual(body, self.thumb.read_bytes())
                self.assertNotEqual(hdrs2["etag"], etag)

    def test_same_second_same_size_rewrite_is_fresh(self):
        # a fast reanalysis can rewrite within one second at the same size
        base = 1_700_000_000 * 10**9 + 100 * 10**6
        os.utime(self.thumb, ns=(base, base))
        _, hdrs, _, _ = self.get()
        etag = hdrs["etag"]
        self.thumb.write_bytes(b"\xff\xd8 FIRST \xff\xd9")
        os.utime(self.thumb, ns=(base, base + 10**6))
        status, hdrs2, body, _ = self.get(("If-None-Match", etag))
        self.assertEqual(status, 200)
        self.assertEqual(body, b"\xff\xd8 FIRST \xff\xd9")
        self.assertNotEqual(hdrs2["etag"], etag)

    def test_if_none_match_wins_over_if_modified_since(self):
        _, hdrs, _, _ = self.get()
        status, _, body, _ = self.get(("If-None-Match", '"other"'),
                                      ("If-Modified-Since",
                                       hdrs["last-modified"]))
        self.assertEqual(status, 200)
        self.assertEqual(body, self.thumb.read_bytes())

    def test_dates_without_a_zone_are_read_as_gmt(self):
        # formatdate writes -0000; read as local time it would shift by
        # the zone offset, one way or the other
        mtime = 1_700_000_000
        os.utime(self.thumb, (mtime, mtime))
        for when, want in ((mtime, 304), (mtime - 3600, 200)):
            stamp = email.utils.formatdate(when)
            self.assertTrue(stamp.endswith("-0000"), stamp)
            with self.subTest(stamp=stamp):
                status, _, _, _ = self.get(("If-Modified-Since", stamp))
                self.assertEqual(status, want)

    def test_bad_validator_headers_send_the_thumb(self):
        for extra in (("If-Modified-Since", "yesterday"),
                      ("If-None-Match", '"stale"')):
            with self.subTest(header=extra):
                status, _, body, _ = self.get(extra)
                self.assertEqual(status, 200)
                self.assertEqual(body, self.thumb.read_bytes())


class TestMediaFiles(WebBase):
    UNICODE = "caf\u00e9 night 1.mp4"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        other = cls.root / "Sony SLOG-3" / "Iceland"
        other.mkdir()
        # scan never probes, so any bytes stand in for these containers
        (other / "c.mov").write_bytes(b"mov body " * 40)
        (other / "d.mxf").write_bytes(b"mxf body " * 30)
        (cls.clip_a.parent / cls.UNICODE).write_bytes(
            bytes(range(256)) * 3)

    def setUp(self):
        super().setUp()
        self.media = f"/media/{self.ids['a.mp4']}"
        self.data = self.clip_a.read_bytes()
        self.size = len(self.data)

    def get(self, path, *extra):
        return self.raw("GET", path, [self.host(), *extra])

    def test_media_open_ended_and_clamped_ranges(self):
        last = self.size - 1
        cases = [("bytes=0-", 0, last), ("bytes=10-" + "9" * 15, 10, last),
                 ("bytes=0-0", 0, 0)]
        for rng, start, end in cases:
            with self.subTest(rng=rng):
                status, hdrs, body, _ = self.get(self.media, ("Range", rng))
                self.assertEqual(status, 206)
                self.assertEqual(hdrs["content-range"],
                                 f"bytes {start}-{end}/{self.size}")
                self.assertEqual(hdrs["content-length"], str(end - start + 1))
                self.assertEqual(body, self.data[start:end + 1])

    def test_media_missing_file_and_content_types(self):
        for name, ctype in (("a.mp4", "video/mp4"),
                            ("c.mov", "video/quicktime"),
                            ("d.mxf", "application/mxf")):
            with self.subTest(name=name):
                status, hdrs, _, _ = self.get(f"/media/{self.ids[name]}")
                self.assertEqual(status, 200)
                self.assertEqual(hdrs["content-type"], ctype)
        with self.subTest(case="unknown id"):
            status, _, body, _ = self.get("/media/987654")
            self.assertEqual(status, 404)
            self.assertIn("error", json.loads(body))
        with self.subTest(case="missing file"):
            gone = self.clip_a.with_name("a.moved")
            os.rename(self.clip_a, gone)
            self.addCleanup(os.rename, gone, self.clip_a)
            status, _, body, _ = self.get(self.media)
            self.assertEqual(status, 404)
            self.assertIn("missing", json.loads(body)["error"])

    def test_media_unicode_and_space_path(self):
        cid = self.ids[self.UNICODE]
        want = (self.clip_a.parent / self.UNICODE).read_bytes()
        status, _, body, _ = self.get(f"/media/{cid}")
        self.assertEqual(status, 200)
        self.assertEqual(body, want)
        status, _, body, _ = self.get(f"/media/{cid}", ("Range", "bytes=5-9"))
        self.assertEqual((status, body), (206, want[5:10]))

    def listed(self, **params):
        status, _, body, _ = self.get(
            "/api/clips?" + urllib.parse.urlencode(params))
        self.assertEqual(status, 200, body)
        return sorted(c["name"] for c in json.loads(body)["clips"])

    def test_clips_filters(self):
        conn = catalog.connect()
        conn.execute("UPDATE features SET summary=? WHERE clip_id=?",
                     (SUMMARY.replace('"start_class": "pan_right"',
                                      '"start_class": "static"'),
                      self.ids["c.mov"]))
        conn.commit()
        conn.close()
        self.assertEqual(len(self.listed()), 5)
        self.assertEqual(self.listed(country="Iceland"), ["c.mov", "d.mxf"])
        self.assertEqual(self.listed(country="Nowhere"), [])
        self.assertEqual(self.listed(klass="static"), ["c.mov"])
        self.assertEqual(len(self.listed(klass="pan_right")), 5)
        self.assertEqual(self.listed(q="D.MXF"), ["d.mxf"])
        self.assertEqual(self.listed(q="CAF\u00c9"), [self.UNICODE])
        self.assertEqual(
            self.listed(country="Japan", klass="static"), [])


class TestRouteEdges(WebBase):
    def get(self, path):
        return self.raw("GET", path, [self.host()])

    def post(self, path, obj):
        return self.raw("POST", path,
                        [self.host(), ("Content-Type", "application/json")],
                        json.dumps(obj).encode())

    def test_api_400_paths(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        for path in ("/api/matches?clip=987654",
                     "/api/sequence?seed=987654&length=2"):
            with self.subTest(path=path):
                status, _, body, _ = self.get(path)
                self.assertEqual(status, 400, body)
                self.assertIn("features", json.loads(body)["error"])
        with self.subTest(export="one clip"):
            status, _, body, _ = self.post("/api/export", {"ids": [a]})
            self.assertEqual(status, 400, body)
            self.assertIn("two clips", json.loads(body)["error"])
        with self.subTest(export="unanalyzed"):
            conn = catalog.connect()
            conn.execute("DELETE FROM features WHERE clip_id=?", (b,))
            conn.commit()
            conn.close()
            status, _, body, _ = self.post("/api/export", {"ids": [a, b]})
            self.assertEqual(status, 400, body)
            self.assertIn(str(b), json.loads(body)["error"])
        with self.subTest(post="unknown route"):
            status, _, body, _ = self.post("/api/nowhere", {"ids": [a, b]})
            self.assertEqual(status, 404, body)
            self.assertIn("error", json.loads(body))
        self.assertEqual(self.exported(), [])

    def test_thumb_positions(self):
        cid = self.ids["a.mp4"]
        config.THUMB_DIR.mkdir(parents=True, exist_ok=True)
        for pos in ("mid", "end"):
            (config.THUMB_DIR / f"{cid}_{pos}.jpg").write_bytes(
                f"jpeg {pos}".encode())
        # a file next to the thumbs folder that no url may reach
        (config.THUMB_DIR.parent / "secret.jpg").write_bytes(b"secret")
        for pos in ("mid", "end"):
            with self.subTest(pos=pos):
                status, hdrs, body, _ = self.get(f"/thumb/{cid}/{pos}.jpg")
                self.assertEqual(status, 200)
                self.assertEqual(hdrs["content-type"], "image/jpeg")
                self.assertEqual(body, f"jpeg {pos}".encode())
        for pos in ("side", "MID", "start"):
            with self.subTest(pos=pos):
                status, _, body, _ = self.get(f"/thumb/{cid}/{pos}.jpg")
                self.assertEqual(status, 404)
                self.assertIn("error", json.loads(body))
        for query in ("?../../secret", "?p=../secret.jpg", "?/etc/passwd"):
            with self.subTest(query=query):
                status, _, body, _ = self.get(f"/thumb/{cid}/mid.jpg{query}")
                self.assertEqual((status, body), (200, b"jpeg mid"))

    def test_path_traversal_routes_404(self):
        cid = self.ids["a.mp4"]
        for path in ("/../../etc/passwd", "/static/../server.py",
                     "/index.html", "/static/index.html",
                     "/%2e%2e/%2e%2e/etc/passwd", "//etc/passwd",
                     f"/media/{cid}/../{cid}", f"/media/../{cid}",
                     f"/media/{cid}%2f..%2fetc", f"/thumb/{cid}/../start.jpg",
                     f"/thumb/{cid}/..%2f..%2fsecret.jpg",
                     f"/api/clip/{cid}/..", "/api/../api/overview"):
            with self.subTest(path=path):
                status, hdrs, body, _ = self.get(path)
                self.assertEqual(status, 404, body[:200])
                self.assertEqual(hdrs["content-type"], "application/json")
                self.assertIn("error", json.loads(body))


    def test_concurrent_reads(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        data = self.clip_a.read_bytes()
        paths = ["/api/overview", "/api/clips", f"/api/clip/{a}",
                 f"/api/matches?clip={a}", f"/api/sequence?seed={b}&length=2",
                 f"/media/{a}", "/"]

        def one(i):
            path = paths[i % len(paths)]
            extra = [("Range", "bytes=0-99")] if i % 14 == 5 else []
            status, _, body, _ = self.raw("GET", path, [self.host(), *extra])
            if path.startswith("/media/"):
                want = data[:100] if extra else data
                return status in (200, 206) and body == want
            return status == 200

        with ThreadPoolExecutor(max_workers=48) as pool:
            results = list(pool.map(one, range(200)))
        self.assertEqual(results.count(False), 0)

    def test_empty_and_single_clip_library(self):
        a, b = self.ids["a.mp4"], self.ids["b.mp4"]
        conn = catalog.connect()
        conn.execute("DELETE FROM clips WHERE id=?", (b,))
        conn.commit()
        conn.close()
        with self.subTest(library="one clip"):
            listed = json.loads(self.get("/api/clips")[2])["clips"]
            self.assertEqual([c["id"] for c in listed], [a])
            status, _, body, _ = self.get(f"/api/matches?clip={a}")
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["matches"], [])
            status, _, body, _ = self.get(f"/api/sequence?seed={a}&length=4")
            self.assertEqual(status, 200, body)
            seq = json.loads(body)
            self.assertEqual([c["id"] for c in seq["clips"]], [a])
            self.assertEqual((seq["edges"], seq["total"]), ([], 0))
        conn = catalog.connect()
        conn.execute("DELETE FROM clips")
        conn.commit()
        conn.close()
        with self.subTest(library="empty"):
            status, _, body, _ = self.get("/api/overview")
            self.assertEqual(status, 200, body)
            ov = json.loads(body)
            self.assertEqual(ov["totals"]["total"], 0)
            self.assertEqual((ov["countries"], ov["missing"]), ([], 0))
            self.assertEqual(json.loads(self.get("/api/clips")[2]),
                             {"clips": []})
            for path in (f"/api/matches?clip={a}",
                         f"/api/sequence?seed={a}&length=2"):
                self.assertEqual(self.get(path)[0], 400)
            status, _, body, _ = self.post("/api/export", {"ids": [a, b]})
            self.assertEqual(status, 400, body)
        self.assertEqual(self.exported(), [])


class TestServe(TempDirsMixin, unittest.TestCase):
    def test_serve_binds_loopback(self):
        made = []

        class FakeServer:
            def __init__(self, address, handler):
                made.append(self)
                self.address, self.handler = address, handler
                self.closed = False

            def serve_forever(self):
                raise KeyboardInterrupt

            def server_close(self):
                self.closed = True

        out = io.StringIO()
        with mock.patch.object(server, "ThreadingHTTPServer", FakeServer), \
                contextlib.redirect_stdout(out):
            server.serve(8123)
        self.assertEqual(len(made), 1)
        self.assertEqual(made[0].address, ("127.0.0.1", 8123))
        self.assertIs(made[0].handler, server.Handler)
        self.assertTrue(made[0].closed)
        self.assertIn("http://127.0.0.1:8123", out.getvalue())
        self.assertIn("stopped", out.getvalue())


if __name__ == "__main__":
    unittest.main()
