# server.py
# localhost only web ui. stdlib http server: no frameworks, no external
# assets, nothing ever leaves 127.0.0.1. video previews stream straight
# from the footage tree with byte range support so the browser can scrub.

import json
import os
import re
import unicodedata
from datetime import timezone
from email.utils import formatdate, parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from clipengine import catalog, config, matching, sequence

_STATIC = Path(__file__).parent / "static"
_CTYPES = {".mp4": "video/mp4", ".mov": "video/quicktime",
           ".mxf": "application/mxf"}
_CHUNK = 1024 * 1024
_THUMB_RE = re.compile(r"^/thumb/(\d+)/(start|mid|end)\.jpg$")
_MEDIA_RE = re.compile(r"^/media/(\d+)$")
_CLIP_RE = re.compile(r"^/api/clip/(\d+)$")
_RANGE_SPEC_RE = re.compile(r"^\s*(\d*)\s*-\s*(\d*)\s*$")
_LOOPBACK = ("127.0.0.1", "localhost", "[::1]")
_DIGITS_RE = re.compile(r"[0-9]+")
# sqlite keeps ids in a signed 64 bit integer; anything larger is unknown
_MAX_ID = 2 ** 63 - 1


def _too_long(digits: str) -> bool:
    """more digits than any 64 bit value. measured before int(), which
    refuses strings past 4300 digits with an error of its own."""
    return len(digits.lstrip("0")) > len(str(_MAX_ID))


class _BadRequest(Exception):
    """a client mistake: answered with its own code and a short message
    instead of a 500."""

    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _int(q: dict, key: str, default: int) -> int:
    """a whole number query parameter, or 400."""
    raw = q.get(key)
    if raw is None or raw == "":
        return default
    if not _DIGITS_RE.fullmatch(raw):
        raise _BadRequest(400, f"{key} must be a whole number, got {raw!r}")
    if _too_long(raw):
        raise _BadRequest(400, f"{key} is out of range")
    return int(raw)


def _route_id(text: str) -> int:
    if _too_long(text):
        raise _BadRequest(404, "unknown clip")
    clip_id = int(text)
    if clip_id > _MAX_ID:
        raise _BadRequest(404, "unknown clip")
    return clip_id


def _mode(value) -> str:
    if not isinstance(value, str) or value not in config.SCORING_MODES:
        raise _BadRequest(400, f"unknown mode: {value}")
    return value


def _fold(text: str) -> str:
    # one spelling for search: macos hands out nfd names, browsers send nfc
    return unicodedata.normalize("NFC", text).casefold()


class Handler(BaseHTTPRequestHandler):
    server_version = "ClipEngine/0.1"
    # set per request: head drops every body, started means a status line
    # is already out and a second response would corrupt the stream
    _head = False
    _started = False

    def log_message(self, fmt, *args):
        pass

    def send_response(self, code, message=None):
        self._started = True
        super().send_response(code, message)

    # -- plumbing ---------------------------------------------------------

    def _json(self, obj, code: int = 200) -> None:
        # nan or infinity is not json; a leak fails loudly here instead of
        # breaking the json parse in the page
        body = json.dumps(obj, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not self._head:
            self.wfile.write(body)

    def _error(self, code: int, msg: str) -> None:
        self._json({"error": msg}, code)

    def _bytes(self, body: bytes, ctype: str, cache: str = "no-store") -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        if not self._head:
            self.wfile.write(body)

    def _fail(self, exc: Exception) -> None:
        """answer an unexpected error, unless a response already started:
        then the only safe move is to drop the connection."""
        if self._started:
            self.close_connection = True
            return
        try:
            self._error(500, f"{type(exc).__name__}: {exc}")
        except Exception:
            pass

    # -- guards -------------------------------------------------------------

    def _hosts(self) -> set:
        port = self.server.server_address[1]
        names = {f"{n}:{port}" for n in _LOOPBACK}
        if port == 80:
            names.update(_LOOPBACK)
        return names

    def _host_ok(self) -> bool:
        """a page on another site can still reach a loopback port through
        dns rebinding, so only loopback names are answered. no host at all
        is fine: raw http/1.0 clients never send one."""
        host = self.headers.get("Host")
        if host is None or host.strip().lower() in self._hosts():
            return True
        self.close_connection = True
        self._error(421, "unknown host; open the ui on 127.0.0.1")
        return False

    def _post_ok(self) -> bool:
        """a cross site form can post text/plain without a preflight, so
        writes need json and an origin that is absent or this server."""
        origin = self.headers.get("Origin")
        if origin is not None and origin.strip().lower() not in {
                f"http://{h}" for h in self._hosts()}:
            self.close_connection = True
            self._error(403, "cross origin request refused")
            return False
        ctype = (self.headers.get("Content-Type") or "").split(";")[0]
        if ctype.strip().lower() != "application/json":
            self.close_connection = True
            self._error(415, "send application/json")
            return False
        return True

    # -- routing ------------------------------------------------------------

    def do_HEAD(self):
        self._head, self._started = True, False
        self._get()

    def do_GET(self):
        self._head, self._started = False, False
        self._get()

    def _get(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        route = url.path
        try:
            if not self._host_ok():
                return
            if route == "/":
                return self._bytes((_STATIC / "index.html").read_bytes(),
                                   "text/html; charset=utf-8")
            if route == "/api/overview":
                return self._overview()
            if route == "/api/clips":
                return self._clips(q)
            m = _CLIP_RE.match(route)
            if m:
                return self._clip(_route_id(m.group(1)))
            if route == "/api/matches":
                return self._matches(q)
            if route == "/api/sequence":
                return self._sequence(q)
            m = _THUMB_RE.match(route)
            if m:
                return self._thumb(_route_id(m.group(1)), m.group(2))
            m = _MEDIA_RE.match(route)
            if m:
                return self._media(_route_id(m.group(1)))
            self._error(404, "not found")
        except _BadRequest as exc:
            self._error(exc.code, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            self._fail(exc)

    def do_POST(self):
        self._head, self._started = False, False
        url = urlparse(self.path)
        try:
            if not self._host_ok() or not self._post_ok():
                return
            if url.path == "/api/export":
                return self._export(self._body())
            self._error(404, "not found")
        except _BadRequest as exc:
            self._error(exc.code, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:
            self._fail(exc)

    def _body(self):
        """the posted json document, or 400."""
        raw = self.headers.get("Content-Length") or "0"
        if not _DIGITS_RE.fullmatch(raw.strip()):
            self.close_connection = True
            raise _BadRequest(400, "bad content length")
        if _too_long(raw.strip()) or int(raw) > _CHUNK:
            self.close_connection = True
            raise _BadRequest(413, "body too large")
        data = self.rfile.read(int(raw))
        try:
            return json.loads(data or b"{}")
        except (ValueError, RecursionError):
            # a deeply nested document blows the decoder stack
            raise _BadRequest(400, "body is not valid json") from None

    # -- api ------------------------------------------------------------------

    def _overview(self):
        conn = catalog.connect()
        try:
            ov = catalog.overview(conn)
        finally:
            conn.close()
        ov["modes"] = list(config.SCORING_MODES)
        self._json(ov)

    def _clips(self, q):
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn, q.get("country") or None)
        finally:
            conn.close()
        klass = q.get("klass") or ""
        needle = _fold(q.get("q") or "")
        out = []
        for m in lib.meta:
            if klass and klass not in (m["start_class"], m["end_class"]):
                continue
            if needle and needle not in _fold(m["name"]):
                continue
            out.append({k: m[k] for k in
                        ("id", "name", "country", "profile", "duration_s",
                         "start_class", "end_class")})
        self._json({"clips": out})

    def _clip(self, clip_id: int):
        conn = catalog.connect()
        try:
            row = catalog.get_clip(conn, clip_id)
            if not row:
                return self._error(404, "unknown clip")
            feat = conn.execute(
                "SELECT summary, error FROM features WHERE clip_id=?",
                (clip_id,)).fetchone()
        finally:
            conn.close()
        detail = {"id": row["id"], "name": row["name"],
                  "country": row["country"], "profile": row["profile"],
                  "rel_path": row["rel_path"], "available": row["available"],
                  "size_bytes": row["size_bytes"],
                  "summary": (json.loads(feat["summary"])
                              if feat and feat["summary"] else None),
                  "error": feat["error"] if feat else None}
        self._json(detail)

    def _matches(self, q):
        clip_id = _int(q, "clip", 0)
        mode = _mode(q.get("mode", config.DEFAULT_MODE))
        n = _int(q, "n", 12)
        country = q.get("country", "any")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        if clip_id not in lib.row_of:
            return self._error(400, "clip has no features yet")
        try:
            results = matching.rank(lib, clip_id, mode, n=n, country=country)
        except ValueError as exc:
            return self._error(400, str(exc))
        self._json({"clip": lib.meta[lib.row_of[clip_id]],
                    "mode": mode, "matches": results})

    def _sequence(self, q):
        seed = _int(q, "seed", 0)
        mode = _mode(q.get("mode", config.DEFAULT_MODE))
        length = _int(q, "length", 8)
        country_mode = q.get("country_mode", "any")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        if seed not in lib.row_of:
            return self._error(400, "seed clip has no features yet")
        matrix = matching.full_matrix(lib, mode)
        countries = [m["country"] for m in lib.meta]
        try:
            rows_idx, edges = sequence.build_chain(
                matrix, lib.row_of[seed], length,
                countries=countries, country_mode=country_mode)
        except ValueError as exc:
            return self._error(400, str(exc))
        chain = [lib.meta[i] for i in rows_idx]
        self._json({"mode": mode, "total": round(sum(edges), 4),
                    "edges": [round(e, 4) for e in edges],
                    "clips": [{k: m[k] for k in
                               ("id", "name", "country", "duration_s",
                                "start_class", "end_class")}
                              for m in chain]})

    def _export(self, payload):
        if not isinstance(payload, dict):
            return self._error(400, "send a json object with ids and mode")
        ids = payload.get("ids", [])
        # bool is an int subclass, and a string would export per character
        if not isinstance(ids, list) or not all(
                type(i) is int for i in ids):
            return self._error(400, "ids must be a list of clip ids")
        if len(set(ids)) != len(ids):
            return self._error(400, "a clip can appear only once")
        mode = _mode(payload.get("mode", config.DEFAULT_MODE))
        if len(ids) < 2:
            return self._error(400, "need at least two clips to export")
        conn = catalog.connect()
        try:
            lib = matching.load_library(conn)
        finally:
            conn.close()
        missing = [i for i in ids if i not in lib.row_of]
        if missing:
            return self._error(400, f"clips without features: {missing}")
        edges = []
        for a, b in zip(ids, ids[1:]):
            scores, _ = matching.score_against(lib, lib.row_of[a], mode)
            edges.append(float(scores[lib.row_of[b]]))
        meta = [lib.meta[lib.row_of[i]] for i in ids]
        json_path, m3u_path, xml_path = sequence.export_chain(
            meta, edges, mode)
        self._json({"json": str(json_path), "m3u8": str(m3u_path),
                    "fcpxml": str(xml_path),
                    "total": round(sum(edges), 4)})

    # -- files --------------------------------------------------------------

    def _thumb(self, clip_id: int, pos: str):
        """reanalysis rewrites a thumbnail under the same url, so the
        browser keeps a copy but asks again each time: an etag and a
        last modified stamp let an unchanged jpeg answer 304."""
        path = config.THUMB_DIR / f"{clip_id}_{pos}.jpg"
        try:
            with open(path, "rb") as f:
                st = os.fstat(f.fileno())
                body = f.read()
        except OSError:
            return self._error(404, "no thumbnail")
        etag = f'"{st.st_mtime_ns:x}-{st.st_size:x}"'
        stamp = formatdate(st.st_mtime, usegmt=True)
        fresh = self._not_modified(etag, int(st.st_mtime))
        self.send_response(304 if fresh else 200)
        self.send_header("ETag", etag)
        self.send_header("Last-Modified", stamp)
        self.send_header("Cache-Control", "no-cache")
        if fresh:
            self.end_headers()
            return
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if not self._head:
            self.wfile.write(body)

    def _not_modified(self, etag: str, mtime: int) -> bool:
        """rfc 9110: if none match wins over if modified since, and
        compares weakly. an unreadable date means send the body."""
        match = self.headers.get("If-None-Match")
        if match is not None:
            tags = [t.strip() for t in match.split(",")]
            return "*" in tags or any(
                t.removeprefix("W/") == etag for t in tags)
        since = self.headers.get("If-Modified-Since")
        if not since:
            return False
        try:
            when = parsedate_to_datetime(since)
        except (TypeError, ValueError, IndexError):
            return False
        if when.tzinfo is None:
            # http dates are always gmt, even when written as -0000
            when = when.replace(tzinfo=timezone.utc)
        return mtime <= when.timestamp()

    def _media(self, clip_id: int):
        """stream a clip with http range support. ids come from the
        catalog, so only cataloged footage paths are ever served, and an
        evicted file is refused rather than silently pulled from icloud."""
        conn = catalog.connect()
        try:
            row = catalog.get_clip(conn, clip_id)
        finally:
            conn.close()
        if not row:
            return self._error(404, "unknown clip")
        path = Path(row["path"])
        try:
            st = os.stat(path)
        except OSError:
            return self._error(404, "file missing on disk")
        if st.st_blocks == 0 and st.st_size > 0:
            return self._error(409, "clip is evicted to icloud;"
                               " download it in finder first")
        size = st.st_size
        ctype = _CTYPES.get(path.suffix.lower(), "application/octet-stream")
        rng = _parse_range(self.headers.get("Range"), size)
        if rng is False:
            self.close_connection = True
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = rng or (0, size - 1)
        # open before any status line goes out, so a locked or vanished
        # file still gets one clean error response
        try:
            f = open(path, "rb")
        except PermissionError:
            return self._error(403, "clip is not readable")
        except OSError:
            return self._error(404, "file missing on disk")
        with f:
            self.send_response(206 if rng else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if rng:
                self.send_header("Content-Range",
                                 f"bytes {start}-{end}/{size}")
            self.end_headers()
            if self._head:
                return
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(_CHUNK, remaining))
                if not chunk:
                    # the file shrank under us; a short body must not be
                    # followed by another response on this connection
                    self.close_connection = True
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def _parse_range(header, size: int):
    """read a range header. None means serve the whole file: no header,
    another unit, or several ranges (all allowed to be ignored by rfc
    9110). False means 416. otherwise an inclusive (start, end)."""
    if not header:
        return None
    unit, sep, spec = header.partition("=")
    if not sep or unit.strip().lower() != "bytes" or "," in spec:
        return None
    m = _RANGE_SPEC_RE.match(spec)
    if not m or not (m.group(1) or m.group(2)):
        return False
    first, last = m.group(1), m.group(2)
    # a position too long to be a 64 bit value is past the end of any file
    if first and _too_long(first):
        return False
    if last and _too_long(last):
        last = str(size)
    if first:
        start = int(first)
        end = min(int(last), size - 1) if last else size - 1
    else:
        # suffix form: last n bytes
        start, end = max(0, size - int(last)), size - 1
    if start >= size or start > end:
        return False
    return start, end


def serve(port: int = config.SERVER_PORT) -> None:
    config.create_directories()
    httpd = ThreadingHTTPServer((config.SERVER_HOST, port), Handler)
    print(f"clipengine ui: http://{config.SERVER_HOST}:{port}"
          " (ctrl+c to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
