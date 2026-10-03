# catalog.py
# sqlite catalog of every clip under the footage tree. the scan is
# stat only: it never opens a video file, so stubs evicted to icloud
# are cataloged without triggering a single byte of download.

import logging
import os
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from clipengine import config

logger = logging.getLogger(__name__)

SCHEMA = """
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
    analyzed_at TEXT NOT NULL,
    is_log INTEGER
);
"""

# join condition for a feature row made from this clip as it is now:
# same algorithm version, the file has not changed since analysis, and
# the clip still has the log flag it was analyzed with. it matches
# failures too, so pending can tell tried from never tried
_TRIED = ("f.clip_id = c.id AND f.version = ? AND f.content_key = c.content_key"
          " AND f.is_log = c.is_log")
# a tried row that holds a usable vector
_FRESH = _TRIED + " AND f.error IS NULL AND f.vector IS NOT NULL"


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """open the catalog, creating directories and schema on first use.
    the path resolves at call time, not import time, so tests can
    repoint config at a temp directory."""
    if db_path is None:
        db_path = config.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    # country filters compare through this, see _IN_COUNTRY
    conn.create_function("fold", 1, _fold, deterministic=True)
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    return any(r["name"] == column
               for r in conn.execute(f"PRAGMA table_info({table})"))


def _migrate(conn: sqlite3.Connection) -> None:
    """bring a catalog from an older release up to SCHEMA in place."""
    if _has_column(conn, "features", "is_log"):
        return
    # take the write lock first so two processes opening the same old
    # catalog cannot both add the column
    conn.execute("BEGIN IMMEDIATE")
    try:
        if not _has_column(conn, "features", "is_log"):
            conn.execute("ALTER TABLE features ADD COLUMN is_log INTEGER")
            # scans never changed is_log before this column existed, so
            # every stored row was made with its clip's current flag
            conn.execute(
                "UPDATE features SET is_log ="
                " (SELECT c.is_log FROM clips c WHERE c.id = features.clip_id)")
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _now() -> str:
    # microsecond precision: two scans in the same second must still
    # produce distinct last_seen stamps
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def is_materialized(st: os.stat_result) -> bool:
    """true when the file has local data. stubs evicted to icloud report full
    size but occupy zero blocks; reading one forces a network download.
    an empty file has nothing to download, so it counts as local and
    fails where analyze can report it."""
    return st.st_blocks > 0 or st.st_size == 0


def _nfc(text: str) -> str:
    # one spelling for names copied from hfs+ or smb with nfd bytes
    return unicodedata.normalize("NFC", text)


def _fold(text):
    # a typed country against a stored one: one spelling, any case,
    # the way a case insensitive apfs volume compares folder names
    return _nfc(text).casefold() if isinstance(text, str) else text


# the country filter pending, known_failures and analyzed share
_IN_COUNTRY = " AND fold(c.country) = fold(?)"


def classify_path(rel: Path) -> tuple[str, int, str]:
    """map a path relative to the media root onto (profile, is_log, country).
    layout is <camera profile>/<country>/file, so country is the vibe tag.
    a file straight in the root has no profile folder. names come back
    nfc so one country never splits into two spellings."""
    parts = [_nfc(p) for p in rel.parts]
    profile_key, is_log = "unknown", 0
    country = "Unsorted"
    if len(parts) >= 2:
        cam = config.CAMERA_PROFILES.get(parts[0])
        if cam:
            profile_key = cam["key"]
            is_log = 1 if cam["log"] else 0
        else:
            profile_key = parts[0]
    if len(parts) >= 3:
        country = parts[1]
    return profile_key, is_log, country


def normalize_root(path) -> Path:
    """absolute, ~ expanded and .. folded, without resolving symlinks, so
    every spelling of one tree stores the same path strings."""
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _key(st: os.stat_result) -> str:
    # size and mtime, never a hash: hashing reads every byte
    return f"{st.st_size}-{st.st_mtime_ns}"


def _under(path: str, folder: str) -> bool:
    prefix = folder if folder.endswith(os.sep) else folder + os.sep
    return path.startswith(prefix)


def _clean(path: str) -> bool:
    """a path the current scan could have written: absolute, normalized.
    rows from older relative or .. roots fail this and count as gone."""
    return os.path.isabs(path) and os.path.normpath(path) == path


def _tree_of(row) -> Optional[str]:
    """the root a row was first cataloged under: path minus rel_path."""
    path, rel = row["path"], row["rel_path"]
    if not _clean(path) or not path.endswith(os.sep + rel):
        return None
    return path[:len(path) - len(rel) - 1] or os.sep


def _rel(full: str, row, root: str, tree: str) -> str:
    """the media root relative path classify_path reads. the bases tried,
    in order: the tree a known row was first cataloged under, the
    configured media root, the scanned root. the first that puts the clip
    under a known camera folder wins, else the first that holds it. so a
    subfolder scan cannot shift a clip, a library below the media root
    reads its own camera folders, and a clip filed wrong by an older
    subfolder scan is put right."""
    bases = [b for b in (_tree_of(row) if row is not None else None,
                         tree, root)
             if b is not None and _under(full, b)]
    rels = [os.path.relpath(full, b) for b in bases]
    for rel in rels:
        if _nfc(rel.split(os.sep)[0]) in config.CAMERA_PROFILES:
            return rel
    # a library a few folders down: its shallowest camera folder
    parts = rels[0].split(os.sep)
    for i in range(1, len(parts) - 1):
        if _nfc(parts[i]) in config.CAMERA_PROFILES:
            return os.sep.join(parts[i:])
    return rels[0]


def _whereabouts(path: str, st: Optional[os.stat_result] = None) -> str:
    """stat a cataloged path again: 'same' when it is the file st describes
    under another spelling, 'gone' when nothing is there, 'here' when a
    different file is, 'unknown' when we cannot tell."""
    if not _clean(path):
        return "gone"
    try:
        old = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return "gone"
    except OSError:
        return "unknown"
    if st is not None and (old.st_dev, old.st_ino) == (st.st_dev, st.st_ino):
        return "same"
    return "here"


def _claim(full: str, st: os.stat_result, candidates: list,
           one_new_file: bool, left: set,
           away: Optional[str] = None) -> Optional[sqlite3.Row]:
    """the untouched row a loose path takes over, if any. the same file
    under another spelling wins outright. otherwise exactly one row with
    this content key whose file is gone, rows lost in this pass before
    rows already missing, and only when no other loose path shares the
    key. two candidates for one key are never guessed between. rows in
    left are gone: their path now holds other content. rows under away,
    a media root that is not mounted, are neither claimed nor gone."""
    # a symlink or hard link is a second name for a file that is still
    # at its first, not a new spelling of the first
    respelled = not os.path.islink(full) and st.st_nlink == 1
    gone_now, gone_before = [], []
    for row in candidates:
        if away is not None and _under(row["path"], away):
            continue
        if row["id"] in left:
            where = "gone"
        else:
            where = _whereabouts(row["path"], st if respelled else None)
        if where == "same":
            return row
        if where == "gone":
            (gone_before if row["missing"] else gone_now).append(row)
    if not one_new_file:
        return None
    for group in (gone_now, gone_before):
        if group:
            return group[0] if len(group) == 1 else None
    return None


def scan(conn: sqlite3.Connection, media_root: Optional[Path] = None) -> dict:
    """walk the footage tree with stat calls only and upsert the catalog.

    a clip is marked missing only when the scan knows its file is gone:
    its folder was listed without it, it sits outside the scanned root
    and a direct stat finds nothing, or another clip moved onto its path.
    folders that cannot be read leave their clips as they were. a path
    whose content key matches a row whose file is gone takes over that
    row, so a moved or renamed clip keeps its id, features and thumbnails,
    even when it lands on a path a row already holds.

    scan commits any transaction the caller left open on conn before it
    takes the write lock, and commits its own work when it is done."""
    if media_root is None:
        media_root = config.MEDIA_ROOT
    media_root = normalize_root(media_root)
    if not media_root.exists():
        raise FileNotFoundError(f"media root not found: {media_root}")
    if not media_root.is_dir():
        raise NotADirectoryError(f"media root is not a folder: {media_root}")
    root = str(media_root)
    tree = str(normalize_root(config.MEDIA_ROOT))
    stats = {"seen": 0, "new": 0, "moved": 0, "changed": 0, "available": 0,
             "evicted": 0, "oversize": 0, "missing": 0, "unreadable": 0}
    now = _now()
    # each row as it stood before the walk. a row another scan writes
    # while we walk is newer than our listing, so the listing cannot
    # say it is gone
    before = {r["id"]: r["last_seen"]
              for r in conn.execute("SELECT id, last_seen FROM clips")}

    # pass one: list the tree, before taking the write lock. a folder we
    # cannot list says nothing about the clips in it, so remember it
    # instead of losing them
    unreadable: list[str] = []
    unknown: set[str] = set()

    def on_error(exc: OSError) -> None:
        # no filename means we cannot tell which folder failed, so the
        # whole root counts as unread
        folder = os.fspath(exc.filename) if exc.filename else root
        logger.warning("cannot read folder %s: %s", folder, exc)
        unreadable.append(folder)

    def descend(dirpath: str, d: str) -> bool:
        if d.startswith("."):
            return False
        # walk never follows folder links, so say so instead of
        # dropping a linked country folder without a word
        if os.path.islink(os.path.join(dirpath, d)):
            logger.warning("skipping symlinked folder %s: scan does not"
                           " follow folder links", os.path.join(dirpath, d))
            return False
        return True

    found = []
    for dirpath, dirnames, filenames in os.walk(root, onerror=on_error):
        dirnames[:] = [d for d in dirnames if descend(dirpath, d)]
        for fname in sorted(filenames):
            if fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext not in config.VIDEO_EXTENSIONS:
                continue
            full = os.path.join(dirpath, fname)
            try:
                st = os.stat(full)
            except FileNotFoundError as exc:
                # a link whose target is gone is gone, so a cataloged
                # one goes missing; any other failed stat is unknown
                if os.path.islink(full):
                    logger.warning("skipping broken link %s: %s", full, exc)
                    continue
                logger.warning("stat failed for %s: %s", full, exc)
                unknown.add(full)
                continue
            except OSError as exc:
                logger.warning("stat failed for %s: %s", full, exc)
                unknown.add(full)
                continue
            found.append((full, fname, ext, st))
    stats["unreadable"] = len(unreadable) + len(unknown)

    # the rest holds the write lock and reads the known rows inside it,
    # so a scan that committed while we walked is seen, not collided
    # with, and a pass that fails leaves nothing half written
    if conn.in_transaction:
        conn.commit()
    conn.execute("BEGIN IMMEDIATE")
    try:
        _reconcile(conn, found, root, tree, now, stats, unreadable, unknown,
                   before)
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    return stats


def _reconcile(conn: sqlite3.Connection, found: list, root: str, tree: str,
               now: str, stats: dict, unreadable: list,
               unknown: set, before: dict) -> None:
    """passes two to four of scan, inside its transaction."""
    existing = {row["path"]: row for row in conn.execute(
        "SELECT id, path, rel_path, content_key, missing, last_seen"
        " FROM clips")}

    def write(row_id: int, full: str, rel: str, fname: str, ext: str,
              st: os.stat_result) -> None:
        # classification is redone every pass so profile edits in config
        # reach clips that are already cataloged
        profile, is_log, country = classify_path(Path(rel))
        conn.execute(
            "UPDATE clips SET path=?, rel_path=?, name=?, profile=?,"
            " is_log=?, country=?, ext=?, size_bytes=?, mtime_ns=?,"
            " content_key=?, available=?, oversize=?, missing=0, last_seen=?"
            " WHERE id=?",
            (full, rel, _nfc(fname), profile, is_log, country, ext,
             st.st_size, st.st_mtime_ns, _key(st),
             1 if is_materialized(st) else 0,
             1 if st.st_size > config.MAX_ANALYZE_BYTES else 0, now, row_id))

    # pass two: rows still at their path with the same content. any other
    # path is loose; a row whose path now holds other content is held,
    # since the clip it describes has left that path
    touched: set[int] = set()
    loose = []
    held: dict[str, sqlite3.Row] = {}
    for full, fname, ext, st in found:
        stats["seen"] += 1
        stats["available" if is_materialized(st) else "evicted"] += 1
        stats["oversize"] += 1 if st.st_size > config.MAX_ANALYZE_BYTES else 0
        row = existing.get(full)
        if row is not None and row["content_key"] == _key(st):
            write(row["id"], full, _rel(full, row, root, tree), fname, ext, st)
            touched.add(row["id"])
            continue
        if row is not None:
            held[full] = row
        loose.append((full, fname, ext, st))

    # pass three: each loose path takes over a moved row, keeps its own
    # row as changed, or becomes a new row
    by_key: dict[str, list] = {}
    for row in existing.values():
        if row["id"] not in touched:
            by_key.setdefault(row["content_key"], []).append(row)
    per_key: dict[str, int] = {}
    for _, _, _, st in loose:
        per_key[_key(st)] = per_key.get(_key(st), 0) + 1
    left = {row["id"] for row in held.values()}
    # an unmounted media root says nothing about the clips under it
    away = None if os.path.isdir(tree) else tree
    claims: dict[str, sqlite3.Row] = {}
    for full, _, _, st in loose:
        candidates = by_key.get(_key(st), [])
        row = _claim(full, st, candidates, per_key[_key(st)] == 1, left,
                     away)
        if row is not None:
            candidates.remove(row)
            claims[full] = row
    claimed = {row["id"] for row in claims.values()}
    # free the paths that change hands before anything moves in. a held
    # row nothing claims stays missing under a placeholder, which counts
    # as gone, so a later scan can still find its clip
    for full, row in held.items():
        if full in claims or row["id"] in claimed:
            conn.execute("UPDATE clips SET path=?, missing=1 WHERE id=?",
                         (f"replaced:{row['id']}:{full}", row["id"]))
            touched.add(row["id"])
            if row["id"] not in claimed and not row["missing"]:
                stats["missing"] += 1
    for full, fname, ext, st in loose:
        row = claims.get(full)
        if row is not None:
            stats["moved"] += 1
        elif full in held and held[full]["id"] not in claimed:
            row = held[full]
            stats["changed"] += 1
        if row is not None:
            write(row["id"], full, _rel(full, row, root, tree), fname, ext, st)
            touched.add(row["id"])
            continue
        key = _key(st)
        rel = _rel(full, None, root, tree)
        profile, is_log, country = classify_path(Path(rel))
        conn.execute(
            "INSERT INTO clips (path, rel_path, name, profile, is_log,"
            " country, ext, size_bytes, mtime_ns, content_key, available,"
            " oversize, missing, first_seen, last_seen)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)",
            (full, rel, _nfc(fname), profile, is_log, country, ext,
             st.st_size, st.st_mtime_ns, key, 1 if is_materialized(st) else 0,
             1 if st.st_size > config.MAX_ANALYZE_BYTES else 0, now, now))
        stats["new"] += 1

    # pass four: rows nothing touched. under the root the listing decides,
    # unless their folder could not be read; outside it a stat decides
    gone = []
    for path, row in existing.items():
        if row["id"] in touched or row["missing"]:
            continue
        if _clean(path) and _under(path, root):
            if path in unknown or any(_under(path, d) for d in unreadable):
                continue
            if before.get(row["id"]) != row["last_seen"]:
                continue
            gone.append((row["id"],))
        elif _whereabouts(path) == "gone":
            gone.append((row["id"],))
    conn.executemany("UPDATE clips SET missing=1 WHERE id=?", gone)
    stats["missing"] += len(gone)


# -- queries -----------------------------------------------------------

def overview(conn: sqlite3.Connection) -> dict:
    """headline counts plus a rollup by country for status displays.
    errors and every rollup cover clips still on disk; missing counts
    the clips a scan found gone, which keep their analysis."""
    v = config.FEATURE_VERSION
    totals = conn.execute(
        f"""SELECT COUNT(*) AS total,
                   SUM(c.available) AS available,
                   SUM(CASE WHEN c.available=0 THEN 1 ELSE 0 END) AS evicted,
                   SUM(c.oversize) AS oversize,
                   SUM(CASE WHEN f.clip_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed
            FROM clips c LEFT JOIN features f ON {_FRESH}
            WHERE c.missing=0""", (v,)).fetchone()
    # a deleted clip's last failure is not a failure in the library, and
    # one from an older extractor or older copy of the file is pending
    errors = conn.execute(
        f"SELECT COUNT(*) FROM clips c JOIN features f ON {_TRIED}"
        " WHERE f.error IS NOT NULL AND c.missing=0", (v,)).fetchone()[0]
    missing = conn.execute(
        "SELECT COUNT(*) FROM clips WHERE missing=1").fetchone()[0]
    countries = [dict(r) for r in conn.execute(
        f"""SELECT c.country,
                   COUNT(*) AS total,
                   SUM(c.available) AS available,
                   SUM(CASE WHEN f.clip_id IS NOT NULL THEN 1 ELSE 0 END) AS analyzed
            FROM clips c LEFT JOIN features f ON {_FRESH}
            WHERE c.missing=0
            GROUP BY c.country ORDER BY c.country""", (v,))]
    profiles = [dict(r) for r in conn.execute(
        """SELECT profile, COUNT(*) AS total, SUM(available) AS available
           FROM clips WHERE missing=0 GROUP BY profile ORDER BY profile""")]
    return {"totals": dict(totals), "errors": errors, "missing": missing,
            "countries": countries, "profiles": profiles}


def pending(conn: sqlite3.Connection, country: Optional[str] = None,
            limit: Optional[int] = None, force: bool = False,
            skip_failed: bool = False) -> list[sqlite3.Row]:
    """clips that can and should be analyzed: locally materialized, not
    oversize, and lacking a fresh feature row (unless force). clips never
    tried come first and earlier attempts follow, oldest first, so a few
    broken files cannot starve a limited run. skip_failed leaves out
    clips that already failed on this exact file. limit 0 is no clips,
    none is every clip, and a negative limit is an error."""
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be 0 or more, got {limit}")
    v = config.FEATURE_VERSION
    sql = (f"SELECT c.* FROM clips c LEFT JOIN features f ON {_TRIED}"
           " WHERE c.available=1 AND c.missing=0 AND c.oversize=0")
    params: list = [v]
    if not force:
        sql += (" AND (f.clip_id IS NULL OR f.error IS NOT NULL"
                " OR f.vector IS NULL)")
        if skip_failed:
            sql += " AND f.error IS NULL"
    if country:
        sql += _IN_COUNTRY
        params.append(country)
    sql += (" ORDER BY f.clip_id IS NOT NULL, f.analyzed_at,"
            " c.country, c.name")
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return conn.execute(sql, params).fetchall()


def known_failures(conn: sqlite3.Connection,
                   country: Optional[str] = None) -> int:
    """clips skip_failed leaves out: they failed on this exact file."""
    sql = (f"SELECT COUNT(*) FROM clips c JOIN features f ON {_TRIED}"
           " WHERE c.available=1 AND c.missing=0 AND c.oversize=0"
           " AND f.error IS NOT NULL")
    params: list = [config.FEATURE_VERSION]
    if country:
        sql += _IN_COUNTRY
        params.append(country)
    return conn.execute(sql, params).fetchone()[0]


def recheck(conn: sqlite3.Connection, row: sqlite3.Row) -> Optional[str]:
    """stat a pending clip again right before it is decoded, since the
    scan may be hours old. returns why it must not be opened now, or none.
    a clip evicted since the scan is flagged so it leaves pending."""
    try:
        st = os.stat(row["path"])
    except (FileNotFoundError, NotADirectoryError):
        return "gone from disk since the last scan, run scan"
    except OSError as exc:
        return f"cannot stat it: {exc}"
    if not is_materialized(st):
        conn.execute("UPDATE clips SET available=0 WHERE id=?", (row["id"],))
        conn.commit()
        return "icloud evicted since the last scan"
    if _key(st) != row["content_key"]:
        return "changed on disk since the last scan, run scan"
    return None


def analyzed(conn: sqlite3.Connection,
             country: Optional[str] = None) -> list[sqlite3.Row]:
    """clips with a fresh feature vector, joined with the vector itself."""
    v = config.FEATURE_VERSION
    sql = (f"SELECT c.*, f.vector AS vector, f.summary AS summary"
           f" FROM clips c JOIN features f ON {_FRESH}"
           " WHERE c.missing=0")
    params: list = [v]
    if country:
        sql += _IN_COUNTRY
        params.append(country)
    sql += " ORDER BY c.id"
    return conn.execute(sql, params).fetchall()


def save_features(conn: sqlite3.Connection, clip_id: int, content_key: str,
                  vector: Optional[bytes], summary: Optional[str],
                  error: Optional[str] = None,
                  is_log: Optional[int] = None) -> None:
    """store one analysis result. is_log is the flag the analysis ran
    with; left out, it is the clip's current flag."""
    conn.execute(
        "INSERT OR REPLACE INTO features"
        " (clip_id, version, content_key, vector, summary, error, analyzed_at,"
        " is_log)"
        " VALUES (?,?,?,?,?,?,?, COALESCE(?, (SELECT is_log FROM clips"
        " WHERE id=?)))",
        (clip_id, config.FEATURE_VERSION, content_key, vector, summary,
         error, _now(), is_log, clip_id))
    conn.commit()


def get_clip(conn: sqlite3.Connection, clip_id: int) -> Optional[sqlite3.Row]:
    return conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()


def find_clip(conn: sqlite3.Connection, token: str) -> Optional[sqlite3.Row]:
    """resolve a cli argument: numeric id first, then name substring.
    an empty token names no clip, and % and _ match only themselves."""
    token = _nfc(token.strip())
    if not token:
        return None
    # ascii digits only: a superscript passes isdigit but not int
    if token.isascii() and token.isdigit():
        try:
            row = get_clip(conn, int(token))
        except OverflowError:
            # past sqlite's integer range, so no id can match
            row = None
        if row:
            return row
    pattern = (token.replace("\\", "\\\\").replace("%", "\\%")
               .replace("_", "\\_"))
    return conn.execute(
        "SELECT * FROM clips WHERE name LIKE ? ESCAPE '\\' AND missing=0"
        " ORDER BY id LIMIT 1", (f"%{pattern}%",)).fetchone()
