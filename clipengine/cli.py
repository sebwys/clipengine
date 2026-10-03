# cli.py
# entry point. argparse subparsers, one per verb; main returns the exit
# code. heavy imports (cv2, numpy paths) stay inside the commands that
# need them so `status` and `help` start instantly.

import argparse
import functools
import json
import sys
import time
import unicodedata

from clipengine import catalog, config

HELP = """clipengine: local footage transition matching engine

usage: python -m clipengine <command>

commands:
  scan [--root=PATH]          stat only walk of the footage tree, or one
                              folder of it, into the catalog. never opens
                              files, never downloads.
  status                      catalog counts by country and profile
  analyze [--limit=N] [--country=X] [--force] [--skip-failed]
                              extract features from materialized clips.
                              --limit counts clips opened. clips that
                              failed before go last; --skip-failed leaves
                              out those whose file has not changed
  relabel                     recompute motion labels from stored vectors
                              after threshold changes; no video decoding
  audit                       tuning dashboard: class distribution, energy
                              percentiles vs thresholds, per country vibe
  match <id|name> [--mode=M] [--n=10] [--country=any|same|different]
                              rank transition candidates out of one clip
  sequence --seed=<id|name> [--length=8] [--mode=M]
           [--country-mode=any|same|travel] [--no-export]
                              build a best chain and export json, m3u8
                              and fcpxml
  ui [--port=8763]            launch the local web ui (127.0.0.1 only)

modes: momentum (carry motion through the cut), whip (blur to blur),
calm (still to still, color led), contrast (deliberate vibe flip)

options take --name=value or --name value, spelled out in full.
<command> --help, help <command> or --help <command> shows one
command. exit status: 0 ok, 1 when the command failed, 2 for a usage
error.
"""


PROG = "python -m clipengine"


def _whole(text: str, low: int, high=None) -> int:
    # argparse turns this error into a usage error naming the option
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a whole number: '{text}'")
    if value < low or (high is not None and value > high):
        span = f"{low} to {high}" if high is not None else f"{low} or more"
        raise argparse.ArgumentTypeError(f"must be {span}, got {value}")
    return value


def _limit(text: str):
    # an empty --limit= means no limit
    return None if text == "" else _whole(text, 0)


def _count(text: str) -> int:
    return _whole(text, 1)


def _length(text: str) -> int:
    # a chain needs a cut, so one clip is not a sequence
    return _whole(text, 2)


def _port(text: str) -> int:
    return _whole(text, 1, 65535)


def build_parser() -> argparse.ArgumentParser:
    """one subparser per verb. both --name=value and --name value work,
    unknown options and bad values exit 2 before anything runs, and
    each verb answers -h and --help on its own."""
    modes = list(config.SCORING_MODES)
    parser = argparse.ArgumentParser(prog=PROG, allow_abbrev=False,
                                     add_help=False)
    verbs = parser.add_subparsers(dest="command", required=True,
                                  metavar="<command>")
    # kept so a stray word is reported with its own command's usage
    parser.verbs = {}

    def verb(name: str, text: str) -> argparse.ArgumentParser:
        parser.verbs[name] = verbs.add_parser(
            name, help=text, description=text, allow_abbrev=False)
        return parser.verbs[name]

    p = verb("scan", "stat only walk of the footage tree, or one folder of"
             " it, into the catalog. never opens files, never downloads.")
    p.add_argument("--root", metavar="PATH",
                   help="folder to scan (default: the media root)")
    verb("status", "catalog counts by country and profile")
    p = verb("analyze", "extract features from materialized clips")
    p.add_argument("--limit", type=_limit, metavar="N",
                   help="clips to open at most; 0 opens none")
    p.add_argument("--country", help="only clips from this country")
    p.add_argument("--force", action="store_true",
                   help="analyze again even when features are fresh")
    p.add_argument("--skip-failed", action="store_true",
                   help="leave out clips that failed before and have not"
                   " changed")
    verb("relabel", "recompute motion labels from stored vectors after"
         " threshold changes; no video decoding")
    verb("audit", "tuning dashboard: class distribution, energy"
         " percentiles vs thresholds, per country vibe")
    p = verb("match", "rank transition candidates out of one clip")
    p.add_argument("clip", metavar="<id|name>")
    p.add_argument("--mode", choices=modes, default=config.DEFAULT_MODE)
    p.add_argument("--n", type=_count, default=10, metavar="N")
    p.add_argument("--country", choices=["any", "same", "different"],
                   default="any")
    p = verb("sequence", "build a best chain and export json, m3u8 and"
             " fcpxml")
    p.add_argument("--seed", required=True, metavar="<id|name>")
    p.add_argument("--length", type=_length, default=8, metavar="N")
    p.add_argument("--mode", choices=modes, default=config.DEFAULT_MODE)
    p.add_argument("--country-mode", choices=["any", "same", "travel"],
                   default="any")
    p.add_argument("--no-export", action="store_true",
                   help="print the chain without writing files")
    p = verb("ui", "launch the local web ui (127.0.0.1 only)")
    p.add_argument("--port", type=_port, default=config.SERVER_PORT)
    return parser


def _parse(args: list) -> argparse.Namespace:
    """the command line as a namespace. a usage error exits 2 the way
    argparse does, and words the command does not take are reported
    under that command's usage, not the top one."""
    parser = build_parser()
    ns, extra = parser.parse_known_args(args)
    if extra:
        parser.verbs[ns.command].error(
            f"unrecognized arguments: {' '.join(extra)}")
    return ns


def _verb(fn):
    """lets a command also take its option list, parsed by the same
    subparser as the command line."""
    name = fn.__name__[len("cmd_"):]

    @functools.wraps(fn)
    def wrapper(args):
        if not isinstance(args, argparse.Namespace):
            args = _parse([name, *args])
        return fn(args)
    return wrapper


def _summary(text):
    """the stored summary as a dict, or None when it is not readable
    json or not an object. a null summary, sql null or the json text
    null, reads as an empty one."""
    try:
        value = json.loads(text or "{}")
    except ValueError:
        return None
    if value is None:
        return {}
    return value if isinstance(value, dict) else None


def _find(conn, token: str, role: str):
    """the clip named on the command line, or None after saying why.
    a clip gone from disk says so instead of asking for analyze, by id
    or by name."""
    row = catalog.find_clip(conn, token) or _find_missing(conn, token)
    if not row:
        print(f"no clip matches '{token}'")
    elif row["missing"]:
        print(f"{role} #{row['id']} ({row['name']}) is missing on disk;"
              " rescan or restore it")
        return None
    return row


def _find_missing(conn, token: str):
    """a clip gone from disk whose name holds the token. find_clip only
    looks at names on disk, so this is asked only after it found none.
    the token is matched the way find_clip matches it."""
    token = unicodedata.normalize("NFC", token.strip())
    if not token:
        return None
    pattern = (token.replace("\\", "\\\\").replace("%", "\\%")
               .replace("_", "\\_"))
    return conn.execute(
        "SELECT * FROM clips WHERE name LIKE ? ESCAPE '\\' AND missing=1"
        " ORDER BY id LIMIT 1", (f"%{pattern}%",)).fetchone()


def _print_countries(conn) -> None:
    ov = catalog.overview(conn)
    print(f"\n  {'country':<24} {'clips':>6} {'local':>6} {'analyzed':>9}")
    for c in ov["countries"]:
        print(f"  {c['country']:<24} {c['total']:>6}"
              f" {c['available'] or 0:>6} {c['analyzed'] or 0:>9}")


@_verb
def cmd_scan(args: argparse.Namespace) -> int:
    conn = catalog.connect()
    root = args.root
    media_root = catalog.normalize_root(root if root else config.MEDIA_ROOT)
    try:
        stats = catalog.scan(conn, media_root)
    except (FileNotFoundError, NotADirectoryError) as exc:
        conn.close()
        print(f"scan: {exc}")
        # cmd_scan exits instead of returning
        sys.exit(1)
    print(f"scan of {media_root}")
    print(f"  seen {stats['seen']} videos | new {stats['new']}"
          f" | moved {stats['moved']} | changed {stats['changed']}"
          f" | gone missing {stats['missing']}")
    print(f"  materialized {stats['available']}"
          f" | icloud evicted {stats['evicted']}"
          f" | oversize skipped {stats['oversize']}")
    if stats["unreadable"]:
        print(f"  unreadable {stats['unreadable']}: their clips were left"
              " as they were, see the warnings above")
    elif not stats["seen"]:
        print("  no videos found under this root")
    _print_countries(conn)
    conn.close()
    return 0


@_verb
def cmd_status(args: argparse.Namespace) -> int:
    conn = catalog.connect()
    ov = catalog.overview(conn)
    t = ov["totals"]
    # an overview without the missing count prints without it
    missing = ov.get("missing")
    gone = "" if missing is None else f" | {missing or 0} missing"
    print(f"catalog: {t['total'] or 0} clips | {t['available'] or 0} local"
          f" | {t['evicted'] or 0} evicted{gone}"
          f" | {t['analyzed'] or 0} analyzed | {ov['errors']} errors")
    for p in ov["profiles"]:
        print(f"  {p['profile']:<20} {p['total']:>5} clips"
              f" ({p['available'] or 0} local)")
    _print_countries(conn)
    conn.close()
    return 0


@_verb
def cmd_analyze(args: argparse.Namespace) -> int:
    from clipengine import analysis, features
    conn = catalog.connect()
    config.create_directories()
    limit, country = args.limit, args.country
    force, skip = args.force, args.skip_failed
    if country and not conn.execute(
            "SELECT 1 FROM clips WHERE missing=0 AND fold(country)=fold(?)"
            " LIMIT 1", (country,)).fetchone():
        known = [c["country"] for c in catalog.overview(conn)["countries"]]
        print(f"no clips from country '{country}'; known: "
              + (", ".join(known) or "none, run scan first"))
        conn.close()
        return 1
    if limit == 0:
        print("limit 0: nothing analyzed")
        conn.close()
        return 0
    # the limit is applied below, not in the query, so clips skipped
    # without being opened do not use it up
    rows = catalog.pending(conn, country=country, force=force,
                           skip_failed=skip)
    known = catalog.known_failures(conn, country) if skip and not force else 0
    if known:
        print(f"skipping {known} clip(s) that failed before and have not"
              " changed")
    if not rows:
        print("nothing else to analyze" if known else
              "nothing to analyze: every available clip has fresh features")
        conn.close()
        return 0
    total = min(len(rows), limit) if limit is not None else len(rows)
    print(f"analyzing {total} clip(s)")
    done = failed = skipped = 0
    t0 = time.time()
    for row in rows:
        if done + failed == total:
            break
        name = f"{row['country']}/{row['name']}"
        # stat again first: opening a clip evicted since the scan would
        # pull it back from icloud
        reason = catalog.recheck(conn, row)
        if reason:
            print(f"  {name}: skipped ({reason})")
            skipped += 1
            continue
        label = f"[{done + failed + 1}/{total}] {name}"
        t1 = time.time()
        try:
            result = analysis.analyze_clip(row["path"], bool(row["is_log"]))
            for pos, blob in result.thumbs.items():
                (config.THUMB_DIR / f"{row['id']}_{pos}.jpg").write_bytes(blob)
            catalog.save_features(conn, row["id"], row["content_key"],
                                  features.to_bytes(result.vector),
                                  json.dumps(result.summary),
                                  is_log=row["is_log"])
            s = result.summary
            print(f"{label}: {s['start_class']} -> {s['end_class']}"
                  f" | {s['duration_s']}s | {time.time() - t1:.1f}s")
            done += 1
        except Exception as exc:
            catalog.save_features(conn, row["id"], row["content_key"],
                                  None, None, error=str(exc),
                                  is_log=row["is_log"])
            print(f"{label}: FAILED ({exc})")
            failed += 1
    tail = f", {skipped} skipped" if skipped else ""
    print(f"done: {done} analyzed, {failed} failed{tail},"
          f" {time.time() - t0:.0f}s total")
    conn.close()
    # any failed clip makes the run a failure for scripts and cron
    return 1 if failed else 0


@_verb
def cmd_relabel(args: argparse.Namespace) -> int:
    from clipengine import analysis, features
    conn = catalog.connect()
    rows = conn.execute(
        "SELECT f.clip_id, f.vector, f.summary, c.name FROM features f"
        " JOIN clips c ON c.id = f.clip_id"
        " WHERE f.vector IS NOT NULL").fetchall()
    changed = 0
    unreadable = []
    for row in rows:
        try:
            vec = features.from_bytes(row["vector"])
        except ValueError:
            continue
        summary = _summary(row["summary"])
        if summary is None:
            # left as stored so analyze --force can rebuild it
            unreadable.append(row["name"])
            continue
        start = analysis.motion_from_vector(vec, "start")
        end = analysis.motion_from_vector(vec, "end")
        if (summary.get("start_class") == start.label
                and summary.get("end_class") == end.label):
            continue
        summary["start_class"] = start.label
        summary["end_class"] = end.label
        if isinstance(summary.get("start"), dict):
            summary["start"]["class"] = start.label
        if isinstance(summary.get("end"), dict):
            summary["end"]["class"] = end.label
        conn.execute("UPDATE features SET summary=? WHERE clip_id=?",
                     (json.dumps(summary), row["clip_id"]))
        changed += 1
    conn.commit()
    print(f"relabeled {changed} of {len(rows)} analyzed clips")
    conn.close()
    if unreadable:
        print(f"skipped {len(unreadable)} with an unreadable summary"
              f" (analyze --force rebuilds them): {', '.join(unreadable)}")
        return 1
    return 0


@_verb
def cmd_audit(args: argparse.Namespace) -> int:
    """distribution report over analyzed clips. the tuning dashboard:
    run after big analyze passes or threshold changes and check that the
    thresholds carve the real distribution at sensible points."""
    from collections import Counter

    import numpy as np

    from clipengine import features, matching
    conn = catalog.connect()
    lib = matching.load_library(conn)
    if not len(lib):
        print("nothing analyzed yet")
        conn.close()
        return 0
    F, idx, n = lib.F, features.INDEX, len(lib)
    print(f"audit over {n} analyzed clips")

    classes = Counter()
    for m in lib.meta:
        classes[m["start_class"]] += 1
        classes[m["end_class"]] += 1
    print("\nclass distribution (start + end windows):")
    for name, cnt in classes.most_common():
        bar = "#" * max(1, round(40 * cnt / (2 * n)))
        print(f"  {str(name):<14} {cnt:>4}  {bar}")

    energy = np.concatenate([F[:, idx["start_energy"]],
                             F[:, idx["end_energy"]]])
    pct = np.percentile(energy, [10, 25, 50, 75, 90, 99])
    print("\nwindow energy, widths/sec (p10 p25 p50 p75 p90 p99):")
    print("  " + "  ".join(f"{v:.3f}" for v in pct))
    print(f"  thresholds: static < {config.STATIC_MAX}"
          f" | pan >= {config.PAN_MIN} | whip >= {config.WHIP_MIN}")

    tx = np.concatenate([F[:, idx["start_flow_x"]], F[:, idx["end_flow_x"]]])
    ty = np.concatenate([F[:, idx["start_flow_y"]], F[:, idx["end_flow_y"]]])
    jit = np.concatenate([F[:, idx["start_jitter"]], F[:, idx["end_jitter"]]])
    t = np.hypot(tx, ty)
    eligible = t >= config.PAN_MIN
    ratio = t[eligible] / (jit[eligible] + 1e-6)
    steady = ratio >= config.STEADY_RATIO
    near = ((ratio >= config.STEADY_RATIO * 0.75)
            & (ratio <= config.STEADY_RATIO * 1.25))
    print(f"\nsteadiness (windows with pan level translation):"
          f" {int(eligible.sum())} eligible,"
          f" {int(steady.sum())} steady, {int((~steady).sum())} handheld,"
          f" {int(near.sum())} near the boundary")

    flat = 0
    for row in catalog.analyzed(conn):
        summary = _summary(row["summary"]) or {}
        flat += 1 if summary.get("flat") else 0
    print(f"\nflat/log detected: {flat}/{n} clips"
          " (expect nearly all: every profile shoots log)")

    print("\nper country vibe (mean luma / sat / warmth, count):")
    by_country: dict = {}
    for i, m in enumerate(lib.meta):
        by_country.setdefault(m["country"], []).append(i)
    for country in sorted(by_country):
        rows = by_country[country]
        luma = float(F[rows, idx["global_luma"]].mean())
        sat = float(F[rows, idx["global_sat"]].mean())
        warm = float(F[rows, idx["global_warmth"]].mean())
        print(f"  {country:<22} luma {luma:5.1f}  sat {sat:5.1f}"
              f"  warmth {warm:+5.1f}  ({len(rows)})")

    # the same failures status counts: clips on disk, current extractor
    errors = conn.execute(
        f"SELECT c.name, f.error FROM clips c JOIN features f"
        f" ON {catalog._TRIED} WHERE f.error IS NOT NULL AND c.missing=0"
        " ORDER BY c.id", (config.FEATURE_VERSION,)).fetchall()
    if errors:
        print(f"\n{len(errors)} analysis errors:")
        for e in errors[:10]:
            print(f"  {e['name']}: {e['error']}")
    else:
        print("\nno analysis errors")
    conn.close()
    return 0


@_verb
def cmd_match(args: argparse.Namespace) -> int:
    from clipengine import matching
    conn = catalog.connect()
    row = _find(conn, args.clip, "clip")
    if not row:
        conn.close()
        return 1
    mode = args.mode
    lib = matching.load_library(conn)
    if row["id"] not in lib.row_of:
        print(f"clip #{row['id']} ({row['name']}) has no features yet;"
              " run analyze first")
        conn.close()
        return 1
    a = lib.meta[lib.row_of[row["id"]]]
    print(f"cuts out of #{row['id']} {row['country']}/{row['name']}"
          f" (end: {a['end_class']}) mode={mode}")
    results = matching.rank(lib, row["id"], mode, n=args.n,
                            country=args.country)
    if not results:
        print("no positive score candidates under this mode's gate")
        conn.close()
        return 0
    for r_i, r in enumerate(results, 1):
        b = r["breakdown"]
        print(f"{r_i:>2}. {r['score']:.3f}  #{r['id']:<4}"
              f" {r['country']}/{r['name']}  starts:{r['start_class']}"
              f"  (motion {b['motion']}, energy {b['energy']},"
              f" color {b['color']}, gate {b['gate']})")
    conn.close()
    return 0


@_verb
def cmd_sequence(args: argparse.Namespace) -> int:
    from clipengine import matching, sequence
    seed_token = args.seed
    conn = catalog.connect()
    row = _find(conn, seed_token, "seed")
    if not row:
        conn.close()
        return 1
    lib = matching.load_library(conn)
    if row["id"] not in lib.row_of:
        print(f"seed #{row['id']} has no features yet; run analyze first")
        conn.close()
        return 1
    mode = args.mode
    matrix = matching.full_matrix(lib, mode)
    countries = [m["country"] for m in lib.meta]
    rows_idx, edges = sequence.build_chain(
        matrix, lib.row_of[row["id"]],
        length=args.length,
        countries=countries,
        country_mode=args.country_mode)
    if len(rows_idx) < 2:
        # the same rule as the ui and server: one clip is not a sequence
        print(f"no cut out of seed #{row['id']} ({row['name']}) passes"
              f" the {mode} gate; a sequence needs at least two clips,"
              " so nothing was exported")
        conn.close()
        return 1
    meta = [lib.meta[i] for i in rows_idx]
    print(f"sequence ({mode}), total score {sum(edges):.3f}")
    for i, m in enumerate(meta):
        cut = "  seed" if i == 0 else f"{edges[i - 1]:.3f}"
        print(f" {i + 1:>2}. [{cut:>6}] #{m['id']:<4} {m['country']}/{m['name']}"
              f" ({m['start_class']} -> {m['end_class']})")
    if not args.no_export:
        json_path, m3u_path, xml_path = sequence.export_chain(
            meta, edges, mode)
        print(f"exported: {json_path}")
        print(f"          {m3u_path}")
        print(f"          {xml_path}  (import into fcp or resolve)")
    conn.close()
    return 0


@_verb
def cmd_ui(args: argparse.Namespace) -> int:
    from clipengine.web import server
    try:
        server.serve(args.port)
    except OSError as exc:
        # most often the port is taken by another ui
        print(f"ui: cannot serve on port {args.port}:"
              f" {exc.strerror or exc}")
        return 1
    return 0


COMMANDS = {"scan": cmd_scan, "status": cmd_status, "analyze": cmd_analyze,
            "relabel": cmd_relabel, "audit": cmd_audit, "match": cmd_match,
            "sequence": cmd_sequence, "ui": cmd_ui}


def main(argv=None) -> int:
    """run one verb and return its exit code: 0 ok, 1 when the command
    failed (a clip failed analysis, a clip or root was not found), 2 for
    a usage error, which prints before anything runs."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args in (["help"], ["-h"], ["--help"]):
        print(HELP)
        return 0
    try:
        if args[0] in ("help", "-h", "--help"):
            # help <verb>, -h <verb> and --help <verb> are the verb's --help
            if len(args) > 2:
                build_parser().error("help takes one command, got: "
                                     + " ".join(args[1:]))
            args = [args[1], "--help"]
        ns = _parse(args)
    except SystemExit as exc:
        # argparse exits 0 after --help and 2 after a usage error
        code = exc.code
        return code if isinstance(code, int) else int(code is not None)
    try:
        return COMMANDS[ns.command](ns)
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else int(code is not None)
