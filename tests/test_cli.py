# test_cli.py
# the command line itself: both option forms, help per command, values
# checked before any output, exit codes, and what match and sequence say
# about a clip that is missing, unknown or alone. analysis is stubbed and
# vectors are built by hand, so nothing here decodes video. one test runs
# python -m clipengine for real to check the exit status reaches the shell.

import contextlib
import io
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from typing import NamedTuple
from unittest import mock

from clipengine import analysis, catalog, cli, config, features, matching
from clipengine.web import server
from tests.test_matching import make_vec
from tests.util import TempDirsMixin

# every window pans right, so momentum links any two of these clips and
# the calm gate shuts every cut out of them
MOVING = dict(start_flow_x=-0.5, start_energy=0.5,
              end_flow_x=-0.5, end_energy=0.5)
PAN_SUMMARY = {"start_class": "pan_right", "end_class": "pan_right",
               "duration_s": 5.0}


class Run(NamedTuple):
    status: int     # what the process would exit with
    out: str
    err: str
    leaked: str     # an exception that escaped main, as a traceback would


def run(argv) -> Run:
    out, err = io.StringIO(), io.StringIO()
    leaked = ""
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            returned = cli.main(list(argv))
            status = returned if isinstance(returned, int) else 0
        except SystemExit as exc:
            code = exc.code
            status = code if isinstance(code, int) else int(code is not None)
        except Exception as exc:
            status, leaked = 1, f"{type(exc).__name__}: {exc}"
    return Run(status, out.getvalue(), err.getvalue(), leaked)


def touch(path: Path, size: int = 4096) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"v" * size)


def ranked_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln[:3].strip().rstrip(".")
            .isdigit() and "  #" in ln]


class CliCase(TempDirsMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.media = config.MEDIA_ROOT

    def scan(self) -> None:
        conn = catalog.connect()
        catalog.scan(conn, self.media)
        conn.close()

    def exports(self) -> list[str]:
        return sorted(p.name for p in config.EXPORT_DIR.iterdir())


class LibraryCase(CliCase):
    """three analyzed clips that all pan right: two in japan, one in italy."""

    def setUp(self):
        super().setUp()
        for rel in ("Sony SLOG-3/Japan/a.mp4", "Sony SLOG-3/Japan/b.mp4",
                    "DJI DLOG-M/Italy/c.mp4"):
            touch(self.media / rel)
        conn = catalog.connect()
        catalog.scan(conn, self.media)
        blob = features.to_bytes(make_vec(**MOVING))
        self.ids = {}
        for r in conn.execute("SELECT id, name, content_key FROM clips"):
            catalog.save_features(conn, r["id"], r["content_key"], blob,
                                  json.dumps(PAN_SUMMARY))
            self.ids[r["name"]] = r["id"]
        conn.close()


class AnalyzeCase(CliCase):
    """seven clips waiting for analysis, four in japan and three in italy.
    analyze_clip is a stub that records each call."""

    def setUp(self):
        super().setUp()
        for i in range(4):
            touch(self.media / "Sony SLOG-3" / "Japan" / f"C000{i}.MP4")
        for i in range(3):
            touch(self.media / "DJI DLOG-M" / "Italy" / f"DJI_000{i}.MP4")
        self.scan()

    def analyze(self, argv, fail=()):
        calls = []

        def fake(path, is_log):
            name = Path(path).name
            calls.append(name)
            if name in fail or "*" in fail:
                raise RuntimeError("stub decoder could not open it")
            return types.SimpleNamespace(
                thumbs={}, vector=make_vec(),
                summary={"start_class": "static", "end_class": "static",
                         "duration_s": 5.0})

        with mock.patch.object(analysis, "analyze_clip", fake):
            result = run(argv)
        return result, calls


# -- everyday commands -----------------------------------------------------

class TestHelpAndCommands(CliCase):
    def test_help_lists_every_command_and_exits_zero(self):
        for argv in ([], ["help"], ["-h"], ["--help"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.leaked), (0, ""))
                for verb in ("scan", "status", "analyze", "relabel",
                             "audit", "match", "sequence", "ui"):
                    self.assertIn(verb, r.out)

    def test_help_flag_before_a_command_shows_that_command(self):
        # these once printed the full help; the one command is closer
        for argv in (["--help", "status"], ["-h", "analyze"],
                     ["--help", "match"], ["help", "match"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.leaked, r.err), (0, "", ""))
                self.assertIn(f"usage: {cli.PROG} {argv[1]}", r.out)
                self.assertNotIn("analyzing", r.out)

    def test_help_with_more_than_one_command_is_a_usage_error(self):
        for argv in (["help", "match", "x"], ["--help", "match", "x"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
                self.assertIn("match x", r.err)

    def test_main_returns_the_status_instead_of_raising(self):
        # run() hides the difference, so call main directly
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["bogus"]), 2)
            self.assertEqual(cli.main(["status", "--help"]), 0)
            self.assertEqual(
                cli.main(["scan", f"--root={self.tmp / 'nope'}"]), 1)

    def test_unknown_command_exits_two_and_names_it(self):
        r = run(["bogus"])
        self.assertEqual((r.status, r.leaked), (2, ""))
        self.assertIn("bogus", r.out + r.err)

    def test_scan_and_status_print_their_counts(self):
        for rel in ("Sony SLOG-3/Japan/a.mp4", "DJI DLOG-M/Italy/b.mp4",
                    "DJI DLOG-M/Italy/c.mp4"):
            touch(self.media / rel)
        r = run(["scan", f"--root={self.media}"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn(f"scan of {self.media}", r.out)
        self.assertIn("seen 3 videos", r.out)
        self.assertIn("new 3", r.out)
        r = run(["status"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("catalog: 3 clips | 3 local", r.out)
        self.assertIn("0 analyzed", r.out)
        self.assertIn("Italy", r.out)

    def test_status_without_a_missing_key_still_prints(self):
        real = catalog.overview

        def without_missing(conn):
            ov = real(conn)
            ov.pop("missing", None)
            ov["totals"].pop("missing", None)
            return ov

        with mock.patch.object(catalog, "overview", without_missing):
            r = run(["status"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("catalog: 0 clips", r.out)
        self.assertNotIn("missing", r.out)


class TestEqualsFormStillWorks(AnalyzeCase):
    def test_analyze_limit_country_and_force(self):
        r, calls = self.analyze(["analyze", "--country=Italy"])
        self.assertEqual((r.status, sorted(calls)),
                         (0, ["DJI_0000.MP4", "DJI_0001.MP4", "DJI_0002.MP4"]))
        r, calls = self.analyze(["analyze", "--limit=1"])
        self.assertEqual((r.status, calls), (0, ["C0000.MP4"]))
        r, calls = self.analyze(["analyze"])
        self.assertEqual((r.status, len(calls)), (0, 3))
        r, calls = self.analyze(["analyze"])
        self.assertEqual((r.status, calls), (0, []))
        self.assertIn("nothing to analyze", r.out)
        r, calls = self.analyze(["analyze", "--force"])
        self.assertEqual((r.status, len(calls)), (0, 7))

    def test_ui_port(self):
        with mock.patch.object(server, "serve") as serve:
            self.assertEqual(run(["ui", "--port=8764"]).status, 0)
            self.assertEqual(run(["ui"]).status, 0)
        self.assertEqual(serve.call_args_list,
                         [mock.call(8764), mock.call(config.SERVER_PORT)])


class TestMatchAndSequenceOutput(LibraryCase):
    def test_match_ranks_cuts_with_breakdowns(self):
        r = run(["match", "a.mp4"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn(f"cuts out of #{self.ids['a.mp4']} Japan/a.mp4"
                      " (end: pan_right) mode=momentum", r.out)
        lines = ranked_lines(r.out)
        self.assertEqual(len(lines), 2)
        self.assertIn("motion", lines[0])
        self.assertIn("gate", lines[0])

    def test_match_equals_options(self):
        r = run(["match", str(self.ids["a.mp4"]), "--mode=whip"])
        self.assertIn("mode=whip", r.out)
        r = run(["match", "a.mp4", "--n=1"])
        self.assertEqual(len(ranked_lines(r.out)), 1)
        r = run(["match", "a.mp4", "--country=different"])
        lines = ranked_lines(r.out)
        self.assertEqual(len(lines), 1)
        self.assertIn("Italy/c.mp4", lines[0])

    def test_sequence_prints_and_exports_the_chain(self):
        r = run(["sequence", "--seed=a.mp4", "--length=2", "--mode=momentum",
                 "--country-mode=any", "--no-export"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("sequence (momentum)", r.out)
        self.assertIn("seed] #", r.out)
        self.assertEqual(self.exports(), [])
        r = run(["sequence", "--seed=a.mp4"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        names = self.exports()
        self.assertEqual([Path(n).suffix for n in names],
                         [".fcpxml", ".json", ".m3u8"])
        plan = json.loads((config.EXPORT_DIR / names[1]).read_text())
        self.assertEqual(len(plan["clips"]), 3)


# -- option syntax ---------------------------------------------------------

class TestOptionSyntax(AnalyzeCase):
    def test_space_separated_values_are_honored(self):
        r, calls = self.analyze(["analyze", "--country", "Italy"])
        self.assertEqual(sorted(calls),
                         ["DJI_0000.MP4", "DJI_0001.MP4", "DJI_0002.MP4"])
        r, calls = self.analyze(["analyze", "--limit", "1"])
        self.assertEqual((r.status, len(calls)), (0, 1),
                         f"--limit 1 opened {len(calls)} clips")
        with mock.patch.object(server, "serve") as serve:
            run(["ui", "--port", "8764"])
        self.assertEqual(serve.call_args, mock.call(8764))

    def test_unknown_or_misspelled_options_are_rejected(self):
        for argv in (["analyze", "--contry=Italy"], ["analyze", "--forse"],
                     ["analyze", "extra"], ["status", "--verbose"]):
            with self.subTest(argv=argv):
                r, calls = self.analyze(argv)
                self.assertEqual((r.status, calls, r.leaked), (2, [], ""))
                self.assertIn(argv[1], r.err)

    def test_verb_help_prints_help_and_runs_nothing(self):
        for argv in (["analyze", "--help"], ["analyze", "-h"],
                     ["help", "analyze"]):
            with self.subTest(argv=argv):
                r, calls = self.analyze(argv)
                self.assertEqual((r.status, calls), (0, []))
                self.assertIn("usage: python -m clipengine analyze", r.out)
                self.assertIn("--skip-failed", r.out)
        touch(self.media / "Sony SLOG-3" / "Japan" / "late.MP4")
        r = run(["scan", "--help"])
        self.assertEqual(r.status, 0)
        self.assertNotIn("seen", r.out)
        conn = catalog.connect()
        self.addCleanup(conn.close)
        self.assertIsNone(catalog.find_clip(conn, "late"))
        with mock.patch.object(server, "serve") as serve:
            self.assertEqual(run(["ui", "--help"]).status, 0)
        serve.assert_not_called()


class TestMatchOptionSyntax(LibraryCase):
    def test_space_separated_match_and_sequence_options(self):
        r = run(["match", "a.mp4", "--mode", "whip"])
        self.assertIn("mode=whip", r.out)
        r = run(["match", "--n", "1", "a.mp4"])
        self.assertEqual(len(ranked_lines(r.out)), 1, r.out)
        r = run(["sequence", "--seed", "a.mp4", "--length", "2",
                 "--no-export"])
        self.assertEqual(r.status, 0, r.out + r.err)
        self.assertIn("2. [", r.out)
        self.assertNotIn("3. [", r.out)

    def test_misspelled_flag_exports_nothing(self):
        r = run(["sequence", "--seed=a.mp4", "--noexport"])
        self.assertEqual((r.status, r.out), (2, ""))
        self.assertEqual(self.exports(), [])

    def test_shortened_option_names_are_rejected(self):
        # a prefix that happens to fit today could name another option later
        for argv in (["sequence", "--seed=a.mp4", "--no-exp"],
                     ["sequence", "--seed=a.mp4", "--len=2"],
                     ["match", "a.mp4", "--mod=whip"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
                self.assertIn(argv[-1].split("=")[0], r.err)
        self.assertEqual(self.exports(), [])

    def test_a_stray_word_shows_the_usage_of_its_command(self):
        for argv in (["sequence", "--seed=a.mp4", "--bogus"],
                     ["status", "extra"], ["match", "a.mp4", "b.mp4"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
                self.assertIn(f"usage: {cli.PROG} {argv[0]}", r.err)
                self.assertIn(argv[-1], r.err)
        self.assertEqual(self.exports(), [])


# -- values checked before anything runs -----------------------------------

class TestValuesCheckedFirst(LibraryCase):
    def test_bad_values_print_nothing_and_exit_two(self):
        a = str(self.ids["a.mp4"])
        for argv in (["match", a, "--n=0"], ["match", a, "--n=-1"],
                     ["match", a, "--n=abc"], ["match", a, "--country=Japan"],
                     ["sequence", f"--seed={a}", "--length=0"],
                     ["sequence", f"--seed={a}", "--length=-3"],
                     ["sequence", f"--seed={a}", "--length=abc"],
                     ["sequence", f"--seed={a}", "--country-mode=bogus"],
                     ["ui", "--port=0"], ["ui", "--port=http"]):
            with self.subTest(argv=argv):
                with mock.patch.object(server, "serve") as serve:
                    r = run(argv)
                serve.assert_not_called()
                self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
                option = argv[-1].split("=")[0]
                self.assertIn(option, r.err)
        self.assertEqual(self.exports(), [])

    def test_port_stops_at_65535(self):
        with mock.patch.object(server, "serve") as serve:
            r = run(["ui", "--port=65536"])
            self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
            self.assertIn("--port", r.err)
            self.assertEqual(run(["ui", "--port=65535"]).status, 0)
        self.assertEqual(serve.call_args_list, [mock.call(65535)])

    def test_match_country_lists_the_three_choices(self):
        r = run(["match", "a.mp4", "--country=Japan"])
        for word in ("any", "same", "different"):
            self.assertIn(word, r.err)

    def test_unknown_mode_is_refused_before_any_output(self):
        a = str(self.ids["a.mp4"])
        for argv in (["match", a, "--mode=vibes"],
                     ["sequence", f"--seed={a}", "--mode=vibes",
                      "--no-export"],
                     ["sequence", f"--seed={a}", "--mode=vibes"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
                self.assertIn("vibes", r.err)
                for mode in config.SCORING_MODES:
                    self.assertIn(mode, r.err)
        self.assertEqual(self.exports(), [])


# -- exit codes and --limit ------------------------------------------------

class TestAnalyzeExitCodes(AnalyzeCase):
    def test_analyze_exits_one_when_any_clip_failed(self):
        r, calls = self.analyze(["analyze", "--country=Japan"],
                                fail={"C0001.MP4"})
        self.assertEqual(len(calls), 4)
        self.assertIn("3 analyzed, 1 failed", r.out)
        self.assertEqual(r.status, 1)
        r, calls = self.analyze(["analyze", "--country=Italy"], fail={"*"})
        self.assertIn("0 analyzed, 3 failed", r.out)
        self.assertEqual(r.status, 1)

    def test_analyze_exits_zero_when_every_clip_worked(self):
        r, calls = self.analyze(["analyze"])
        self.assertEqual((r.status, len(calls), r.leaked), (0, 7, ""))

    def test_bad_numbers_and_roots_exit_without_traceback(self):
        r, calls = self.analyze(["analyze", "--limit=x"])
        self.assertEqual((r.leaked, calls), ("", []))
        self.assertEqual(r.status, 2)
        self.assertIn("--limit", r.err)
        r = run(["scan", f"--root={self.tmp / 'nope'}"])
        self.assertEqual((r.status, r.leaked), (1, ""))
        self.assertIn("not found", r.out)

    def test_limit_zero_opens_nothing(self):
        r, calls = self.analyze(["analyze", "--limit=0"])
        self.assertEqual((calls, r.status, r.leaked), ([], 0, ""))
        self.assertIn("limit 0: nothing analyzed", r.out)

    def test_negative_limit_is_a_usage_error(self):
        for argv in (["analyze", "--limit=-1"], ["analyze", "--limit", "-1"]):
            with self.subTest(argv=argv):
                r, calls = self.analyze(argv)
                self.assertEqual((calls, r.status, r.leaked), ([], 2, ""))
                self.assertIn("--limit", r.err)

    def test_empty_limit_still_means_no_limit(self):
        r, calls = self.analyze(["analyze", "--limit="])
        self.assertEqual((r.status, len(calls)), (0, 7))

    def test_unknown_country_lists_the_known_ones(self):
        r, calls = self.analyze(["analyze", "--country=Narnia"])
        self.assertEqual((calls, r.status, r.leaked), ([], 1, ""))
        self.assertIn("Narnia", r.out)
        self.assertIn("Italy, Japan", r.out)


class TestLookupExitCodes(LibraryCase):
    def test_unknown_clip_exits_one(self):
        for argv in (["match", "nosuch"], ["sequence", "--seed=nosuch"],
                     ["match", " "]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.leaked), (1, ""))
                self.assertIn("no clip matches", r.out)
                self.assertEqual(ranked_lines(r.out), [])
        self.assertEqual(self.exports(), [])

    def test_missing_arguments_are_usage_errors(self):
        for argv in (["match"], ["sequence"], ["sequence", "--length=3"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual((r.status, r.leaked), (2, ""))
                self.assertIn("usage:", r.err)

    def test_unanalyzed_clip_exits_one(self):
        touch(self.media / "Sony SLOG-3" / "Japan" / "new.mp4")
        self.scan()
        for argv in (["match", "new.mp4"], ["sequence", "--seed=new.mp4"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertEqual(r.status, 1)
                self.assertIn("has no features yet", r.out)

    def test_python_m_clipengine_exits_with_the_command_status(self):
        wrapper = (
            "import runpy, sys\n"
            "from pathlib import Path\n"
            "from clipengine import config\n"
            "tmp = Path(sys.argv.pop(1))\n"
            "config.DATA_DIR = tmp / 'data'\n"
            "config.DB_PATH = config.DATA_DIR / 'catalog.db'\n"
            "config.THUMB_DIR = config.DATA_DIR / 'thumbs'\n"
            "config.EXPORT_DIR = tmp / 'exports'\n"
            "config.MEDIA_ROOT = tmp / 'media'\n"
            "runpy.run_module('clipengine', run_name='__main__',"
            " alter_sys=True)\n")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                   CLIPENGINE_MEDIA_ROOT=str(self.media))
        proc = subprocess.run(
            [sys.executable, "-c", wrapper, str(self.tmp), "match", "nosuch"],
            cwd=Path(cli.__file__).resolve().parents[1], env=env,
            capture_output=True, text=True, timeout=60)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertIn("no clip matches 'nosuch'", proc.stdout)
        self.assertEqual(proc.returncode, 1)


# -- a chain of one clip ---------------------------------------------------

class TestOneClipChain(LibraryCase):
    def test_gated_seed_is_refused_and_exports_nothing(self):
        r = run(["sequence", "--seed=a.mp4", "--mode=calm"])
        self.assertEqual(self.exports(), [], r.out)
        self.assertNotIn("exported:", r.out)
        self.assertIn("two clips", r.out)
        self.assertEqual((r.status, r.leaked), (1, ""))
        r = run(["sequence", "--seed=a.mp4", "--mode=calm", "--no-export"])
        self.assertEqual(r.status, 1)
        self.assertNotIn("seed] #", r.out)

    def test_length_one_is_refused_and_exports_nothing(self):
        r = run(["sequence", "--seed=a.mp4", "--length=1"])
        self.assertEqual(self.exports(), [], r.out)
        self.assertEqual((r.status, r.out, r.leaked), (2, "", ""))
        self.assertIn("--length", r.err)

    def test_two_clip_chain_still_exports(self):
        r = run(["sequence", "--seed=a.mp4", "--length=2"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        names = self.exports()
        self.assertEqual(len(names), 3)
        plan = json.loads(next(config.EXPORT_DIR.glob("*.json")).read_text())
        self.assertEqual(len(plan["clips"]), 2)


# -- missing clips ---------------------------------------------------------

class TestMissingClip(LibraryCase):
    def setUp(self):
        super().setUp()
        (self.media / "Sony SLOG-3" / "Japan" / "a.mp4").unlink()
        self.scan()

    def test_missing_clip_by_id_says_missing_on_disk(self):
        gone = str(self.ids["a.mp4"])
        for argv in (["match", gone], ["sequence", f"--seed={gone}"],
                     ["sequence", f"--seed={gone}", "--no-export"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertNotIn("run analyze first", r.out)
                self.assertIn("missing on disk", r.out)
                self.assertIn("a.mp4", r.out)
                self.assertEqual((r.status, r.leaked), (1, ""))
        self.assertEqual(self.exports(), [])

    def test_missing_clip_by_name_gives_the_same_answer(self):
        for argv in (["match", "a.mp4"], ["sequence", "--seed=a.mp4"]):
            with self.subTest(argv=argv):
                r = run(argv)
                self.assertNotIn("no clip matches", r.out)
                self.assertIn(f"#{self.ids['a.mp4']} (a.mp4) is missing on"
                              " disk", r.out)
                self.assertEqual((r.status, r.leaked), (1, ""))
        self.assertEqual(self.exports(), [])

    def test_a_name_on_disk_wins_over_a_missing_one(self):
        r = run(["match", ".mp4"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("/b.mp4 (end:", r.out)

    def test_clip_still_on_disk_is_unaffected(self):
        r = run(["match", str(self.ids["b.mp4"])])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertEqual(len(ranked_lines(r.out)), 1)


# -- unreadable summaries --------------------------------------------------

class TestUnreadableSummaries(LibraryCase):
    def corrupt(self, name: str, text: str) -> None:
        conn = catalog.connect()
        conn.execute("UPDATE features SET summary=? WHERE clip_id=?",
                     (text, self.ids[name]))
        conn.commit()
        conn.close()

    def summary_of(self, name: str):
        conn = catalog.connect()
        text = conn.execute("SELECT summary FROM features WHERE clip_id=?",
                            (self.ids[name],)).fetchone()[0]
        conn.close()
        return text

    def test_relabel_skips_unreadable_summaries_and_names_them(self):
        self.corrupt("a.mp4", json.dumps({"start_class": "static",
                                          "end_class": "static"}))
        self.corrupt("b.mp4", "{not json")
        self.corrupt("c.mp4", "[]")
        r = run(["relabel"])
        self.assertEqual(r.leaked, "")
        self.assertIn("b.mp4", r.out)
        self.assertIn("c.mp4", r.out)
        self.assertEqual(r.status, 1)
        self.assertEqual(json.loads(self.summary_of("a.mp4"))["end_class"],
                         "pan_right")
        self.assertEqual(self.summary_of("b.mp4"), "{not json")
        self.assertEqual(self.summary_of("c.mp4"), "[]")

    def test_relabel_fills_a_null_summary(self):
        self.corrupt("a.mp4", None)
        r = run(["relabel"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertEqual(json.loads(self.summary_of("a.mp4"))["start_class"],
                         "pan_right")

    def test_relabel_reads_a_json_null_summary_like_a_null_one(self):
        # audit already reads the text null as empty; relabel should agree
        self.corrupt("a.mp4", "null")
        r = run(["relabel"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertNotIn("unreadable", r.out)
        self.assertEqual(json.loads(self.summary_of("a.mp4"))["start_class"],
                         "pan_right")

    def test_audit_skips_unreadable_summaries(self):
        # the library loads before the damage, the way load_library will
        # once it skips bad rows, so only the audit's own read is tested
        conn = catalog.connect()
        lib = matching.load_library(conn)
        conn.close()
        self.corrupt("b.mp4", "{not json")
        self.corrupt("c.mp4", "null")
        with mock.patch.object(matching, "load_library",
                               return_value=lib):
            r = run(["audit"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("audit over 3 analyzed clips", r.out)
        self.assertIn("flat/log detected: 0/3", r.out)


# -- status, audit and help ------------------------------------------------

class TestStatusAndAudit(LibraryCase):
    def test_status_counts_clips_gone_from_disk(self):
        (self.media / "Sony SLOG-3" / "Japan" / "a.mp4").unlink()
        self.scan()
        r = run(["status"])
        self.assertEqual((r.status, r.leaked), (0, ""))
        self.assertIn("catalog: 2 clips", r.out)
        self.assertIn("| 1 missing", r.out.splitlines()[0])

    def broken(self) -> Path:
        bad = self.media / "Sony SLOG-3" / "Japan" / "broken.mp4"
        touch(bad)
        self.scan()
        conn = catalog.connect()
        row = catalog.find_clip(conn, "broken")
        catalog.save_features(conn, row["id"], row["content_key"], None,
                              None, error="decoder could not open")
        conn.close()
        return bad

    def test_audit_lists_errors_only_for_clips_on_disk(self):
        bad = self.broken()
        self.assertIn("1 analysis errors", run(["audit"]).out)
        bad.unlink()
        self.scan()
        r = run(["audit"])
        self.assertIn("no analysis errors", r.out)
        self.assertNotIn("broken.mp4", r.out)

    def test_errors_from_an_older_extractor_are_not_listed(self):
        self.broken()
        self.assertIn("| 1 errors", run(["status"]).out)
        blob = features.to_bytes(make_vec(**MOVING))
        with mock.patch.object(config, "FEATURE_VERSION",
                               config.FEATURE_VERSION + 1):
            # one clip analyzed again so audit has a library to report on
            conn = catalog.connect()
            row = catalog.find_clip(conn, "a.mp4")
            catalog.save_features(conn, row["id"], row["content_key"], blob,
                                  json.dumps(PAN_SUMMARY))
            conn.close()
            r = run(["audit"])
            status = run(["status"])
        self.assertIn("audit over 1 analyzed clips", r.out)
        self.assertIn("no analysis errors", r.out)
        self.assertNotIn("broken.mp4", r.out)
        self.assertIn("| 0 errors", status.out)

    def test_busy_port_is_a_clean_error(self):
        busy = OSError(48, "Address already in use")
        with mock.patch.object(server, "serve", side_effect=busy):
            r = run(["ui", "--port=8764"])
        self.assertEqual((r.status, r.leaked), (1, ""))
        self.assertIn("8764", r.out + r.err)
        self.assertIn("Address already in use", r.out + r.err)


if __name__ == "__main__":
    unittest.main()
