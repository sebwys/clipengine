# test_matching_sequence.py
# contracts of the scorer and the chain builder: the gate formulas, the
# shape of what rank and build_chain hand back, and the rows that must
# never reach a match list. a corrupt vector or summary costs one clip,
# not the library, and a country option nobody defined is an error
# instead of a quiet "any".

import json
import logging
import math
import unicodedata
import unittest
from unittest import mock

import numpy as np

from clipengine import catalog, config, features, matching, sequence
from tests.test_matching import lib_of, make_vec
from tests.util import TempDirsMixin

# every clip pans right at the same speed, in and out
MOVING = dict(start_flow_x=-0.5, start_energy=0.5,
              end_flow_x=-0.5, end_energy=0.5)
BREAKDOWN = {"motion", "energy", "color", "luma", "gate"}


def matrix(n: int, base: float) -> np.ndarray:
    m = np.full((n, n), base, dtype=np.float32)
    np.fill_diagonal(m, -1.0)
    return m


def random_vec(rng: np.random.Generator) -> np.ndarray:
    """a plausible clip: any direction and speed, any color."""
    kw = {}
    for side in ("start", "end"):
        still = rng.random() < 0.2
        fx, fy = (0.0, 0.0) if still else rng.uniform(-2.0, 2.0, 2)
        kw.update({
            f"{side}_flow_x": fx, f"{side}_flow_y": fy,
            f"{side}_energy": 0.0 if still else rng.uniform(0.0, 2.5),
            f"{side}_radial": rng.uniform(-0.5, 0.5),
            f"{side}_jitter": rng.uniform(0.0, 1.0),
            f"{side}_luma": rng.uniform(0.0, 255.0),
            f"{side}_contrast": rng.uniform(0.0, 255.0),
            f"{side}_sat": rng.uniform(0.0, 255.0),
            f"{side}_warmth": rng.uniform(-60.0, 60.0),
            f"{side}_tint": rng.uniform(-60.0, 60.0)})
    vec = make_vec(**kw)
    for sl in (features.START_HUE, features.END_HUE):
        if rng.random() < 0.2:
            vec[sl] = np.eye(features.HUE_BINS)[rng.integers(features.HUE_BINS)]
        else:
            vec[sl] = rng.dirichlet(np.ones(features.HUE_BINS))
    return vec


def greedy(m: np.ndarray, seed: int, length: int) -> list[int]:
    """take the best unused cut each step, stop at a dead end."""
    path = [seed]
    while len(path) < length:
        row = m[path[-1]].astype(np.float64)
        row[path] = -1.0
        j = int(np.argmax(row))
        if row[j] <= 0:
            break
        path.append(j)
    return path


def poisoned(slot: str, value: float, base: dict = MOVING) -> matching.Library:
    """four alike clips, the third (id 3) with one corrupt slot."""
    vecs = [make_vec(**base) for _ in range(4)]
    vecs[2][features.INDEX[slot]] = value
    return lib_of([("A", v) for v in vecs])


def gates(lib: matching.Library, mode: str) -> list[float]:
    _, c = matching.score_against(lib, 0, mode)
    return [float(g) for g in c["gates"][mode][1:]]


class TestModeWeights(unittest.TestCase):
    def test_mode_weights_sum_to_one(self):
        for mode, w in config.SCORING_MODES.items():
            with self.subTest(mode=mode):
                self.assertEqual(set(w), {"motion", "energy", "color", "luma"})
                self.assertAlmostEqual(sum(w.values()), 1.0, places=9)


class TestScoreRange(unittest.TestCase):
    def test_scores_in_unit_range_fuzz(self):
        rng = np.random.default_rng(477)
        lib = lib_of([("A", random_vec(rng)) for _ in range(400)])
        off = ~np.eye(len(lib), dtype=bool)
        for mode in config.SCORING_MODES:
            with self.subTest(mode=mode):
                M = matching.full_matrix(lib, mode)
                self.assertTrue(np.isfinite(M).all())
                self.assertGreaterEqual(float(M[off].min()), 0.0)
                self.assertLessEqual(float(M[off].max()), 1.0)
                np.testing.assert_array_equal(np.diag(M), -1.0)

    def test_identical_still_clips_score_one_in_calm(self):
        for hue_bin in (None, 3):
            with self.subTest(hue_bin=hue_bin):
                v = make_vec(start_energy=0.0, end_energy=0.0,
                             start_hue_bin=hue_bin, end_hue_bin=hue_bin)
                lib = lib_of([("A", v), ("A", v.copy())])
                scores, _ = matching.score_against(lib, 0, "calm")
                self.assertEqual(float(scores[1]), 1.0)


class TestGates(unittest.TestCase):
    def test_momentum_gate_formula(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.5, end_energy=0.5)),
            ("A", make_vec(start_flow_x=-0.06, start_energy=0.06)),
            ("A", make_vec(start_flow_x=-0.03, start_energy=0.03)),
            ("A", make_vec(start_energy=0.0)),
        ])
        for got, want in zip(gates(lib, "momentum"), (1.0, 0.5, 0.0)):
            self.assertAlmostEqual(got, want, places=6)
        ids = [r["id"] for r in matching.rank(lib, 1, "momentum")]
        self.assertNotIn(4, ids)
        self.assertIn(2, ids)

    def test_whip_gate_is_quadratic(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.48, end_energy=0.48)),
            ("A", make_vec(start_flow_x=-0.48, start_energy=0.48)),
            ("A", make_vec(start_flow_x=-0.24, start_energy=0.24)),
            ("A", make_vec(start_flow_x=-0.6, start_energy=0.6)),
        ])
        for got, want in zip(gates(lib, "whip"), (1.0, 0.25, 1.0)):
            self.assertAlmostEqual(got, want, places=6)

    def test_calm_gate_formula_and_exclusion(self):
        lib = lib_of([
            ("A", make_vec(end_energy=0.0)),
            ("A", make_vec(start_energy=0.0)),
            ("A", make_vec(start_energy=0.03)),
            ("A", make_vec(start_flow_x=-0.06, start_energy=0.06)),
        ])
        for got, want in zip(gates(lib, "calm"), (1.0, 0.5, 0.0)):
            self.assertAlmostEqual(got, want, places=6)
        mover = lib_of([("A", make_vec(end_energy=0.0)),
                        ("A", make_vec(**MOVING))])
        self.assertEqual(matching.rank(mover, 1, "calm"), [])

    def test_contrast_gate_always_one(self):
        still = make_vec()
        lib = lib_of([("A", still), ("A", still.copy()),
                      ("A", make_vec(start_flow_x=-1.8, start_energy=1.8))])
        scores, c = matching.score_against(lib, 0, "contrast")
        np.testing.assert_array_equal(c["gates"]["contrast"], 1.0)
        self.assertAlmostEqual(float(scores[1]), 0.3, places=5)
        self.assertLess(float(scores[2]), 0.01)


class TestComponents(unittest.TestCase):
    def test_motion_component_geometry(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-0.5, end_energy=0.5)),
            ("A", make_vec(start_flow_y=0.5, start_energy=0.5)),   # 2: perpendicular
            ("A", make_vec(start_flow_x=0.5, start_energy=0.5)),   # 3: opposite
            ("A", make_vec(start_flow_x=-0.5, start_energy=0.5)),  # 4: same way
            ("A", make_vec()),                                     # 5: still
        ])
        motion = matching.components(lib, 0)["motion"]
        self.assertAlmostEqual(float(motion[1]), 0.5, places=5)
        self.assertLess(float(motion[2]), 0.01)
        self.assertGreater(float(motion[3]), 0.99)
        self.assertEqual(float(motion[4]), 0.5)
        # a still end makes every cut out of it neutral
        lib.F[0] = make_vec()
        np.testing.assert_array_equal(
            matching.components(lib, 0)["motion"][1:], 0.5)

    def test_energy_component_symmetric_and_zero_safe(self):
        def energy(ae: float, be: float) -> float:
            lib = lib_of([("A", make_vec(end_energy=ae)),
                          ("A", make_vec(start_energy=be))])
            return float(matching.components(lib, 0)["energy"][1])

        self.assertAlmostEqual(energy(0.1, 0.4), energy(0.4, 0.1), places=6)
        self.assertAlmostEqual(energy(0.1, 0.4), 0.25, places=5)
        self.assertEqual(energy(0.0, 0.0), 1.0)

    def test_hue_overlap_uniform_vs_one_hot(self):
        lib = lib_of([("A", make_vec()), ("A", make_vec(start_hue_bin=4))])
        color = matching.components(lib, 0)["color"]
        self.assertAlmostEqual(float(color[1]), 0.55 + 0.45 / 12, places=5)
        self.assertAlmostEqual(float(color[1]), 0.5875, places=5)


class TestRank(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(12)
        self.lib = lib_of([("A" if i % 2 else "B", random_vec(rng))
                           for i in range(10)])

    def test_rank_contract(self):
        meta_keys = set(self.lib.meta[0])
        for mode in config.SCORING_MODES:
            for clip_id in (1, 4, 9):
                with self.subTest(mode=mode, clip=clip_id):
                    out = matching.rank(self.lib, clip_id, mode, n=4)
                    scores = [r["score"] for r in out]
                    self.assertLessEqual(len(out), 4)
                    self.assertEqual(scores, sorted(scores, reverse=True))
                    self.assertTrue(all(s > 0 for s in scores))
                    self.assertNotIn(clip_id, [r["id"] for r in out])
                    for r in out:
                        self.assertEqual(set(r), meta_keys
                                         | {"score", "breakdown"})
                        self.assertEqual(set(r["breakdown"]), BREAKDOWN)

    def test_rank_with_no_room_is_empty(self):
        for n in (0, -3):
            self.assertEqual(matching.rank(self.lib, 1, "contrast", n=n), [])

    def test_single_clip_library(self):
        lib = lib_of([("A", make_vec(**MOVING))])
        self.assertEqual(matching.rank(lib, 1, "momentum"), [])
        M = matching.full_matrix(lib, "momentum")
        np.testing.assert_array_equal(M, [[-1.0]])
        self.assertEqual(sequence.build_chain(M, 0, 5), ([0], []))


class TestChain(unittest.TestCase):
    def test_chain_length_exactly_library_size(self):
        rows, edges = sequence.build_chain(matrix(6, 0.4), 2, 6)
        self.assertEqual(len(rows), 6)
        self.assertEqual(sorted(rows), list(range(6)))
        self.assertEqual(rows[0], 2)
        np.testing.assert_allclose(edges, [0.4] * 5, rtol=1e-6)

    def test_chain_stops_at_mid_dead_end(self):
        m = matrix(4, -1.0)
        m[0, 1], m[1, 2] = 0.9, 0.8
        rows, edges = sequence.build_chain(m, 0, 4)
        self.assertEqual(rows, [0, 1, 2])
        np.testing.assert_allclose(edges, [0.9, 0.8], rtol=1e-6)

    def test_chain_stops_at_a_cut_that_scores_exactly_zero(self):
        # a gated cut scores exactly 0, and 0 is no cut
        for width in (1, 12):
            with self.subTest(beam_width=width):
                self.assertEqual(sequence.build_chain(matrix(4, 0.0), 0, 4,
                                                      beam_width=width),
                                 ([0], []))

    def test_beam_width_one_is_greedy(self):
        rng = np.random.default_rng(5)
        m = rng.uniform(0.01, 1.0, (12, 12)).astype(np.float32)
        np.fill_diagonal(m, -1.0)
        for seed in (0, 7):
            with self.subTest(seed=seed):
                rows, _ = sequence.build_chain(m, seed, 12, beam_width=1)
                self.assertEqual(rows, greedy(m, seed, 12))

    def test_same_mode_seed_alone_in_country(self):
        rows = sequence.build_chain(matrix(4, 0.5), 0, 4,
                                    countries=["A", "B", "B", "B"],
                                    country_mode="same")
        self.assertEqual(rows, ([0], []))

    def test_length_zero_or_negative_returns_seed(self):
        for length in (0, -1, -5):
            with self.subTest(length=length):
                self.assertEqual(
                    sequence.build_chain(matrix(4, 0.5), 3, length),
                    ([3], []))

    def test_matrix_and_chain_repeat_exactly(self):
        rng = np.random.default_rng(9)
        lib = lib_of([("A" if i < 6 else "B", random_vec(rng))
                      for i in range(12)])
        for mode in config.SCORING_MODES:
            with self.subTest(scoring=mode):
                self.assertEqual(matching.full_matrix(lib, mode).tobytes(),
                                 matching.full_matrix(lib, mode).tobytes())
        M = matching.full_matrix(lib, "contrast")
        countries = [m["country"] for m in lib.meta]
        for mode in ("any", "same", "travel"):
            with self.subTest(country_mode=mode):
                first = sequence.build_chain(M, 0, 8, countries=countries,
                                             country_mode=mode)
                again = sequence.build_chain(M.copy(), 0, 8,
                                             countries=list(countries),
                                             country_mode=mode)
                self.assertEqual(first, again)


class TestNonFiniteVectors(unittest.TestCase):
    # one corrupt slot in a slot the scorer reads, nan or either infinity
    CASES = [("start_tint", math.nan), ("start_luma", math.nan),
             ("start_flow_x", math.nan), ("start_energy", math.inf),
             ("start_warmth", -math.inf), ("start_hue_3", math.nan)]

    def setUp(self):
        # the scorer still does math on the bad row before masking it
        self.enterContext(np.errstate(divide="ignore", invalid="ignore"))

    def test_non_finite_vectors_are_excluded(self):
        for slot, value in self.CASES:
            lib = poisoned(slot, value)
            for mode in ("momentum", "whip", "contrast"):
                with self.subTest(slot=slot, value=value, mode=mode):
                    out = matching.rank(lib, 1, mode, n=5)
                    self.assertEqual(sorted(r["id"] for r in out), [2, 4])
                    self.assertTrue(all(math.isfinite(r["score"])
                                        for r in out))
            with self.subTest(slot=slot, value=value, mode="calm"):
                self.assertEqual(matching.rank(lib, 1, "calm"), [])

    def test_chain_never_takes_a_non_finite_clip(self):
        for slot, value in self.CASES:
            M = matching.full_matrix(poisoned(slot, value), "momentum")
            for seed in (0, 1, 3):
                with self.subTest(slot=slot, value=value, seed=seed):
                    rows, edges = sequence.build_chain(M, seed, 4)
                    self.assertNotIn(2, rows)
                    self.assertEqual(len(rows), 3)
                    self.assertTrue(np.isfinite(edges).all())

    def test_corrupt_clip_gets_no_matches_out(self):
        for slot in ("end_luma", "end_flow_x", "global_luma"):
            with self.subTest(slot=slot):
                lib = poisoned(slot, math.nan)
                self.assertEqual(matching.rank(lib, 3, "momentum"), [])

    def test_chain_skips_non_finite_matrix_cells(self):
        m = matrix(4, 0.2)
        m[0, 1], m[0, 2], m[0, 3] = np.nan, np.inf, 0.4
        rows, edges = sequence.build_chain(m, 0, 2)
        self.assertEqual(rows, [0, 3])
        np.testing.assert_allclose(edges, [0.4], rtol=1e-6)

    def test_finite_row_whose_motion_goes_nan_is_no_cut(self):
        # every slot is finite, but the cosine overflows to nan
        a = make_vec(end_flow_x=-3e38, end_energy=3e38)
        b = make_vec(start_flow_x=-3e38, start_energy=3e38)
        lib = lib_of([("A", a), ("A", b), ("A", make_vec(**MOVING))])
        self.assertTrue(np.isfinite(lib.F).all())
        # the overflow is the point of the test, keep numpy quiet about it
        with np.errstate(over="ignore", invalid="ignore"):
            _, c = matching.score_against(lib, 0, "momentum")
            ids = [r["id"] for r in matching.rank(lib, 1, "momentum")]
        self.assertTrue(math.isnan(c["motion"][1]))
        self.assertNotIn(2, ids)
        self.assertIn(3, ids)


class TestHugeFiniteValues(unittest.TestCase):
    # a corrupt float32 is as likely huge as nan. these slots push the
    # score far outside [0, 1] while every value stays finite
    CASES = [("start_luma", 3e38), ("start_luma", -3e38),
             ("start_hue_3", -3e38)]

    def setUp(self):
        self.enterContext(np.errstate(over="ignore", invalid="ignore"))

    def test_huge_value_never_ranks(self):
        for slot, value in self.CASES:
            lib = poisoned(slot, value)
            for mode in ("momentum", "whip", "contrast"):
                with self.subTest(slot=slot, value=value, mode=mode):
                    out = matching.rank(lib, 1, mode, n=5)
                    self.assertEqual(sorted(r["id"] for r in out), [2, 4])
                    self.assertTrue(all(0 < r["score"] <= 1 for r in out))

    def test_matrix_holds_no_score_outside_unit_range(self):
        for slot, value in self.CASES:
            for mode in config.SCORING_MODES:
                with self.subTest(slot=slot, value=value, mode=mode):
                    M = matching.full_matrix(poisoned(slot, value), mode)
                    self.assertLessEqual(float(M.max()), 1.0 + 1e-4)
                    self.assertGreaterEqual(float(M.min()), -1.0)

    def test_chain_never_takes_a_huge_clip(self):
        M = matching.full_matrix(poisoned("start_luma", 3e38), "contrast")
        rows, edges = sequence.build_chain(M, 0, 4)
        self.assertNotIn(2, rows)
        self.assertTrue(all(0 < e <= 1 for e in edges))


class TestPackRejectsGarbage(unittest.TestCase):
    # do now improvement: pack fails loudly, as its docstring promises
    def setUp(self):
        self.scalars = {n: 0.5 for n in features.FIELDS
                        if not n.startswith(("start_hue_", "end_hue_"))}
        self.hue = np.full(features.HUE_BINS, 1.0 / features.HUE_BINS)

    def test_pack_rejects_non_finite_scalars(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                bad = {**self.scalars, "end_tint": value}
                with self.assertRaisesRegex(ValueError, "end_tint"):
                    features.pack(bad, self.hue, self.hue)

    def test_pack_rejects_non_finite_hue(self):
        hue = self.hue.copy()
        hue[5] = np.nan
        with self.assertRaisesRegex(ValueError, "start_hue_5"):
            features.pack(self.scalars, hue, self.hue)

    def test_pack_rejects_wrong_hue_shapes(self):
        for shape in ((1,), (), (13,), (11,), (12, 1), (1, 12)):
            with self.subTest(shape=shape):
                hue = np.full(shape, 1.0 / features.HUE_BINS)
                with self.assertRaisesRegex(ValueError, "hue"):
                    features.pack(self.scalars, self.hue, hue)

    def test_pack_still_takes_lists_and_float64(self):
        start = np.arange(features.HUE_BINS, dtype=np.float64) / 66.0
        vec = features.pack(self.scalars, list(start), self.hue)
        self.assertEqual(vec.dtype, np.float32)
        self.assertEqual(vec.shape, (features.VECTOR_LEN,))
        np.testing.assert_allclose(vec[features.START_HUE], start, rtol=1e-6)
        np.testing.assert_allclose(vec[features.END_HUE], self.hue, rtol=1e-6)
        self.assertEqual(features.get(vec, "end_tint"), 0.5)


class TestUnknownCountryOptions(unittest.TestCase):
    def setUp(self):
        self.m = matrix(4, 0.5)
        self.countries = ["Japan", "Japan", "Iceland", "Iceland"]
        self.lib = lib_of([
            ("Japan", make_vec(end_flow_x=-0.5, end_energy=0.5)),
            ("Japan", make_vec(start_flow_x=-0.5, start_energy=0.5)),
            ("Iceland", make_vec(start_flow_x=-0.5, start_energy=0.5)),
        ])

    def test_unknown_country_mode_raises_naming_the_valid_set(self):
        for bad in ("different", "Same", "TRAVEL", "bogus", "", None):
            for countries in (self.countries, None):
                with self.subTest(country_mode=bad, countries=countries):
                    with self.assertRaises(ValueError) as cm:
                        sequence.build_chain(self.m, 0, 4,
                                             countries=countries,
                                             country_mode=bad)
                    for word in ("any", "same", "travel"):
                        self.assertIn(word, str(cm.exception))

    def test_unknown_country_filter_raises_naming_the_valid_set(self):
        for bad in ("travel", "SAME", "diff", "", None):
            with self.subTest(country=bad):
                with self.assertRaises(ValueError) as cm:
                    matching.rank(self.lib, 1, "momentum", country=bad)
                for word in ("any", "same", "different"):
                    self.assertIn(word, str(cm.exception))

    def test_beam_width_below_one_raises(self):
        for width in (0, -1):
            with self.subTest(beam_width=width):
                with self.assertRaisesRegex(ValueError, "beam"):
                    sequence.build_chain(self.m, 0, 4, beam_width=width)

    def test_countries_of_the_wrong_length_raise(self):
        for mode in ("any", "same", "travel"):
            for countries in (self.countries[:3], self.countries + ["Peru"]):
                with self.subTest(country_mode=mode, n=len(countries)):
                    with self.assertRaisesRegex(ValueError, "countries"):
                        sequence.build_chain(self.m, 0, 4,
                                             countries=countries,
                                             country_mode=mode)

    def test_any_spelled_out_matches_the_default(self):
        self.assertEqual(
            matching.rank(self.lib, 1, "momentum", country="any"),
            matching.rank(self.lib, 1, "momentum"))
        self.assertEqual(
            sequence.build_chain(self.m, 0, 4, countries=self.countries,
                                 country_mode="any"),
            sequence.build_chain(self.m, 0, 4))


class TestLoadLibrary(TempDirsMixin, unittest.TestCase):
    # an accent and spaces in a name must not cost the clip
    CAFE = "d\u00edas de caf\u00e9.mp4"
    NAMES = ("a_ok.mp4", CAFE, "short.mp4", "z_bad.mp4")

    def setUp(self):
        super().setUp()
        folder = config.MEDIA_ROOT / "Sony SLOG-3" / "Japan"
        folder.mkdir(parents=True)
        for name in self.NAMES:
            (folder / name).write_bytes(b"x" * 1024)
        self.conn = catalog.connect()
        self.addCleanup(self.conn.close)
        catalog.scan(self.conn, config.MEDIA_ROOT)
        self.rows = {unicodedata.normalize("NFC", r["name"]): r
                     for r in self.conn.execute("SELECT * FROM clips")}
        self.vec = make_vec(**MOVING)
        self.summary = json.dumps({"duration_s": 5.0, "fps": 24.0,
                                   "start_class": "pan_right",
                                   "end_class": "pan_right"})
        for name in self.NAMES:
            self.save(name)

    def save(self, name: str, vec=None, summary=None) -> None:
        r = self.rows[name]
        vec = self.vec if vec is None else vec
        catalog.save_features(self.conn, r["id"], r["content_key"],
                              features.to_bytes(vec),
                              self.summary if summary is None else summary)

    def loaded(self, lib: matching.Library) -> list[str]:
        return [unicodedata.normalize("NFC", m["name"]) for m in lib.meta]

    def test_empty_library_load(self):
        self.conn.execute("DELETE FROM features")
        self.conn.commit()
        lib = matching.load_library(self.conn)
        self.assertEqual(len(lib), 0)
        self.assertEqual(lib.F.shape, (0, features.VECTOR_LEN))
        self.assertEqual(matching.full_matrix(lib).shape, (0, 0))
        with self.assertRaises(KeyError):
            matching.rank(lib, 1)

    def truncate(self, name: str) -> None:
        r = self.rows[name]
        catalog.save_features(self.conn, r["id"], r["content_key"],
                              features.to_bytes(self.vec)[:-4], self.summary)

    def test_load_library_skips_wrong_length_vector(self):
        self.truncate("short.mp4")
        # the skip warning has its own test below
        with mock.patch.object(logging.getLogger("clipengine.matching"),
                               "disabled", True):
            lib = matching.load_library(self.conn)
        want = sorted((self.rows[n]["id"], n) for n in self.NAMES
                      if n != "short.mp4")
        self.assertEqual(list(lib.ids), [i for i, _ in want])
        self.assertEqual(self.loaded(lib), [n for _, n in want])
        cafe = lib.row_of[self.rows[self.CAFE]["id"]]
        np.testing.assert_array_equal(lib.F[cafe], self.vec)
        self.assertEqual(lib.meta[cafe]["start_class"], "pan_right")
        self.assertEqual(lib.meta[cafe]["country"], "Japan")

    def test_wrong_length_vector_warning_names_the_clip(self):
        self.truncate("short.mp4")
        with self.assertLogs("clipengine.matching", "WARNING") as logs:
            lib = matching.load_library(self.conn)
        self.assertEqual(len(lib), 3)
        self.assertIn("short.mp4", "\n".join(logs.output))

    def test_load_library_skips_malformed_summary(self):
        for bad in ("{not json", "[]", "null", '"text"', "5",
                    '{"duration_s": NaN, "start_class": "static"}',
                    '{"fps": Infinity}', '{"width": -Infinity}',
                    '{"duration_s": 1e999}', '{"duration_s": 1' + '0' * 400 + '}',
                    '{"fps": "abc"}', '{"height": true}'):
            with self.subTest(summary=bad):
                self.save("z_bad.mp4", summary=bad)
                with self.assertLogs("clipengine.matching", "WARNING") as logs:
                    lib = matching.load_library(self.conn)
                    self.assertNotIn("z_bad.mp4", self.loaded(lib))
                self.assertIn("a_ok.mp4", self.loaded(lib))
                self.assertEqual(len(lib), 3)
                self.assertIn("z_bad.mp4", "\n".join(logs.output))

    def test_load_library_skips_non_finite_vector(self):
        for slot, value in (("start_tint", math.nan),
                            ("end_energy", math.inf),
                            ("duration_s", math.nan)):
            with self.subTest(slot=slot, value=value):
                bad = self.vec.copy()
                bad[features.INDEX[slot]] = value
                self.save("z_bad.mp4", vec=bad)
                with self.assertLogs("clipengine.matching", "WARNING") as logs:
                    lib = matching.load_library(self.conn)
                    self.assertNotIn("z_bad.mp4", self.loaded(lib))
                self.assertEqual(len(lib), 3)
                self.assertTrue(np.isfinite(lib.F).all())
                self.assertIn("z_bad.mp4", "\n".join(logs.output))

    def test_load_library_skips_a_vector_that_is_not_a_blob(self):
        # sqlite keeps any type in a column, and frombuffer raises
        # TypeError on text or a number, not ValueError
        r = self.rows["z_bad.mp4"]
        for bad in ("not a blob", 5, 2.5):
            with self.subTest(vector=bad):
                self.conn.execute(
                    "UPDATE features SET vector=? WHERE clip_id=?",
                    (bad, r["id"]))
                self.conn.commit()
                with self.assertLogs("clipengine.matching", "WARNING") as logs:
                    lib = matching.load_library(self.conn)
                self.assertNotIn("z_bad.mp4", self.loaded(lib))
                self.assertEqual(len(lib), 3)
                self.assertIn("z_bad.mp4", "\n".join(logs.output))

    def test_null_vector_row_is_left_out(self):
        r = self.rows["z_bad.mp4"]
        self.conn.execute("UPDATE features SET vector=NULL WHERE clip_id=?",
                          (r["id"],))
        self.conn.commit()
        lib = matching.load_library(self.conn)
        self.assertNotIn("z_bad.mp4", self.loaded(lib))
        self.assertEqual(len(lib), 3)

    def test_null_summary_still_loads(self):
        r = self.rows["z_bad.mp4"]
        self.conn.execute("UPDATE features SET summary=NULL WHERE clip_id=?",
                          (r["id"],))
        self.conn.commit()
        lib = matching.load_library(self.conn)
        self.assertIn("z_bad.mp4", self.loaded(lib))
        self.assertIsNone(lib.meta[lib.row_of[r["id"]]]["start_class"])


if __name__ == "__main__":
    unittest.main()
