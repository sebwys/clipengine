# test_matching.py
# scoring math on hand built vectors. building vectors directly (instead
# of analyzing video) makes every expectation exact: we control each
# slot, so we know which clip must win and roughly by how much.

import unittest

import numpy as np

from clipengine import config, features, matching


def make_vec(**kw) -> np.ndarray:
    defaults = dict(
        start_flow_x=0.0, start_flow_y=0.0, start_energy=0.004,
        start_radial=0.0, start_jitter=0.0,
        end_flow_x=0.0, end_flow_y=0.0, end_energy=0.004,
        end_radial=0.0, end_jitter=0.0,
        start_luma=120.0, start_contrast=80.0, start_sat=120.0,
        start_warmth=0.0, start_tint=0.0,
        end_luma=120.0, end_contrast=80.0, end_sat=120.0,
        end_warmth=0.0, end_tint=0.0,
        global_luma=120.0, global_sat=120.0, global_warmth=0.0,
        tempo_mean=0.2, tempo_var=0.01, duration_s=5.0)
    start_bin = kw.pop("start_hue_bin", None)
    end_bin = kw.pop("end_hue_bin", None)
    defaults.update(kw)

    def hue(bin_idx):
        if bin_idx is None:
            return np.full(features.HUE_BINS, 1.0 / features.HUE_BINS)
        h = np.zeros(features.HUE_BINS)
        h[bin_idx] = 1.0
        return h

    return features.pack(defaults, hue(start_bin), hue(end_bin))


def lib_of(specs: list[tuple[str, np.ndarray]]) -> matching.Library:
    """specs: (country, vector) pairs -> in memory library, ids 1..n."""
    ids = np.arange(1, len(specs) + 1, dtype=np.int64)
    F = np.stack([v for _, v in specs]).astype(np.float32)
    meta = [{"id": int(i), "name": f"clip{i}", "country": c, "profile": "t",
             "path": f"/x/clip{i}.mp4", "duration_s": 5.0,
             "start_class": "?", "end_class": "?"}
            for i, (c, _) in zip(ids, specs)]
    return matching.Library(ids, F, meta)


class TestMomentum(unittest.TestCase):
    def setUp(self):
        self.lib = lib_of([
            ("Japan", make_vec(end_flow_x=-0.5, end_energy=0.5)),    # A: ends pan right
            ("Japan", make_vec(start_flow_x=-0.5, start_energy=0.5)),  # B: starts pan right
            ("Iceland", make_vec(start_flow_x=0.5, start_energy=0.5)),  # C: starts pan left
            ("Japan", make_vec()),                                     # D: starts static
        ])

    def test_same_direction_beats_opposite_and_static(self):
        results = matching.rank(self.lib, 1, "momentum", n=5)
        self.assertEqual(results[0]["id"], 2)
        by_id = {r["id"]: r for r in results}
        self.assertGreater(by_id[2]["score"], by_id[3]["score"])
        if 4 in by_id:  # static candidate may be gated out entirely
            self.assertGreater(by_id[2]["score"], by_id[4]["score"])

    def test_motion_component_reads_direction(self):
        results = matching.rank(self.lib, 1, "momentum", n=5)
        by_id = {r["id"]: r for r in results}
        self.assertGreater(by_id[2]["breakdown"]["motion"], 0.9)
        self.assertLess(by_id[3]["breakdown"]["motion"], 0.1)

    def test_static_candidate_is_gated(self):
        results = matching.rank(self.lib, 1, "momentum", n=5)
        by_id = {r["id"]: r for r in results}
        if 4 in by_id:
            self.assertLess(by_id[4]["breakdown"]["gate"], 0.15)

    def test_self_never_matches(self):
        results = matching.rank(self.lib, 1, "momentum", n=10)
        self.assertNotIn(1, [r["id"] for r in results])

    def test_country_filters(self):
        same = matching.rank(self.lib, 1, "momentum", n=10, country="same")
        self.assertTrue(all(r["country"] == "Japan" for r in same))
        diff = matching.rank(self.lib, 1, "momentum", n=10, country="different")
        self.assertTrue(all(r["country"] != "Japan" for r in diff))


class TestWhip(unittest.TestCase):
    def test_whip_needs_speed_on_both_sides(self):
        lib = lib_of([
            ("A", make_vec(end_flow_x=-1.8, end_energy=1.8)),   # 1: ends whipping
            ("B", make_vec(start_flow_x=-1.8, start_energy=1.8)),  # 2: starts whipping
            ("C", make_vec(start_flow_x=-0.3, start_energy=0.3)),  # 3: slow pan
        ])
        scores, _ = matching.score_against(lib, 0, "whip")
        self.assertGreater(scores[1], 2 * scores[2])


class TestCalm(unittest.TestCase):
    def test_still_and_color_led(self):
        lib = lib_of([
            ("A", make_vec(end_hue_bin=4)),                             # 1: still, teal end
            ("B", make_vec(start_hue_bin=4)),                           # 2: still, same palette
            ("C", make_vec(start_hue_bin=10, start_luma=40.0,
                           start_warmth=-30.0)),                        # 3: still, far palette
            ("D", make_vec(start_flow_x=-0.5, start_energy=0.5,
                           start_hue_bin=4)),                           # 4: moving
        ])
        scores, _ = matching.score_against(lib, 0, "calm")
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])
        self.assertLessEqual(scores[3], 0.0)  # movement fails the calm gate


class TestContrast(unittest.TestCase):
    def test_vibe_flip_beats_same_vibe(self):
        lib = lib_of([
            ("A", make_vec(end_luma=200.0, end_warmth=25.0, end_hue_bin=1)),
            ("B", make_vec(start_luma=50.0, start_warmth=-25.0,
                           start_hue_bin=7)),   # 2: opposite vibe
            ("C", make_vec(start_luma=200.0, start_warmth=25.0,
                           start_hue_bin=1)),   # 3: same vibe
        ])
        scores, _ = matching.score_against(lib, 0, "contrast")
        self.assertGreater(scores[1], scores[2])


class TestPlumbing(unittest.TestCase):
    def test_unknown_mode_raises(self):
        lib = lib_of([("A", make_vec()), ("B", make_vec())])
        with self.assertRaises(ValueError):
            matching.score_against(lib, 0, "vibes")

    def test_full_matrix_shape_and_diagonal(self):
        lib = lib_of([("A", make_vec()), ("B", make_vec()),
                      ("C", make_vec())])
        M = matching.full_matrix(lib, "calm")
        self.assertEqual(M.shape, (3, 3))
        for i in range(3):
            self.assertEqual(M[i, i], -1.0)

    def test_unknown_clip_id_raises(self):
        lib = lib_of([("A", make_vec())])
        with self.assertRaises(KeyError):
            matching.rank(lib, 99, "momentum")


if __name__ == "__main__":
    unittest.main()
