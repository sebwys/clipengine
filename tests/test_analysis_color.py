# test_analysis_color.py
# color stats against constructed colors. warmth uses the lab b* axis
# (yellow to blue), luma the l axis, and the hue histogram is weighted by
# saturation and value so gray pixels cannot fake a palette.

import unittest

import numpy as np

from clipengine import analysis, features
from tests import synth


class TestColorStats(unittest.TestCase):
    def test_warmth_orders_red_over_blue(self):
        warm = analysis.color_stats(
            synth.color_frames((40, 60, 200)), force_log=False)   # reddish
        cool = analysis.color_stats(
            synth.color_frames((200, 120, 40)), force_log=False)  # bluish
        self.assertGreater(warm["warmth"], cool["warmth"] + 20)

    def test_luma_orders_bright_over_dark(self):
        bright = analysis.color_stats(
            synth.color_frames((80, 160, 250)), force_log=False)
        dark = analysis.color_stats(
            synth.color_frames((10, 20, 60)), force_log=False)
        self.assertGreater(bright["luma"], dark["luma"] + 60)

    def test_hue_histogram_peaks_on_green(self):
        stats = analysis.color_stats(
            synth.color_frames((40, 220, 40)), force_log=False)
        peak = int(np.argmax(stats["hue"]))
        # opencv hue for green is ~60 of 180; bin width is 15 -> bin 4
        self.assertIn(peak, (3, 4, 5))
        self.assertAlmostEqual(float(stats["hue"].sum()), 1.0, places=4)

    def test_flat_footage_is_detected_and_stretched(self):
        stats = analysis.color_stats(synth.gray_ramp_frames(), force_log=False)
        self.assertTrue(stats["flat"])
        # raw ramp spans ~30 luma levels; normalization must widen it
        self.assertGreater(stats["contrast"], 100)

    def test_force_log_overrides_detection(self):
        stats = analysis.color_stats(
            synth.color_frames((40, 60, 200)), force_log=True)
        self.assertTrue(stats["flat"])

    def test_gray_footage_falls_back_to_uniform_hue(self):
        stats = analysis.color_stats(synth.gray_ramp_frames(), force_log=False)
        spread = float(stats["hue"].max() - stats["hue"].min())
        self.assertLess(spread, 0.08)
        self.assertEqual(stats["hue"].shape[0], features.HUE_BINS)


if __name__ == "__main__":
    unittest.main()
