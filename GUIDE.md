# ClipEngine guide

Daily use and the steps into Final Cut Pro or DaVinci Resolve. [README.md](README.md) covers how it works.

## 1. Run

```bash
.venv/bin/python -m clipengine scan
.venv/bin/python -m clipengine analyze
.venv/bin/python -m clipengine ui
```

Rerun `scan` and `analyze` after adding or downloading footage. The grid shows analyzed clips only, and decode failures show in `status`. A clip shot without log in a log profile folder comes out oversaturated, so move it.

Open http://127.0.0.1:8763 in Safari, which plays HEVC and ProRes previews that Chrome may not. Add `--port=8764` if the port is taken.

## 2. Build a sequence

1. Pick a mode in the header, then a clip in the grid. Filter by country, move or name.
2. Click find matches. A score near 1.0 means the cut should feel invisible or intentional. The four bars are motion, energy, color and luma.
3. Add clips with + tray, or set a length and click `auto-build` to chain from the selected clip.
4. Click export once the tray holds at least two clips.

| mode | use it for |
|---|---|
| momentum | walking or driving energy (default) |
| whip | fast transitions between places |
| calm | intros, outros, ambience |
| contrast | section changes, day to night |

The country options keep one place (same) or force variety (different, travel). Trust the motion bar over the color bar. For whip mode, end a shot with a hard whip and start the next whipping the same way.

## 3. Export

Export writes three files to `exports/`:

| file | use |
|---|---|
| `sequence_<stamp>.json` | cut plan with scores for each cut |
| `sequence_<stamp>.m3u8` | rough preview in IINA or VLC |
| `sequence_<stamp>.fcpxml` | timeline for Final Cut Pro or Resolve |

The fcpxml points at your original files and copies nothing. A clip that is not downloaded imports as offline media: download it, run `scan`, and export again.

## 4. Final Cut Pro

1. File > Import > XML and pick the `.fcpxml`. Clips land end to end in an event named ClipEngine.
2. Apple Log is recognized automatically. For Sony, set Camera LUT to S-Log3/S-Gamut3.Cine in the Info inspector. For DJI, apply a D-Log M LUT.
3. Keep about a second on each side of every cut, where the match was scored.
4. For whip cuts, add a 2 to 4 frame overlap and a short speed ramp.
5. For vertical, duplicate the project, set the format to 1080x1920 and start from Smart Conform.

## 5. DaVinci Resolve

1. File > Import Timeline > Import AAF, EDL, XML, FCPXML and pick the `.fcpxml`. Keep "automatically import source clips" checked and relink offline clips to your footage root.
2. Use DaVinci YRGB Color Managed, or a Color Space Transform per camera group: S-Log3, D-Log M or Apple Log in, Rec.709 Gamma 2.4 out.
3. Deliver at 1080x1920, H.264 High at 10 to 15 Mbps, AAC audio.
