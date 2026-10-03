# ClipEngine

Finds which clips in a video library cut well together, for short vertical edits.

I built it to find cuts in a large footage library without scrubbing every clip by hand. It stays local: no cloud APIs, no telemetry, and the UI binds to 127.0.0.1. [GUIDE.md](GUIDE.md) covers daily use and editing in Final Cut Pro or DaVinci Resolve.

## How it works

Each clip gets a start and an end state: optical flow and color over the 1.2 s at each end, at any frame rate. Above 30 fps the window skips whole frames, so kept frames sit 1/30 to 1/15 s apart, at most 36 of them. Log footage is normalized before color is measured, except luma, which keeps the shot's exposure. Motion is in frame widths per second, so cameras and resolutions compare directly. A cut is the end of clip A against the start of clip B, and each mode gates on physics:

| mode | cut | gate |
|---|---|---|
| momentum | motion carries through | both sides moving |
| whip | blur to blur | both near whip speed |
| calm | still to still | both near static |
| contrast | color and brightness flip | none |

## Decisions

- I detect changed files by size and mtime instead of a hash, because hashing reads every byte, and reading a file evicted to the cloud downloads it.
- I store each clip as one float32 vector with 50 slots, so the library is one numpy matrix and scoring is vectorized. Motion labels come from the vector, so after a threshold change `relabel` recomputes them in seconds without decoding video.
- I chain clips with a beam search of width 12, because greedy walks into dead ends and exhaustive search is factorial.
- I require translation of at least 0.75x the jitter for a pan, tilt or whip label, because an editor cannot cut on a drifting walking shot.
- I kept dependencies to numpy and OpenCV, whose wheel bundles its own decoder, so nothing installs outside the venv, not even ffmpeg.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_tests.py
export CLIPENGINE_MEDIA_ROOT="/path/to/footage"
.venv/bin/python -m clipengine scan
.venv/bin/python -m clipengine analyze
.venv/bin/python -m clipengine ui
```

Footage goes in `<root>/<camera profile>/<country>/`. Profiles named in `clipengine/config.py` are treated as log, and other folders fall back to automatic flat detection. `audit` shows how the motion thresholds split your footage, and `relabel` applies a change to them.

`help` lists every command, and `help <command>` or `<command> --help` shows one. Options are spelled out in full as `--name=value` or `--name value`. Exit status is 0 for success, 1 when the command failed and 2 for a usage error. A usage error exits before anything runs.

Analysis changes need a `FEATURE_VERSION` bump in config.py. After a bump every clip is pending and the grid stays empty until a full `analyze`. Clips evicted to iCloud stay out until downloaded and analyzed.

The 468 tests take about 20 s. Their media are synthetic clips with known motion and patched mp4 boxes.

## Layout

| path | role |
|---|---|
| `clipengine/catalog.py` | sqlite catalog, scan by file stats only, moved files keep their analysis |
| `clipengine/analysis.py` | motion and color features |
| `clipengine/matching.py` | cut scoring |
| `clipengine/sequence.py` | beam search, json and m3u8 export |
| `clipengine/fcpxml.py` | the timeline for Final Cut Pro and Resolve |
| `clipengine/web/` | localhost UI |
| `tests/` | unittest suite |

## Limits

- Analysis takes about 10 s per clip and only reads downloaded files. It stats each clip again before decoding, so a clip evicted since the scan is skipped. Files over 20 GiB are skipped.
- Whip mode needs footage shot with deliberate whips.
- Log normalization is one generic stretch and saturation boost, not a LUT for each camera.

## License

MIT.
