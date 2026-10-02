# ClipEngine

Finds which clips in a video library cut well together, for short vertical edits.

I built it to find cuts in a large footage library without scrubbing every clip by hand. It stays local: no cloud APIs, no telemetry, and the UI binds to 127.0.0.1. [GUIDE.md](GUIDE.md) covers daily use and editing in Final Cut Pro or DaVinci Resolve.

## How it works

Each clip gets a start and an end state: optical flow and color over its first and last 1.2 s, with log footage normalized before color is measured. Motion is in frame widths per second, so cameras and resolutions compare directly. A cut is the end of clip A against the start of clip B, and each mode gates on physics:

| mode | cut | gate |
|---|---|---|
| momentum | motion carries through | both sides moving |
| whip | blur to blur | both near whip speed |
| calm | still to still | both near static |
| contrast | color and brightness flip | none |

## Decisions

- I key the catalog on size and mtime instead of a hash, because hashing a file evicted to the cloud downloads it.
- I store each clip as a 50 slot float32 vector, so the library is one numpy matrix and scoring is vectorized. Labels are derived from it, so `relabel` retunes them in seconds.
- I chain clips with a beam search of width 12, because greedy walks into dead ends and exhaustive search is factorial.
- I require translation of at least 0.75x the jitter for a pan, tilt or whip label, because an editor cannot cut on a drifting walking shot.
- I kept dependencies to numpy and OpenCV, so nothing installs outside the venv.

## Run

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
export CLIPENGINE_MEDIA_ROOT="/path/to/footage"
.venv/bin/python -m clipengine scan
.venv/bin/python -m clipengine analyze
.venv/bin/python -m clipengine ui
.venv/bin/python run_tests.py
```

Footage goes in `<root>/<camera profile>/<country>/`. Profiles named in `clipengine/config.py` are treated as log. `help` lists every command. Analysis changes need a `FEATURE_VERSION` bump. The 85 tests take about 9 s and use synthetic clips with known motion.

## Layout

| path | role |
|---|---|
| `clipengine/catalog.py` | sqlite catalog, scan by file stats only |
| `clipengine/analysis.py` | motion and color features |
| `clipengine/matching.py` | cut scoring |
| `clipengine/sequence.py` | beam search; json, m3u8 and fcpxml export |
| `clipengine/web/` | localhost UI |
| `tests/` | unittest suite |

## Limits

- Analysis takes about 10 s per clip and only reads downloaded files. Files over 20 GiB are skipped.
- Whip mode needs footage shot with deliberate whips.
- Log normalization is one fixed stretch for all cameras.

## License

MIT.
