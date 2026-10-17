# tests/fixtures

| file | what | produced by |
|---|---|---|
| `tracks_test_video.jsonl` | per-frame perception (tracks + light state) of `test_video.mp4`, models on the **raw** frame (since 3.0b); CPU | `tools/record_tracks.py` |
| `tracks_test_video_v0_overlay.jsonl` | same, recorded before 3.0b (models saw drawn overlays); kept for `tools/diff_tracks.py` and the legacy-port test | idem, at commit `a3f89de` |
| `baseline_events_v1_clean.json` | 44 events (23 label + 21 plate capture) of the 3.0b pipeline: **the current comparison baseline** | `tools/record_tracks.py --events` |
| `baseline_events_v0_cpu.json` | 43 events of the original `app.py` (tag `v0-baseline`) on CPU; historical comparison baseline | `tools/record_events.py --impl original --device cpu` |

Baselines are *comparison snapshots* (`tools/diff_events.py`), not pass conditions for rule changes. A tracks fixture is only valid for the
video and weights whose sha256 are in its first line; the offline tests check that. Track ids are device-specific (fixtures are CPU).
