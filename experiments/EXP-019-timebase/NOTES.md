English translation of NOTES.vi.md (original, Vietnamese).

# EXP-019 – parameters expressed in time: invariance check at 30 FPS (2026-10-19)

Change: `sentinel/timebase.py`, configuration in seconds / ratios, `Pipeline.set_fps` (see `docs/TIME_BASED_PARAMETERS.md`). No value was tuned.

Check of the invariant "at 30 FPS the result is identical":
1. **Characterization tests** (`tests/test_rules_offline.py`, baselines v0…v5) and the whole fast test suite: 317 passed.
2. **`diff_events` against baseline v5** (full pipeline on CPU, `tools/record_events.py --impl refactored --device cpu`): `MISSING 0 / ADDED 0 / CHANGED 0`, the two lists of 29 events are equal element by element (frame, type, track, label).
3. **Event time spans** (GPU, `tools/review_run.py`, same video and same weights, before = `outputs/review_A`, after = `outputs/review_A_timebase`): 14 / 14 events are identical in type, track, `frame_start`, `frame_end`, `trigger_frame`, every sampled box (`boxes`), `key_box`, the plate frame and the number of plate attempts; 14 / 14 characterization events identical.
4. **At 10 FPS** (test `tests/test_timebase.py`): every window lasts exactly the same number of seconds (maximum error half a frame from rounding), the violation flag switches on and off at most one frame apart from 30 FPS, the plate-search budget is 3 s (30 attempts instead of 90), events close after 1 s, the tracker buffer is converted from 30 to 10 frames. The effect on the events of a video run at 10 FPS has not been measured yet (PLAN G10 step 4).
