"""Frame counts derived from durations: every time-like setting is stored in SECONDS (or as a ratio) and turned into a number of frames from the
frame rate the pipeline really sees (source fps / frame stride). At REFERENCE_FPS the counts are exactly the old hard-coded frame values.

Legacy settings written in frames (`window: 15`, `close_after: 30`, ...) are still read: they are interpreted as frames at REFERENCE_FPS
(= seconds * REFERENCE_FPS) and a warning is issued by the config loader.
"""
import math

REFERENCE_FPS = 30.0
EPS = 1e-9


def to_frames(seconds, fps, minimum=1):
    """Round-half-up number of frames covering `seconds` at `fps` (at least `minimum`): 0.5 s -> 15 frames at 30 FPS, 5 at 10 FPS."""
    return max(minimum, int(math.floor(seconds * fps + 0.5 + EPS)))


def seconds_of(seconds, legacy_frames):
    """The effective duration: a legacy frame count (if given) wins and means frames at REFERENCE_FPS; otherwise the seconds value."""
    return legacy_frames / REFERENCE_FPS if legacy_frames is not None else seconds


def votes_needed(fraction, window_frames):
    """K from the fraction K/W: round-half-up of fraction * W, kept inside 1..W (1/3 of 15 -> 5; of 5 -> 2)."""
    return max(1, min(window_frames, int(math.floor(fraction * window_frames + 0.5 + EPS))))


def window_votes(rule, fps):
    """-> (W, K) in frames for a helmet / overload rule config at `fps`.

    Legacy `window` / `min_votes` (frames) are converted: seconds = window / 30, fraction = min_votes / window."""
    if rule.window is not None or rule.min_votes is not None:
        w_ref = rule.window if rule.window is not None else to_frames(rule.window_s, REFERENCE_FPS)
        k_ref = rule.min_votes if rule.min_votes is not None else votes_needed(rule.min_vote_fraction, w_ref)
        seconds, fraction = w_ref / REFERENCE_FPS, k_ref / w_ref
    else:
        seconds, fraction = rule.window_s, rule.min_vote_fraction
    w = to_frames(seconds, fps)
    return w, votes_needed(fraction, w)
