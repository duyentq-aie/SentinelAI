"""No-helmet and overload by voting over a sliding window of observations (P2 3.2). Replaces the legacy running counter.

Per motorcycle track, every frame in which at least one head is attached to it is one *observation* (frames without any head, e.g.
a rider occluded for a moment, are not observations: they neither vote nor push old votes out of the window). The window keeps the last
`window` observations. A violation holds while, inside the window,

    votes >= min_votes   and   votes / observations >= min_ratio

where a vote is "a no-helmet head is attached" (helmet rule) or "more than max_passengers heads are attached" (overload rule).
A 1-2 frame detector glitch cannot reach min_votes; a short occlusion does not dilute the ratio. The recorder opens at most one event per
(track, type) (`evidence.one_event_per_track`), so the verdict flickering around the threshold cannot create duplicates.
Parameters are not tuned on ground truth yet: "chưa kiểm định".
"""
from collections import defaultdict, deque

from sentinel.rules.legacy import LegacyRules

PURGE_EVERY = 300          # frames
PURGE_AFTER = 300          # drop the window of a track not seen for this many frames


class WindowedRules(LegacyRules):
    def __init__(self, rules_cfg, camera):
        super().__init__(rules_cfg, camera)
        self._maxlen = max(rules_cfg.helmet.window, rules_cfg.overload.window)
        self._obs = defaultdict(lambda: deque(maxlen=self._maxlen))   # track id -> observations (no_helmet_vote, overload_vote)
        self._last_seen = {}
        self._on = {}                        # (track id, 'helmet' | 'overload') -> flagged now (only used for hysteresis)
        self._frame = 0

    def reset(self):
        super().reset()
        self._obs.clear()
        self._on.clear()
        self._last_seen.clear()
        self._frame = 0

    def step(self, p):
        self._frame += 1
        result = super().step(p)
        if self._frame % PURGE_EVERY == 0:
            for tid in [t for t, f in self._last_seen.items() if self._frame - f > PURGE_AFTER]:
                self._obs.pop(tid, None)
                self._on.pop((tid, "helmet"), None)
                self._on.pop((tid, "overload"), None)
                del self._last_seen[tid]
        return result

    @staticmethod
    def holds(votes, n, min_votes, min_ratio):
        return n > 0 and votes >= min_votes and votes / n >= min_ratio

    def _verdict(self, key, votes, n, rule):
        """Entry needs ratio >= min_ratio; with exit_ratio set the flag stays on while ratio >= exit_ratio (no flicker around the threshold)."""
        on = self._on.get(key, False)
        thr = rule.exit_ratio if (on and rule.exit_ratio is not None) else rule.min_ratio
        on = self.holds(votes, n, rule.min_votes, thr)
        self._on[key] = on
        return on

    def _judge_motorcycle(self, track_id, n_heads, no_helmet_now):
        h, o = self.cfg.helmet, self.cfg.overload
        self._last_seen[track_id] = self._frame
        window = self._obs[track_id]
        if n_heads >= 1:
            window.append((no_helmet_now, n_heads > o.max_passengers))
        # each rule looks at the last `window` observations of ITS OWN size
        obs = list(window)
        hv = obs[-h.window:]
        ov = obs[-o.window:]
        helmet = self._verdict((track_id, "helmet"), sum(v[0] for v in hv), len(hv), h)
        overload = self._verdict((track_id, "overload"), sum(v[1] for v in ov), len(ov), o)
        if helmet and overload:
            return True, "OVERLOAD & NO-HELMET"
        if overload:
            return True, "OVERLOAD"
        if helmet:
            return True, "NO-HELMET"
        return False, ""


def make_rules(rules_cfg, camera):
    """Rule engine selected by `rules.engine`: "windowed" (default) or "legacy" (v0 behaviour, kept for baseline replays)."""
    if rules_cfg.engine == "legacy":
        return LegacyRules(rules_cfg, camera)
    if rules_cfg.engine == "windowed":
        return WindowedRules(rules_cfg, camera)
    raise ValueError(f"unknown rules.engine {rules_cfg.engine!r} (legacy | windowed)")
