"""Combine several readings of the same plate (different frames) into one: character-wise weighted vote."""
from collections import Counter, defaultdict


def vote(reads):
    """reads: [(text, score)] of NORMALISED plate texts (None / empty are ignored), score in 0..1.

    1. keep the most supported text length (sum of scores); 2. vote every character position among texts of that length, weighted by score.
    -> {"text", "score" (mean agreement 0..1), "n_reads", "unanimous"} or None when there is nothing to vote on.
    """
    reads = [(t, max(float(s), 1e-3)) for t, s in reads if t]
    if not reads:
        return None
    by_len = defaultdict(float)
    for t, s in reads:
        by_len[len(t)] += s
    L = max(by_len, key=by_len.get)
    cands = [(t, s) for t, s in reads if len(t) == L]
    out, agree = [], []
    for i in range(L):
        c = Counter()
        for t, s in cands:
            c[t[i]] += s
        ch, w = c.most_common(1)[0]
        out.append(ch)
        agree.append(w / sum(c.values()))
    return {"text": "".join(out), "score": round(sum(agree) / L, 4), "n_reads": len(cands), "unanimous": len({t for t, _ in cands}) == 1}
