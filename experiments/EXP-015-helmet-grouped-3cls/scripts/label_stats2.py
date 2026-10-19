import csv, json, collections
from pathlib import Path
root = Path("data/raw/helmet")
rows = list(csv.DictReader(open("experiments/EXP-013-splits/A/manifest.csv", encoding="utf-8")))
def folder(r): return "extra_1080p" if r["video"].startswith("other:") else {"train":"train","valid":"val","test":"test"}[r["split"]]
res = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    lab = root / Path(r["path"]).parent.parent / "labels" / (Path(r["path"]).stem + ".txt")
    bx = [(int(float(l.split()[0])), *map(float, l.split()[1:5])) for l in lab.read_text().splitlines() if l.strip()]
    motos = [b for b in bx if b[0] == 1]; heads = [b for b in bx if b[0] in (0, 2)]
    owner = collections.defaultdict(list)
    for h in heads:
        best = None
        for i, m in enumerate(motos):
            if abs(h[1] - m[1]) <= m[3] / 2 and abs(h[2] - m[2]) <= m[4] / 2: best = i; break
        owner[best].append(h[0])
    f = folder(r)
    for i in range(len(motos)):
        hs = owner.get(i, []); res[f]["motos"] += 1
        res[f][f"heads={len(hs)}"] += 1
        res[f][f"no-helmet={hs.count(2)}"] += 1
        if hs.count(2) >= 2: res[f]["motos_with_>=2_no-helmet"] += 1
        if hs.count(2) >= 1 and hs.count(0) >= 1: res[f]["motos_mixed_helmet_and_no-helmet"] += 1
    res[f]["heads_not_inside_any_moto_box"] += len(owner.get(None, []))
for f, c in res.items(): print(f, dict(sorted(c.items())))
json.dump({f: dict(c) for f, c in res.items()}, open("experiments/EXP-015-helmet-grouped-3cls/label_stats_per_motorcycle.json", "w"), indent=1)
