import csv, json, collections
from pathlib import Path
root = Path("data/raw/helmet")
rows = list(csv.DictReader(open("experiments/EXP-013-splits/A/manifest.csv", encoding="utf-8")))
def folder(r): return "extra_1080p" if r["video"].startswith("other:") else {"train":"train","valid":"val","test":"test"}[r["split"]]
def counts(r):
    lab = root / Path(r["path"]).parent.parent / "labels" / (Path(r["path"]).stem + ".txt")
    c = collections.Counter(int(float(l.split()[0])) for l in lab.read_text().splitlines() if l.strip())
    return c
out = {}
def summarize(items):
    nh = collections.Counter(); people = collections.Counter(); moto = collections.Counter(); nh_ge2 = 0; per_moto = None
    for c in items:
        nh[c[2]] += 1; people[c[0] + c[2]] += 1; moto[c[1]] += 1
    return {"images": len(items), "no_helmet_per_image": dict(sorted(nh.items())), "images_with_ge2_no_helmet": sum(v for k, v in nh.items() if k >= 2),
            "images_with_no_no_helmet": nh[0], "heads_per_image(helmet+no-helmet)": dict(sorted(people.items())), "motorcycles_per_image": dict(sorted(moto.items())),
            "mean_heads_per_motorcycle": round(sum(c[0] + c[2] for c in items) / max(1, sum(c[1] for c in items)), 3)}
for scheme, key in (("original_split", lambda r: r["orig_split"]), ("split_A", folder)):
    g = collections.defaultdict(list)
    for r in rows: g[key(r)].append(counts(r))
    out[scheme] = {k: summarize(v) for k, v in g.items()}
json.dump(out, open("experiments/EXP-015-helmet-grouped-3cls/label_stats.json", "w"), indent=1)
for scheme in out:
    print(scheme)
    for k, v in out[scheme].items():
        print(" ", k, v["images"], "no-helmet/img", v["no_helmet_per_image"], ">=2:", v["images_with_ge2_no_helmet"], "heads/img", v["heads_per_image(helmet+no-helmet)"], "heads/moto", v["mean_heads_per_motorcycle"])
