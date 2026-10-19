import zipfile, hashlib, shutil
from pathlib import Path
dst = Path("experiments/EXP-015-helmet-grouped-3cls")
assert not dst.exists()
z = zipfile.ZipFile("results_3cls.zip")
for i in z.infolist():
    n = i.filename
    if i.is_dir() or ".." in n or n.startswith("/"): continue
    if n.endswith("weights/last.pt") or n.endswith("weights/best.pt"): continue
    t = dst / n; t.parent.mkdir(parents=True, exist_ok=True); t.write_bytes(z.read(n))
(dst / "weights").mkdir(exist_ok=True)
best = z.read("helmet_3cls_splitA/weights/best.pt"); last = z.read("helmet_3cls_splitA/weights/last.pt")
(dst / "weights" / "best.pt").write_bytes(best)
print("best sha256", hashlib.sha256(best).hexdigest(), "| identical to last:", best == last, "| bytes", len(best))
print("zip sha256", hashlib.sha256(Path("results_3cls.zip").read_bytes()).hexdigest())
