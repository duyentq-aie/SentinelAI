# %% [markdown]
# # SentinelAI - helmet detector re-training on a video-grouped split (PARAMETERISED)
# Generated into a notebook by `tools/kaggle_train.py kernel`. Everything that matters is read from the attached dataset
# (`params.json` = hyper-parameters of the old checkpoint, `manifest.csv` = which image is in which split, `data.yaml` = 3 classes).
# Nothing is hard-coded here: no user names, no paths other than /kaggle/*.

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.26"], check=True)    # same version as the old checkpoint

# %%
import csv, glob, hashlib, json, os, platform, shutil
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO

# --- run-level switches (the only things to edit) ------------------------------------------------------------------------------------
RUN_NAME = "helmet_v2_split_A"          # output folder name
SMOKE = False                           # True: 2 epochs on a quarter of the data, to check the plumbing before the real run
VERIFY_SHA = True                       # re-hash every image against manifest.csv

# %%
data_dirs = [Path(p).parent for p in glob.glob("/kaggle/input/**/params.json", recursive=True)]
assert len(data_dirs) == 1, f"expected exactly one attached dataset with params.json, found {data_dirs}"
D = data_dirs[0]
P = json.loads((D / "params.json").read_text(encoding="utf-8"))
params, meta = dict(P["params"]), P["meta"]
names = meta["names"]
print("dataset:", D, "| classes:", names, "| source checkpoint:", meta["source_checkpoint"], meta["ultralytics_version"])
assert ultralytics.__version__ == meta["ultralytics_version"], (ultralytics.__version__, meta["ultralytics_version"])
assert len(names) == 3, "this recipe is the 3-class one (helmet, motorcycle, no-helmet)"

# %%
rows = list(csv.DictReader(open(D / "manifest.csv", encoding="utf-8")))
counts = {s: sum(r["split"] == s for r in rows) for s in ("train", "valid", "test")}
print("manifest:", counts, "| manifest sha256:", hashlib.sha256((D / "manifest.csv").read_bytes()).hexdigest())
if VERIFY_SHA:
    bad = [r["path"] for r in rows if hashlib.sha256((D / r["split"] / "images" / Path(r["path"]).name).read_bytes()).hexdigest() != r["sha256"]]
    assert not bad, f"{len(bad)} images differ from the manifest, e.g. {bad[:3]}"
    print("all", len(rows), "images match the manifest")
for s in ("train", "valid", "test"):                                                  # the grouped split must really be grouped
    vids = {r["video"] for r in rows if r["split"] == s}
    others = {r["video"] for r in rows if r["split"] != s}
    assert not (vids & others), f"videos in several splits: {vids & others}"

# %%
cfg = {"path": str(D), "train": "train/images", "val": "valid/images", "test": "test/images", "nc": len(names), "names": {int(k): v for k, v in names.items()}}
Path("/kaggle/working/data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
if SMOKE:
    params.update(epochs=2, fraction=0.25, patience=2)
env = {"torch": torch.__version__, "ultralytics": ultralytics.__version__, "python": platform.python_version(),
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "run_name": RUN_NAME, "smoke": SMOKE,
       "params_sha256": hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()}
print(env)
assert torch.cuda.is_available(), "enable a GPU accelerator for this kernel"

# %%
model = YOLO(meta["base_model"])
model.train(data="/kaggle/working/data.yaml", project="/kaggle/working/runs", name=RUN_NAME, device=0, exist_ok=False, **params)

# %%
best = Path("/kaggle/working/runs") / RUN_NAME / "weights" / "best.pt"
m = YOLO(str(best))
result = {"env": env, "manifest_counts": counts}
for split in ("val", "test"):
    r = m.val(data="/kaggle/working/data.yaml", split=split, project="/kaggle/working/runs", name=f"{RUN_NAME}_{split}", plots=True)
    result[split] = {"map50": float(r.box.map50), "map50_95": float(r.box.map), "precision": float(r.box.mp), "recall": float(r.box.mr),
                     "per_class_map50_95": {names[str(i)] if str(i) in names else str(i): float(v) for i, v in zip(r.box.ap_class_index, r.box.maps[r.box.ap_class_index])}}
Path("/kaggle/working/result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
shutil.copy(best, "/kaggle/working/helmet_v2_best.pt")
print(json.dumps(result, indent=2))
