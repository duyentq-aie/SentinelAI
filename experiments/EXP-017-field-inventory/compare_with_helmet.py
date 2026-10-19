"""dHash of 1 frame per second of test_video.mp4 vs the 1460 helmet images (same hash as EXP-013 / tools/leakage_check.py). Read-only."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "tools")
import leakage_check as lc  # noqa: E402

root = Path("data/raw/helmet")
imgs = []
for s in lc.SPLITS:
    for r in lc.load_split(root, s):
        if r["hash"] is not None:
            imgs.append({**r, "split": s})
H = np.array([r["hash"] for r in imgs], dtype=np.uint64)

cap = cv2.VideoCapture("data/raw/field/test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frames = []
for sec in range(int(n / fps) + 1):
    idx = int(sec * fps)
    if idx >= n:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    if ok:
        frames.append((sec, idx, fr, lc.dhash(fr)))
FH = np.array([f[3] for f in frames], dtype=np.uint64)
D = lc.hamming_matrix(FH, H)
best_d, best_i = D.min(axis=1), D.argmin(axis=1)

# context: how close are helmet images to each OTHER when they come from different videos (what "same scene" vs "other scene" looks like)
rng = np.random.default_rng(0)
vid = np.array([r["video"] or "?" for r in imgs])
ctx = []
for i in rng.choice(len(imgs), 300, replace=False):
    m = lc.hamming_matrix(H[i:i + 1], H)[0]
    m[vid == vid[i]] = 99
    ctx.append(int(m.min()))
ctx_same = []
for i in rng.choice(len(imgs), 300, replace=False):
    m = lc.hamming_matrix(H[i:i + 1], H)[0]
    m[i] = 99
    ctx_same.append(int(m.min()))

q = lambda v: {"min": int(np.min(v)), "q25": float(np.quantile(v, .25)), "median": float(np.median(v)), "q75": float(np.quantile(v, .75)), "max": int(np.max(v))}
res = {"frames_tested": len(frames), "hash": "dHash 64 bit (9x8 grayscale), Hamming distance",
       "video_frame_to_nearest_helmet_image": q(best_d) | {"n_le_4": int((best_d <= 4).sum()), "n_le_8": int((best_d <= 8).sum()), "n_le_12": int((best_d <= 12).sum())},
       "context_helmet_image_to_nearest_image_of_ANOTHER_video": q(ctx), "context_helmet_image_to_nearest_image_any_video": q(ctx_same),
       "per_second": [{"second": f[0], "frame": f[1], "nearest_distance": int(d), "nearest_image": imgs[i]["name"], "nearest_video": imgs[i]["video"]} for f, d, i in zip(frames, best_d, best_i)]}
Path("experiments/EXP-017-field-inventory/compare_with_helmet.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(json.dumps({k: v for k, v in res.items() if k != "per_second"}, indent=1))
print("distances per second:", [int(x) for x in best_d])

order = np.argsort(best_d)[:5]
rows = []
for k in order:
    sec, idx, fr, _ = frames[k]
    im = cv2.imdecode(np.fromfile(str(root / imgs[best_i[k]]["path"]), np.uint8), cv2.IMREAD_COLOR)
    a = cv2.resize(fr, (640, 360)); b = cv2.resize(im, (640, 360))
    cv2.putText(a, f"test_video t={sec}s  d={int(best_d[k])}", (6, 22), 0, 0.6, (0, 255, 255), 2)
    cv2.putText(b, f"helmet/{imgs[best_i[k]]['split']}/{imgs[best_i[k]]['video']}", (6, 22), 0, 0.6, (0, 255, 255), 2)
    rows.append(np.hstack([a, b]))
cv2.imwrite("experiments/EXP-017-field-inventory/closest_5_pairs.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
