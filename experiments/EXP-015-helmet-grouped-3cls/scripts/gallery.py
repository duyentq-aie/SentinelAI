import json
from pathlib import Path
import numpy as np, cv2
root = Path("data/raw/helmet"); out = Path("experiments/EXP-015-helmet-grouped-3cls"); g = out / "error_gallery"; g.mkdir(exist_ok=True)
cases = json.load(open(out / "val_error_cases.json"))
for i, c in enumerate(cases): c["id"] = f"E{i:02d}"
json.dump(cases, open(out / "val_error_cases.json", "w"), indent=1)
def tile(c):
    img = cv2.imdecode(np.fromfile(str(root / c["img"]), np.uint8), cv2.IMREAD_COLOR); H, W = img.shape[:2]
    bs = [b for b in (c["gt"], c["pred"]) if b]; x1 = min(b[0] for b in bs); y1 = min(b[1] for b in bs); x2 = max(b[2] for b in bs); y2 = max(b[3] for b in bs)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2; s = max(x2 - x1, y2 - y1) * 2.6 + 60
    X1, Y1, X2, Y2 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
    crop = img[Y1:Y2, X1:X2].copy(); sc = 360 / max(crop.shape[:2]); 
    crop = cv2.resize(crop, None, fx=sc, fy=sc)
    def dr(b, col, txt):
        p1 = (int((b[0] - X1) * sc), int((b[1] - Y1) * sc)); p2 = (int((b[2] - X1) * sc), int((b[3] - Y1) * sc))
        cv2.rectangle(crop, p1, p2, col, 2); cv2.putText(crop, txt, (p1[0], max(12, p1[1] - 4)), 0, 0.5, col, 1)
    if c["gt"]: dr(c["gt"], (0, 255, 0), "GT no-helmet")
    if c["pred"]: dr(c["pred"], (0, 0, 255), f"pred {c['conf']}")
    canvas = np.zeros((400, 360, 3), np.uint8); canvas[:crop.shape[0], :crop.shape[1]] = crop
    cv2.putText(canvas, f"{c['id']} {c['video']}", (4, 392), 0, 0.5, (255, 255, 0), 1)
    return canvas
for kind in sorted({c["kind"] for c in cases}):
    cs = [c for c in cases if c["kind"] == kind]
    for k in range(0, len(cs), 12):
        tiles = [tile(c) for c in cs[k:k + 12]]
        while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]
        cv2.imwrite(str(g / f"{kind}_{k // 12 + 1}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
print(sorted(p.name for p in g.iterdir()))
