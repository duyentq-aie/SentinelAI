import json, csv
from pathlib import Path
import numpy as np, cv2
root = Path("data/raw/helmet"); out = Path("experiments/EXP-015-helmet-grouped-3cls"); g = out / "error_gallery"
cases = [c for c in json.load(open(out / "val_error_cases.json")) if c["kind"].startswith("c")]
def gts(path):
    p = root / path; img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR); H, W = img.shape[:2]
    lab = root / Path(path).parent.parent / "labels" / (p.stem + ".txt"); L = []
    for l in lab.read_text().splitlines():
        if l.strip():
            c, x, y, w, h = int(float(l.split()[0])), *map(float, l.split()[1:5]); L.append((c, [(x-w/2)*W, (y-h/2)*H, (x+w/2)*W, (y+h/2)*H]))
    return img, L
tiles = []
for c in cases:
    img, L = gts(c["img"]); b = c["pred"]; cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2; s = max(b[2]-b[0], b[3]-b[1]) * 3.2 + 80
    X1, Y1, X2, Y2 = int(max(0, cx-s/2)), int(max(0, cy-s/2)), int(min(img.shape[1], cx+s/2)), int(min(img.shape[0], cy+s/2))
    crop = img[Y1:Y2, X1:X2].copy(); sc = 360 / max(crop.shape[:2]); crop = cv2.resize(crop, None, fx=sc, fy=sc)
    col = {0: (255, 128, 0), 1: (128, 128, 128), 2: (0, 255, 0)}
    for k, gb in L:
        if k not in (0, 2): continue
        cv2.rectangle(crop, (int((gb[0]-X1)*sc), int((gb[1]-Y1)*sc)), (int((gb[2]-X1)*sc), int((gb[3]-Y1)*sc)), col[k], 2)
    cv2.rectangle(crop, (int((b[0]-X1)*sc), int((b[1]-Y1)*sc)), (int((b[2]-X1)*sc), int((b[3]-Y1)*sc)), (0, 0, 255), 2)
    cv = np.zeros((400, 360, 3), np.uint8); cv[:crop.shape[0], :crop.shape[1]] = crop; cv2.putText(cv, f"{c['id']} {c['video']} conf {c['conf']}", (4, 392), 0, 0.45, (255, 255, 0), 1); tiles.append(cv)
while len(tiles) % 4: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(g / "c_with_all_gt_heads.jpg"), np.vstack([np.hstack(tiles[i:i+4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
