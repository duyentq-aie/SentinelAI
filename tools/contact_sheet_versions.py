"""Row-by-row contact sheet for several versions of the pipeline on the same video: one row per event, one crop per version (same time +-1 s, overlapping boxes).

  python tools/contact_sheet_versions.py --video test_video.mp4 --out outputs/regression_diag --prefix contact4 \\
      --rows outputs/regression_diag/event_rows.json --v0 outputs/regression_diag/v0/labels.json \\
      --run "current + old model=outputs/review_A" --run "EXP-015 3cls=outputs/review_B" --run "EXP-016 4cls=outputs/review_C_4cls"

`--rows` (optional) = the rows of an earlier sheet (event_rows.json): their numbers are kept so a grade given earlier stays valid; events no earlier row explains get
the next numbers. The first column is the original v0 code (labels.json from the one-pass hook run); every `--run` is a tools/review_run.py output folder.
Writes <prefix>_<page>.jpg (5 rows per page, gitignored folder: people are visible), <prefix>_rows.json and <prefix>_grading.csv (empty verdict columns).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FPS = 30.0
H, W = 260, 280


def iou(a, b):
    ix, iy = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u else 0.0


def centre_in(outer, inner):
    cx, cy = (inner[0] + inner[2]) / 2, (inner[1] + inner[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def related(a, b, tol=30):
    if a["type"] != b["type"] or abs(a["frame"] - b["frame"]) > tol:
        return 0.0
    v = iou(a["box"], b["box"])
    return v if (v >= 0.2 or centre_in(a["box"], b["box"]) or centre_in(b["box"], a["box"])) else 0.0


def load_v0(path):
    out, seen = [], set()
    for l in json.loads(Path(path).read_text(encoding="utf-8"))["first_seen"]:
        m = l["label"]
        if not m.startswith("ID:"):
            continue
        tid = m.split()[0][3:]
        for typ, key in (("no_helmet", "NO-HELMET"), ("overload", "OVERLOAD"), ("wrong_way", "WRONG-WAY")):
            if key in m and (tid, typ) not in seen:
                seen.add((tid, typ))
                out.append({"track": tid, "type": typ, "frame": l["frame"], "box": l["box"]})
    return out


def load_run(folder):
    last = {}
    for line in (Path(folder) / "evidence" / "events.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["id"]] = r
    return [{"track": r["track_id"], "type": r["type"], "frame": r["trigger_frame"], "box": r["key_box"], "start": r["frame_start"], "end": r["frame_end"]}
            for r in last.values() if r.get("status") != "evicted"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="contact4")
    ap.add_argument("--rows", help="event_rows.json of an earlier sheet (keeps its row numbers)")
    ap.add_argument("--v0", required=True)
    ap.add_argument("--run", action="append", required=True, help='"label=review folder", repeatable')
    a = ap.parse_args()
    out = Path(a.out)
    versions = {"v0": load_v0(a.v0)}
    for spec in a.run:
        label, folder = spec.split("=", 1)
        versions[label] = load_run(folder)
    names = list(versions)

    rows = []                                              # {"n", "type", "frame", "box", "ev": {version: event or None}}
    if a.rows:
        old = json.loads(Path(a.rows).read_text(encoding="utf-8"))
        v0_by = {(e["track"], e["type"]): e for e in versions["v0"]}
        for r in old:
            ev = {n: None for n in names}
            if r["v0"]:
                ev["v0"] = v0_by.get((r["v0"]["track"], r["type"]))
            rows.append({"n": r["n"], "type": r["type"], "ev": ev})
        # the earlier sheet's `current` column is the first --run (current pipeline + the old model) when it is the same folder
        first = names[1]
        cur_by = {(e["track"], e["type"]): e for e in versions[first]}
        for r, row in zip(old, rows):
            if r["current"]:
                row["ev"][first] = cur_by.get((r["current"]["track"], r["type"]))
    for name in names:                                     # place every event of every version into the best related row, else a new row
        placed = {id(row["ev"][name]) for row in rows if row["ev"].get(name)}
        for e in sorted(versions[name], key=lambda e: e["frame"]):
            if id(e) in placed:
                continue
            best, bi = 0.0, None
            for i, row in enumerate(rows):
                if row["ev"].get(name) is None and row["type"] == e["type"]:
                    rep = next((x for x in row["ev"].values() if x), None)
                    v = related(e, rep) if rep else 0.0
                    if v > best:
                        best, bi = v, i
            if bi is None:
                rows.append({"n": max([r["n"] for r in rows] + [0]) + 1, "type": e["type"], "ev": {n: None for n in names}})
                bi = len(rows) - 1
            rows[bi]["ev"][name] = e
    for row in rows:
        row["frame"] = min(e["frame"] for e in row["ev"].values() if e)
    rows.sort(key=lambda r: (r["frame"], r["n"]))

    cap = cv2.VideoCapture(a.video)

    def crop(e, label, col):
        cap.set(cv2.CAP_PROP_POS_FRAMES, e["frame"])
        ok, f = cap.read()
        f = cv2.resize(f, (1280, 720))
        x1, y1, x2, y2 = e["box"]
        w, h = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = int(max(0, x1 - 0.25 * w)), int(max(0, y1 - 0.25 * h)), int(min(1280, x2 + 0.25 * w)), int(min(720, y2 + 0.25 * h))
        c = f[Y1:Y2, X1:X2].copy()
        cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), col, 2)
        c = cv2.resize(c, (max(40, int(c.shape[1] * H / c.shape[0])), H))[:, :W]
        cv2.putText(c, f"{label}: {e['track']} f{e['frame']}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        return cv2.copyMakeBorder(c, 0, 0, 0, W - c.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))

    palette = [(0, 255, 255), (0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 165, 255)]

    def row_img(r):
        head = np.full((H, 170, 3), 20, np.uint8)
        present = sum(1 for e in r["ev"].values() if e)
        for k, s in enumerate([f"#{r['n']}", f"t={r['frame'] / FPS:.2f}s", r["type"], f"{present}/{len(names)} versions"]):
            cv2.putText(head, s, (8, 30 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9 if k == 0 else 0.6, (0, 255, 0) if present == len(names) else (0, 165, 255), 2)
        cols = []
        for k, n in enumerate(names):
            e = r["ev"][n]
            if e:
                cols.append(crop(e, n, palette[k % len(palette)]))
            else:
                b = np.full((H, W, 3), 30, np.uint8)
                cv2.putText(b, "no event", (60, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(b, n, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                cols.append(b)
        return np.hstack([head] + cols)

    pages = []
    for p in range(0, len(rows), 5):
        f = out / f"{a.prefix}_{p // 5 + 1}.jpg"
        cv2.imwrite(str(f), np.vstack([row_img(r) for r in rows[p:p + 5]]), [cv2.IMWRITE_JPEG_QUALITY, 88])
        pages.append(f.name)
    summary = [{"n": r["n"], "time_s": round(r["frame"] / FPS, 2), "type": r["type"],
                "versions": {n: (None if not e else {"track": e["track"], "frame": e["frame"], **({"frames": [e["start"], e["end"]]} if "start" in e else {})}) for n, e in r["ev"].items()}}
               for r in rows]
    (out / f"{a.prefix}_rows.json").write_text(json.dumps({"versions": names, "rows": summary, "pages": pages}, indent=1), encoding="utf-8")
    with open(out / f"{a.prefix}_grading.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["row", "time_s", "type"] + [f"{n}: track" for n in names] + [f"{n}: verdict (đúng/sai/không rõ)" for n in names])
        for s in summary:
            w.writerow([s["n"], s["time_s"], s["type"]] + [(s["versions"][n] or {}).get("track", "") for n in names] + [""] * len(names))
    print(f"{len(rows)} rows, {len(pages)} pages -> {out}")
    for n in names:
        print(f"  {n}: {sum(1 for r in rows if r['ev'][n])} events in rows")


if __name__ == "__main__":
    main()
