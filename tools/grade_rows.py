"""Quick by-eye grading of the EXP-020 contact-sheet rows, one row at a time, in an OpenCV window.

  python tools/grade_rows.py                 # grade (resumes at the first row without a verdict)
  python tools/grade_rows.py --report        # only compute the precision table from grading_rows.csv and write it into the EXP-020 README

Rows come from experiments/EXP-020-regression-diag/event_rows_5_versions.json (the 5-version contact sheet); wrong_way rows are skipped (that rule is off), which leaves the no_helmet / overload rows.
For every row the window shows the crop of every version that has an event in that row (>= 300 px high, side by side, with the version name), a reduced whole frame with the
vehicles circled, and "row N, time, type, row X/total".
Keys:  d = true violation   s = wrong (helmet worn / wrong person)   k = unclear   n = the crops of this row are DIFFERENT people: then every version of the row is graded on its own (d/s/k, b = back)
       b = previous row   q = save and quit
The verdict is for the whole ROW (it applies to every version that has an event in it) unless n is pressed. grading_rows.csv (row, truth, note) is rewritten after EVERY key,
so quitting in the middle loses nothing; truth = dung | sai | khong_ro | split (split: note lists "version=verdict" for each version). When q is pressed and every row has a verdict the
precision table is computed and written into the EXP-020 README (section "Precision by eye").
The crops are rebuilt from outputs/review_* (gitignored) and test_video.mp4; nothing is written except grading_rows.csv and the README section.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
EXP = REPO / "experiments" / "EXP-020-regression-diag"
OUT_CSV = EXP / "grading_rows.csv"
README = EXP / "README.md"
ROWS_JSON = EXP / "event_rows_5_versions.json"
V0_LABELS = REPO / "outputs" / "regression_diag" / "v0" / "labels.json"
RUNS = {"current+old model": ("outputs/review_A", 1), "EXP-015 3cls": ("outputs/review_B", 1), "EXP-016 4cls": ("outputs/review_C_4cls", 1),
        "current+old model @10FPS": ("outputs/review_10fps", 3)}                      # folder, frame scale (the 10 FPS run processed every 3rd frame)
VERSIONS = ["v0"] + list(RUNS)
SKIP_TYPES = {"wrong_way"}
TRUTHS = {"d": "dung", "s": "sai", "k": "khong_ro"}
MARK = "<!-- grade_rows:start -->", "<!-- grade_rows:end -->"
CSV_FIELDS = ["row", "truth", "note"]
PALETTE = [(0, 255, 255), (0, 255, 0), (255, 200, 0), (255, 0, 255), (0, 165, 255)]


# ---------------------------------------------------------------- data (pure)
def load_rows(path=ROWS_JSON):
    """-> [{"n", "time_s", "type", "versions": {name: {"track", "frame"} or None}}] without the skipped types, in the order of the grading template (time, then row number)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["rows"]
    rows = [r for r in rows if r["type"] not in SKIP_TYPES]
    return sorted(rows, key=lambda r: (r["time_s"], r["n"]))


def read_grades(path=OUT_CSV):
    """-> {row number: {"truth", "note"}}"""
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8", newline="") as fh:
        return {int(r["row"]): {"truth": r["truth"], "note": r["note"]} for r in csv.DictReader(fh)}


def write_grades(grades, path=OUT_CSV):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(CSV_FIELDS)
        for n in sorted(grades):
            w.writerow([n, grades[n]["truth"], grades[n]["note"]])
    tmp.replace(path)                                                                   # atomic: a crash never leaves a half-written file


def split_note(note):
    """'different people; v0=sai|EXP-015 3cls=dung' -> {"v0": "sai", "EXP-015 3cls": "dung"}"""
    out = {}
    for part in note.split("|"):
        part = part.split(";")[-1].strip()
        if "=" in part:
            k, v = part.rsplit("=", 1)
            out[k.strip()] = v.strip()
    return out


def is_complete(row, grade):
    """A row is graded when it has a row verdict, or (split) a verdict for every version that has an event in it."""
    if grade is None:
        return False
    if grade["truth"] != "split":
        return True
    per = split_note(grade["note"])
    return all(v in per for v, e in row["versions"].items() if e)


def verdicts_of(row, grade):
    """{version: truth} for the versions that have an event in the row, from the row verdict or, for `split`, from the per-version note."""
    present = [v for v, e in row["versions"].items() if e]
    if grade["truth"] == "split":
        per = split_note(grade["note"])
        return {v: per.get(v, "khong_ro") for v in present}
    return {v: grade["truth"] for v in present}


def stats(rows, grades):
    """Precision per version = dung / (dung + sai), khong_ro left out, with the counts; relative coverage = among the rows where at least one version is `dung`, how many each version has
    as `dung` (NOT recall: a violation that all versions missed is not in the denominator)."""
    per = {v: {"dung": 0, "sai": 0, "khong_ro": 0} for v in VERSIONS}
    true_rows = []
    for r in rows:
        g = grades.get(r["n"])
        if g is None:
            continue
        vd = verdicts_of(r, g)
        for v, t in vd.items():
            per[v][t] += 1
        if "dung" in vd.values():
            true_rows.append((r["n"], vd))
    for v, c in per.items():
        n = c["dung"] + c["sai"]
        c["events"] = n + c["khong_ro"]
        c["precision"] = round(c["dung"] / n, 3) if n else None
        c["covered_true_rows"] = sum(1 for _, vd in true_rows if vd.get(v) == "dung")
    return {"versions": per, "true_rows": len(true_rows), "rows_graded": len([r for r in rows if r["n"] in grades]), "rows": len(rows)}


def render_markdown(st):
    v = st["versions"]
    L = ["", MARK[0], "## Precision by eye (`tools/grade_rows.py`)", "",
         f"Graded by the project owner by eye, {st['rows_graded']} of {st['rows']} rows (no_helmet / overload; wrong_way rows skipped: that rule is off). The verdict of a row applies to every version that has "
         "an event in it, unless the row was marked as different people (then each version was graded on its own). Precision = right / (right + wrong), unclear left out.", "",
         "| version | events graded | right | wrong | unclear | precision | relative coverage |", "|---|---|---|---|---|---|---|"]
    for name in VERSIONS:
        c = v[name]
        p = "n/a" if c["precision"] is None else f"{c['precision']:.2f}"
        L.append(f"| {name} | {c['events']} | {c['dung']} | {c['sai']} | {c['khong_ro']} | {p} ({c['dung']}/{c['dung'] + c['sai']}) | {c['covered_true_rows']}/{st['true_rows']} |")
    L += ["", f"Relative coverage = among the {st['true_rows']} rows where at least one version was graded right, how many each version caught. **It is NOT recall**: a violation that all five versions missed "
          "is in nobody's denominator, so the true number of violations in the clip is unknown.", "",
          "**Read with care:** n is small (every precision above rests on a handful of events), the whole evidence is ONE 26-second clip, and the rules (windows, votes, hysteresis, the ratios) were "
          "adjusted on this very clip, so the numbers say little about new footage. The versions differ in model AND settings (see the table above), so a difference between them is not "
          "attributed to one cause. The default model is not changed by this table.", MARK[1], ""]
    return "\n".join(L)


def write_readme(st, path=README):
    s = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n")
    block = render_markdown(st)
    if MARK[0] in s:
        a, b = s.index(MARK[0]), s.index(MARK[1]) + len(MARK[1])
        s = s[:a].rstrip("\n") + "\n" + block.strip("\n") + "\n" + s[b:].lstrip("\n")
    else:
        s = s.rstrip("\n") + "\n" + block
    Path(path).write_text(s, encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- images
def load_events():
    """{version: {(track, type): {"frame", "box"}}} rebuilt from the review runs and the v0 labels (same loaders as tools/contact_sheet_versions.py)."""
    sys.path.insert(0, str(REPO / "tools"))
    from contact_sheet_versions import load_run, load_v0
    ev = {"v0": {(e["track"], e["type"]): e for e in load_v0(V0_LABELS)}}
    for name, (folder, scale) in RUNS.items():
        ev[name] = {(e["track"], e["type"]): e for e in load_run(REPO / folder, scale)}
    return ev


class Frames:
    def __init__(self, video):
        import cv2
        self.cv2, self.cap, self.cache = cv2, cv2.VideoCapture(str(video)), {}

    def get(self, i):
        if i not in self.cache:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = self.cap.read()
            self.cache[i] = self.cv2.resize(f, (1280, 720)) if ok else None
            if len(self.cache) > 40:
                self.cache.pop(next(iter(self.cache)))
        return self.cache[i]


def row_events(row, events):
    return {v: events[v][(e["track"], row["type"])] for v, e in row["versions"].items() if e and (e["track"], row["type"]) in events[v]}


def compose(row, idx, total, evs, frames, grade=None, focus=None, per_version=None):
    """The canvas of one row. evs: {version: event}. focus: version being graded alone (n mode)."""
    import cv2
    import numpy as np
    H = 340
    crops = []
    for k, (v, e) in enumerate(evs.items()):
        f = frames.get(e["frame"])
        x1, y1, x2, y2 = e["box"]
        bw, bh = x2 - x1, y2 - y1
        X1, Y1, X2, Y2 = int(max(0, x1 - 0.25 * bw)), int(max(0, y1 - 0.25 * bh)), int(min(1280, x2 + 0.25 * bw)), int(min(720, y2 + 0.25 * bh))
        c = f[Y1:Y2, X1:X2].copy()
        cv2.rectangle(c, (x1 - X1, y1 - Y1), (x2 - X1, y2 - Y1), PALETTE[VERSIONS.index(v) % 5], 2)
        c = cv2.resize(c, (max(60, int(c.shape[1] * H / c.shape[0])), H))[:, :360]
        tile = np.full((H + 52, c.shape[1], 3), 30, np.uint8)
        tile[52:] = c
        shown = (per_version or {}).get(v)
        col = (255, 255, 255) if focus in (None, v) else (110, 110, 110)
        cv2.putText(tile, v[:34], (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PALETTE[VERSIONS.index(v) % 5], 1)
        cv2.putText(tile, f"{e['track']}  t={e['frame'] / 30:.2f}s" + (f"  -> {shown}" if shown else ""), (4, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        if focus == v:
            cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 255, 255), 3)
        crops.append(tile)
    strip = np.hstack(crops) if crops else np.full((H + 52, 400, 3), 30, np.uint8)
    W = max(1280, strip.shape[1] + 20)
    first = next(iter(evs.values()))
    full = frames.get(first["frame"]).copy()
    for v, e in evs.items():
        x1, y1, x2, y2 = e["box"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(full, (cx, cy), int(max(x2 - x1, y2 - y1) * 0.75) + 12, PALETTE[VERSIONS.index(v) % 5], 3)
    full = cv2.resize(full, (640, 360))
    canvas = np.full((60 + strip.shape[0] + 380, W, 3), 20, np.uint8)
    cv2.putText(canvas, f"row #{row['n']}   t={row['time_s']:.2f}s   {row['type']}   row {idx + 1}/{total}", (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (0, 255, 255), 2)
    canvas[60:60 + strip.shape[0], 10:10 + strip.shape[1]] = strip
    y0 = 60 + strip.shape[0] + 8
    canvas[y0:y0 + 360, 10:650] = full
    cv2.putText(canvas, f"whole frame {first['frame']} (circles = the vehicles of the versions above)", (660, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    help_lines = ["d = true violation   s = wrong (helmet / wrong person)   k = unclear", "n = different people in this row: grade each version alone",
                  "b = back one row   q = save and quit"]
    if focus:
        help_lines = [f"grading '{focus}' alone:  d / s / k      b = back"]
    for i, t in enumerate(help_lines):
        cv2.putText(canvas, t, (660, y0 + 60 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    if grade:
        cv2.putText(canvas, f"current: {grade['truth']} {grade['note']}"[:90], (660, y0 + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    return canvas


# ---------------------------------------------------------------- UI
def run_ui(rows, grades, events, frames):
    import cv2
    win = "grade_rows"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = next((k for k, r in enumerate(rows) if not is_complete(r, grades.get(r["n"]))), len(rows) - 1)        # resume at the first row without a verdict
    while 0 <= i < len(rows):
        row = rows[i]
        evs = row_events(row, events)
        cv2.imshow(win, compose(row, i, len(rows), evs, frames, grades.get(row["n"])))
        key = cv2.waitKey(0) & 0xFF
        ch = chr(key).lower() if key < 128 else ""
        if ch in TRUTHS:
            grades[row["n"]] = {"truth": TRUTHS[ch], "note": ""}
            write_grades(grades)
            i += 1
        elif ch == "n":
            per = {}
            names = list(evs)
            j = 0
            while 0 <= j < len(names):
                cv2.imshow(win, compose(row, i, len(rows), evs, frames, grades.get(row["n"]), focus=names[j], per_version=per))
                k2 = chr(cv2.waitKey(0) & 0xFF).lower()
                if k2 in TRUTHS:
                    per[names[j]] = TRUTHS[k2]
                    j += 1
                    grades[row["n"]] = {"truth": "split", "note": "different people; " + "|".join(f"{a}={b}" for a, b in per.items())}
                    write_grades(grades)                                                    # saved after every key, also half-way through a split row
                elif k2 == "b":
                    j = max(0, j - 1)
                    per.pop(names[j], None)
                elif k2 == "q":
                    break
            if len(per) == len(names):
                i += 1
            elif len(per) == 0:
                grades.pop(row["n"], None)
                write_grades(grades)
        elif ch == "b":
            i = max(0, i - 1)
        elif ch == "q":
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true", help="no window: compute the table from grading_rows.csv and write it into the README")
    a = ap.parse_args()
    rows = load_rows()
    grades = read_grades()
    if not a.report:
        events = load_events()
        for r in rows:
            for v, e in r["versions"].items():
                if e and (e["track"], r["type"]) not in events[v]:
                    print(f"[!] row {r['n']}: no event for {v} {e['track']} (outputs/ folders missing?)")
        run_ui(rows, grades, events, Frames(REPO / "test_video.mp4"))
        grades = read_grades()
    done = [r for r in rows if is_complete(r, grades.get(r["n"]))]
    print(f"{len(done)}/{len(rows)} rows graded -> {OUT_CSV}")
    if len(done) == len(rows):
        st = stats(rows, grades)
        write_readme(st)
        print(json.dumps(st["versions"], indent=1))
        print(f"table written into {README}")
    else:
        print("not all rows are graded yet: run again to continue; the README is written when all rows have a verdict")


if __name__ == "__main__":
    main()
