"""Rewrite the frame-based settings of a pipeline config as seconds / fractions (see sentinel/timebase.py).

  python tools/convert_config_to_seconds.py configs/pipeline/old.yaml            # prints the converted text, changes nothing
  python tools/convert_config_to_seconds.py configs/pipeline/old.yaml --write    # rewrites the file in place (comments and the other lines are kept)

Old frame values mean frames at 30 FPS, so at 30 FPS the converted config behaves exactly like the old one:
  window: 15 -> window_s: 0.5            min_votes: 5 -> min_vote_fraction: 0.333333 (= K / W)      violation_frames: 5 -> violation_s: 0.166667
  history_len / min_history: 30 / 15 -> history_s / min_history_s: 1 / 0.5      close_after: 30 -> close_after_s: 1
  plate.max_attempts: 90 (every 1) -> search_s: 3      plate.every: k -> every_s: k / 30 (1 -> 0)      ocr.min_gap: 3 -> min_gap_s: 0.1
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from sentinel.timebase import REFERENCE_FPS  # noqa: E402

F = REFERENCE_FPS


def num(x):
    x = round(float(x), 6)
    return str(int(x)) if x == int(x) else str(x)


def plan(data):
    """-> {dotted legacy key: (new key, new value)} for the keys present in `data`."""
    out = {}
    node = lambda path: _dig(data, path)
    for scope in ("rules.helmet", "rules.overload"):
        n = node(scope)
        if n is None:
            continue
        window = n.get("window")
        if "window" in n:
            out[f"{scope}.window"] = ("window_s", num(n["window"] / F))
        if "min_votes" in n:
            w = window if window is not None else round(0.5 * F)
            out[f"{scope}.min_votes"] = ("min_vote_fraction", num(n["min_votes"] / w))
    n = node("rules.helmet")
    if n and "violation_frames" in n:
        out["rules.helmet.violation_frames"] = ("violation_s", num(n["violation_frames"] / F))
    n = node("rules.wrong_way")
    if n:
        if "history_len" in n:
            out["rules.wrong_way.history_len"] = ("history_s", num(n["history_len"] / F))
        if "min_history" in n:
            out["rules.wrong_way.min_history"] = ("min_history_s", num(n["min_history"] / F))
    n = node("evidence")
    if n and "close_after" in n:
        out["evidence.close_after"] = ("close_after_s", num(n["close_after"] / F))
    n = node("evidence.plate")
    if n:
        every = n.get("every", 1)
        if "max_attempts" in n:
            out["evidence.plate.max_attempts"] = ("search_s", num(n["max_attempts"] * every / F))
        if "every" in n:
            out["evidence.plate.every"] = ("every_s", "0" if every == 1 else num(every / F))
    n = node("evidence.ocr")
    if n and "min_gap" in n:
        out["evidence.ocr.min_gap"] = ("min_gap_s", num(n["min_gap"] / F))
    return out


def _dig(data, path):
    for part in path.split("."):
        data = data.get(part) if isinstance(data, dict) else None
    return data if isinstance(data, dict) else None


def convert_text(text):
    """-> (new text, {old key: (new key, value)})."""
    data = yaml.safe_load(text) or {}
    changes = plan(data)
    stack, lines = [], text.split("\n")
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_][\w]*):(\s*)(.*)$", line)
        if not m or line.lstrip().startswith("#"):
            continue
        indent, key, _, rest = len(m.group(1)), m.group(2), m.group(3), m.group(4)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        dotted = ".".join([k for _, k in stack] + [key])
        if dotted in changes:
            new_key, value = changes[dotted]
            comment = re.search(r"\s+#.*$", rest)
            lines[i] = f"{m.group(1)}{new_key}: {value}" + (comment.group(0) if comment else "")
        if rest == "" or rest.startswith("#"):
            stack.append((indent, key))
    out = "\n".join(lines)
    wanted = {}
    for new_key, _ in changes.values():
        wanted[new_key] = wanted.get(new_key, 0) + 1
    if any(len(re.findall(rf"^\s*{re.escape(k)}:", out, re.M)) < n for k, n in wanted.items()):               # e.g. flow-style mappings `helmet: {window: 15}`: no line to edit -> rebuild from the parsed data
        out = _rebuild(data, changes)
    return out, changes


def _rebuild(data, changes):
    """Fallback for flow-style input: dump the converted mapping (comments and key order of the original are lost)."""
    for dotted, (new_key, value) in changes.items():
        scope, old = dotted.rsplit(".", 1)
        node = _dig(data, scope)
        node.pop(old, None)
        node[new_key] = float(value) if "." in value else int(value)
    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    p = Path(a.config)
    raw = p.read_bytes().decode("utf-8")
    nl = "\r\n" if "\r\n" in raw else "\n"
    text, changes = convert_text(raw.replace("\r\n", "\n"))
    for old, (new, v) in changes.items():
        print(f"{old} -> {new}: {v}", file=sys.stderr)
    if a.write:
        p.write_bytes(text.replace("\n", nl).encode("utf-8"))
        print(f"rewrote {p} ({len(changes)} setting(s))", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
