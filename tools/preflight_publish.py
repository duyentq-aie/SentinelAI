"""Pre-publication scan of the repository (what a `git push` would publish). Reads only; changes and deletes nothing.

  python tools/preflight_publish.py [--max-mb 5] [--json outputs/preflight.json] [--history]

Checks: secrets / tokens (working tree and, with --history, every revision), large files (working tree and history), personal paths and identifiers
(Windows home paths, absolute project paths, e-mail addresses, the Kaggle user name), weights or videos accidentally tracked, personal identifiers embedded
in local weight files (never committed, but they would leak if someone uploaded them), the LICENSE file, dataset-licence status, commit authors.
Findings are information for a human decision, not verdicts: false positives are possible, absence of findings is not a guarantee.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b|\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    "AWS access key id": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "Slack token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    "OpenAI/Anthropic-style key": re.compile(r"\bsk-(?:ant-)?[A-Za-z0-9_-]{20,}\b"),
    "Google API key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "Kaggle credentials": re.compile(r'"username"\s*:\s*"[^"]+"\s*,\s*"key"\s*:\s*"[0-9a-f]{20,}"'),
    "hard-coded credential": re.compile(r"""(?i)\b(?:api[_-]?key|secret|token|passwd|password)\b\s*[:=]\s*['"][^'"\s]{8,}['"]"""),
}
PERSONAL_PATTERNS = {
    "Windows home path": re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+(?!Public\b|Default\b|All Users\b)[^\\/\s\"'<>|:*?]+", re.I),
    "absolute project path": re.compile(r"\b[A-Za-z]:[\\/]+SentinelAI\b", re.I),
    "e-mail address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}
DEFAULT_NAMES = ["tranquangduyen"]            # Kaggle user name stem and similar identifiers; extend with --names
BINARY_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".pt", ".onnx", ".engine", ".bin", ".zip", ".pyc"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".safetensors", ".ckpt", ".h5"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
HARMLESS_EMAILS = ("noreply@anthropic.com", "example.com", "users.noreply.github.com")


def git(*args, cwd=REPO):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True, encoding="utf-8", errors="replace")


def scan_text(path, text, names=DEFAULT_NAMES):
    """-> list of findings {kind, category, path, line, text} for one text blob."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if "preflight-ignore" in line:                 # deliberate fake values (the scanner's own tests)
            continue
        for kind, rx in SECRET_PATTERNS.items():
            if rx.search(line):
                out.append({"category": "SECRET", "kind": kind, "path": path, "line": n, "text": line.strip()[:160]})
        for kind, rx in PERSONAL_PATTERNS.items():
            for m in rx.finditer(line):
                if kind == "e-mail address" and any(h in m.group(0) for h in HARMLESS_EMAILS):
                    continue
                out.append({"category": "PERSONAL", "kind": kind, "path": path, "line": n, "text": m.group(0)})
        for name in names:
            if name.lower() in line.lower():
                out.append({"category": "PERSONAL", "kind": f"identifier '{name}'", "path": path, "line": n, "text": line.strip()[:160]})
    return out


def scan_working_tree(names, max_mb, root=REPO):
    tracked = git("ls-files", cwd=root).split("\n")
    untracked = git("ls-files", "--others", "--exclude-standard", cwd=root).split("\n")
    files = [f for f in tracked + untracked if f]
    findings, big = [], []
    for f in files:
        p = Path(root) / f
        if not p.is_file():
            continue
        size = p.stat().st_size
        if size > max_mb * 2**20:
            big.append({"category": "LARGE", "kind": "large file in working tree", "path": f, "text": f"{size / 2**20:.1f} MiB", "tracked": f in tracked})
        if p.suffix.lower() in WEIGHT_SUFFIXES:
            findings.append({"category": "BINARY", "kind": "model weights would be published", "path": f, "text": f"{size / 2**20:.1f} MiB", "tracked": f in tracked})
        if p.suffix.lower() in VIDEO_SUFFIXES:
            findings.append({"category": "BINARY", "kind": "video would be published (plates/faces?)", "path": f, "text": f"{size / 2**20:.1f} MiB", "tracked": f in tracked})
        if p.suffix.lower() in BINARY_SUFFIXES or size > 5 * 2**20:
            continue
        try:
            findings += scan_text(f, p.read_text(encoding="utf-8"), names)
        except (UnicodeDecodeError, OSError):
            pass
    return files, findings + big


def scan_history(names, max_mb, root=REPO):
    """Large blobs and secret patterns in every revision (git grep per commit)."""
    findings = []
    objs = git("rev-list", "--objects", "--all", cwd=root).split("\n")
    path_of = {line.split(" ", 1)[0]: (line.split(" ", 1)[1] if " " in line else "") for line in objs if line}
    proc = subprocess.run(["git", "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"], input="\n".join(path_of), cwd=root, text=True, capture_output=True)
    for line in proc.stdout.splitlines():
        sha, typ, size = line.split()
        if typ == "blob" and int(size) > max_mb * 2**20:
            findings.append({"category": "LARGE", "kind": "large blob in history", "path": path_of.get(sha, "?"), "text": f"{int(size) / 2**20:.1f} MiB ({sha[:10]})"})
    # every distinct text blob once (a secret deleted later still lives in an old revision); patterns run in Python (git grep -E has no \b or (?i))
    small = [line.split()[0] for line in proc.stdout.splitlines() if line.split()[1] == "blob" and int(line.split()[2]) <= min(max_mb, 1) * 2**20]
    for sha in small:
        blob = subprocess.run(["git", "cat-file", "blob", sha], cwd=root, capture_output=True).stdout
        if b"\0" in blob[:4096]:
            continue
        for f in scan_text(path_of.get(sha, "?"), blob.decode("utf-8", errors="replace"), names):
            if f["category"] == "SECRET":
                findings.append({**f, "kind": f"{f['kind']} (history)", "text": f"{f['text'][:100]} [{sha[:10]}]"})
    return findings


def scan_weight_metadata(names, root=REPO):
    """Local (ignored) weight files embed training metadata: look for personal identifiers inside their bytes."""
    out = []
    cands = sorted(Path(root).glob("*.pt"))
    if (Path(root) / "models").exists():
        cands += sorted((Path(root) / "models").rglob("*.onnx"))
    for p in cands:
        try:
            blob = p.read_bytes()
        except OSError:
            continue
        for name in names:
            if name.encode().lower() in blob.lower():
                out.append({"category": "PERSONAL", "kind": f"identifier '{name}' inside a weight file (NOT tracked: do not publish it as is)", "path": str(p.relative_to(root)), "text": f"{blob.lower().count(name.encode().lower())} occurrence(s)"})
    return out


def check_license(root=REPO):
    f = Path(root) / "LICENSE"
    if not f.exists():
        return [{"category": "LICENSE", "kind": "LICENSE file missing", "path": "LICENSE", "text": "add the AGPL-3.0 text (the project depends on Ultralytics, AGPL-3.0)"}]
    t = f.read_text(encoding="utf-8", errors="replace")
    if "GNU AFFERO GENERAL PUBLIC LICENSE" not in t or "Version 3" not in t:
        return [{"category": "LICENSE", "kind": "LICENSE is not the AGPL-3.0 text", "path": "LICENSE", "text": t[:80]}]
    return []


def authors(root=REPO):
    return sorted(set(git("log", "--all", "--format=%an <%ae>", cwd=root).split("\n")) - {""})


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")      # Windows consoles (cp1258...) cannot print every character found in files
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-mb", type=float, default=5.0)
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES, help="personal identifiers to look for")
    ap.add_argument("--history", action="store_true", help="also scan every revision (slower)")
    ap.add_argument("--json")
    a = ap.parse_args()
    files, findings = scan_working_tree(a.names, a.max_mb)
    findings += scan_weight_metadata(a.names) + check_license()
    if a.history:
        findings += scan_history(a.names, a.max_mb)
    auth = authors()
    by = {}
    for f in findings:
        by.setdefault((f["category"], f["kind"]), []).append(f)
    print(f"scanned {len(files)} files (tracked + untracked, not ignored){' + history' if a.history else ''}; {len(findings)} findings\n")
    for (cat, kind), items in sorted(by.items()):
        print(f"[{cat}] {kind}: {len(items)}")
        for it in items[:6]:
            print(f"    {it['path']}{':' + str(it['line']) if 'line' in it else ''}  {it['text'][:110]}")
        if len(items) > 6:
            print(f"    ... {len(items) - 6} more")
    print("\ncommit authors (published with the history):", *auth, sep="\n    ")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"findings": findings, "authors": auth, "n_files": len(files)}, indent=2), encoding="utf-8")
    secrets = sum(1 for f in findings if f["category"] == "SECRET")
    print(f"\nSECRET findings: {secrets}  (must be 0 before any push)")
    sys.exit(1 if secrets else 0)


if __name__ == "__main__":
    main()
