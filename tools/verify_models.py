"""Check downloaded model files against the sha256 list of the release (docs/releases/models-v1.sha256).

  python tools/verify_models.py [folder]      # default: the current folder; also accepts the SHA256SUMS file that comes with the release

Exit code 0 when every file listed is present and matches, 1 otherwise. Missing files are reported, not ignored.
"""
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIST = REPO / "docs" / "releases" / "models-v1.sha256"


def parse(text):
    """'<sha256> *<name>' lines -> {name: sha256}"""
    out = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            out[parts[1].lstrip("*").strip()] = parts[0].lower()
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(folder, expected):
    """-> [(name, "ok" | "MISSING" | "MISMATCH")]"""
    res = []
    for name, want in sorted(expected.items()):
        p = Path(folder) / name
        res.append((name, "MISSING" if not p.exists() else ("ok" if sha256(p) == want else "MISMATCH")))
    return res


def main(argv):
    folder = Path(argv[1]) if len(argv) > 1 else Path(".")
    expected = parse((folder / "SHA256SUMS").read_text(encoding="utf-8") if (folder / "SHA256SUMS").exists() else LIST.read_text(encoding="utf-8"))
    res = verify(folder, expected)
    for name, status in res:
        print(f"{status:9s} {name}")
    return 0 if all(s == "ok" for _, s in res) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
