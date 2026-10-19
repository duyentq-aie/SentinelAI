"""tools/preflight_publish.py on a throw-away git repository (never on the real one)."""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import preflight_publish as pf  # noqa: E402


def make_repo(tmp_path, files):
    def run(*a):
        subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.name", "Test User")
    run("config", "user.email", "test@example.invalid")
    for name, content in files.items():
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content if isinstance(content, bytes) else content.encode("utf-8"))
    run("add", "-A")
    run("commit", "-q", "-m", "x")
    return tmp_path


def kinds(findings):
    return {(f["category"], f["kind"]) for f in findings}


def test_text_scan_finds_secrets_personal_paths_emails_and_identifiers():
    txt = "\n".join([
        "token = 'abcd1234efgh5678'",  # preflight-ignore
        "-----BEGIN RSA PRIVATE KEY-----",  # preflight-ignore
        "key ghp_" + "a" * 36,
        "path " + "C:" + "\\Users\\Alice\\work\\data and " + "Q:" + "\\SentinelAI\\x",      # built at run time: the literals must not appear in the repository
        "mail bob@corp.org and noreply@anthropic.com",
        "dataset tranquangduyenk17ct/helmet",
    ])
    k = kinds(pf.scan_text("f.txt", txt))
    assert ("SECRET", "hard-coded credential") in k and ("SECRET", "private key") in k and ("SECRET", "GitHub token") in k
    assert ("PERSONAL", "Windows home path") in k and ("PERSONAL", "absolute project path") in k and ("PERSONAL", "e-mail address") in k
    assert ("PERSONAL", "identifier 'tranquangduyen'") in k
    emails = [f["text"] for f in pf.scan_text("f.txt", txt) if f["kind"] == "e-mail address"]
    assert emails == ["bob@corp.org"]                                  # the harmless noreply address is not reported


def test_clean_text_has_no_findings():
    assert pf.scan_text("a.py", "def f(x):\n    return x * 2  # nothing to see\npassword_field = None\n") == []


def test_working_tree_scan_flags_weights_videos_big_files_and_secrets(tmp_path):
    repo = make_repo(tmp_path, {"README.md": "hello", "w/model.pt": b"\0" * 2048, "clip.mp4": b"\0" * 10, "big.dat": b"\0" * (2 * 2**20),
                                "src/a.py": "api_key = 'sk_live_abcdefghijklmnop'\n"})  # preflight-ignore
    files, findings = pf.scan_working_tree([], max_mb=1, root=repo)
    k = kinds(findings)
    assert ("BINARY", "model weights would be published") in k and ("BINARY", "video would be published (plates/faces?)") in k
    assert ("LARGE", "large file in working tree") in k and ("SECRET", "hard-coded credential") in k
    assert {"README.md", "w/model.pt", "clip.mp4"} <= set(files)


def test_gitignored_files_are_not_scanned(tmp_path):
    repo = make_repo(tmp_path, {".gitignore": "secret.txt\n", "a.txt": "fine"})
    (repo / "secret.txt").write_text("password = 'supersecret123'\n", encoding="utf-8")  # preflight-ignore
    files, findings = pf.scan_working_tree([], 5, root=repo)
    assert "secret.txt" not in files and not [f for f in findings if f["category"] == "SECRET"]


def test_history_scan_finds_a_secret_that_was_later_deleted_and_a_large_blob(tmp_path):
    repo = make_repo(tmp_path, {"cfg.txt": "token = 'abcdefgh12345678'\n", "big.bin": b"\0" * (2 * 2**20)})  # preflight-ignore
    subprocess.run(["git", "rm", "-q", "cfg.txt", "big.bin"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "remove"], cwd=repo, check=True)
    k = kinds(pf.scan_history([], 1, root=repo))
    assert any(c == "SECRET" for c, _ in k) and ("LARGE", "large blob in history") in k
    assert not (repo / "cfg.txt").exists()                               # the working tree is clean, only history has it


def test_license_check(tmp_path):
    assert pf.check_license(tmp_path)[0]["kind"] == "LICENSE file missing"
    (tmp_path / "LICENSE").write_text("MIT License ...", encoding="utf-8")
    assert pf.check_license(tmp_path)[0]["kind"].startswith("LICENSE is not")
    (tmp_path / "LICENSE").write_text("GNU AFFERO GENERAL PUBLIC LICENSE\nVersion 3, 19 November 2007\n", encoding="utf-8")
    assert pf.check_license(tmp_path) == []


def test_weight_metadata_scan_finds_identifiers_inside_local_weight_files(tmp_path):
    (tmp_path / "helmet_best.pt").write_bytes(b"xx/kaggle/input/datasets/TranQuangDuyenk17ct/data.yamlxx")
    (tmp_path / "other.pt").write_bytes(b"nothing here")
    out = pf.scan_weight_metadata(["tranquangduyen"], root=tmp_path)
    assert [f["path"] for f in out] == ["helmet_best.pt"] and "NOT tracked" in out[0]["kind"]


def test_authors_lists_distinct_commit_authors(tmp_path):
    repo = make_repo(tmp_path, {"a": "1"})
    assert pf.authors(repo) == ["Test User <test@example.invalid>"]
