"""scripts/hooks/pre-commit refuses forbidden files and text in a throw-away repository (the real repository is never touched)."""
import shutil
import subprocess

import pytest

from sentinel.config import REPO

HOOK = REPO / "scripts" / "hooks" / "pre-commit"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")

# the forbidden strings are assembled here so that this test file does not trip the hook itself
MAIL = "someone@gmail." + "com"
HOME = "C:" + "\\Users\\" + "x"
PROJECT = "Q:" + "\\SentinelAI"


def git(cwd, *args, check=True):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, capture_output=True, text=True, check=check)


@pytest.fixture()
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.org")
    git(tmp_path, "config", "user.name", "t")
    hook = tmp_path / ".git" / "hooks" / "pre-commit"
    hook.write_bytes(HOOK.read_bytes().replace(b"\r\n", b"\n"))
    hook.chmod(0o755)
    return tmp_path


def try_commit(repo, name, data):
    f = repo / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    git(repo, "add", "-f", name)
    r = git(repo, "commit", "-q", "-m", "t", check=False)
    return r.returncode, r.stdout + r.stderr


def test_a_clean_text_file_is_committed(repo):
    code, out = try_commit(repo, "notes.md", "nothing special\n")
    assert code == 0, out


@pytest.mark.parametrize("name", ["model.pt", "m.onnx", "m.engine", "clip.mp4", "data.zip", "backup.bundle", "UPPER.PT"])
def test_model_video_and_archive_files_are_refused(repo, name):
    code, out = try_commit(repo, name, b"\x00\x01binary")
    assert code != 0 and "blocked file type" in out and name in out


@pytest.mark.parametrize("name", ["frame.jpg", "a/b/crop.png", "x.jpeg", "experiments/EXP-1/frames/f.jpg", "static/uploads/p.png", "demo.gif", "docs/other/demo.gif", "static/x.gif"])
def test_images_outside_the_chart_folders_are_refused(repo, name):
    code, out = try_commit(repo, name, b"\x89PNG\x00")
    assert code != 0 and "image outside the chart folders" in out


@pytest.mark.parametrize("name", ["docs/assets/demo.gif", "docs/assets/review.png", "docs/charts/a.png", "docs/figures/a.jpg", "experiments/EXP-1/charts/a.png", "experiments/EXP-015-x/eval_3cls_val/confusion_matrix.png",
                                  "experiments/EXP-015-x/box_consistency/hist.png", "experiments/EXP-015-x/helmet_3cls_splitA/results.png"])
def test_images_in_the_allow_listed_chart_folders_are_accepted(repo, name):
    code, out = try_commit(repo, name, b"\x89PNG\x00")
    assert code == 0, out


@pytest.mark.parametrize("text", [f"contact {MAIL}\n", f"path {HOME}\n", f"path {PROJECT}\n", f"CONTACT {MAIL.upper()}\n"])
def test_forbidden_text_is_refused_and_never_echoed(repo, text):
    code, out = try_commit(repo, "doc.md", "ok line\n" + text)
    assert code != 0 and "forbidden text in doc.md (line 2" in out
    assert "gmail" not in out.lower() and "users" not in out.lower().replace("pre-commit", "")          # only file and line number are printed


def test_a_refused_commit_leaves_nothing_committed_and_a_later_clean_commit_works(repo):
    assert try_commit(repo, "bad.mp4", b"x")[0] != 0
    assert git(repo, "log", check=False).returncode != 0                               # no commit exists
    git(repo, "rm", "-q", "--cached", "bad.mp4")
    code, out = try_commit(repo, "ok.txt", "fine\n")
    assert code == 0, out


def test_the_hook_does_not_match_its_own_patterns():
    text = HOOK.read_text(encoding="utf-8")
    for needle in (MAIL.split("@")[1], HOME[:-1], PROJECT):
        assert needle not in text
