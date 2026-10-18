"""tools/rewrite_author.py on a throw-away repository (never on the real one)."""
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parent.parent / "tools" / "rewrite_author.py"


def run(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, text=True).stdout


def make(tmp_path):
    run(tmp_path, "init", "-q")
    run(tmp_path, "config", "user.name", "Old Name")
    run(tmp_path, "config", "user.email", "old@example.invalid")
    for i in range(2):
        (tmp_path / f"f{i}").write_text(str(i), encoding="utf-8")
        run(tmp_path, "add", "-A")
        run(tmp_path, "commit", "-q", "-m", f"c{i}")


def tool(tmp_path, *a):
    return subprocess.run([sys.executable, str(TOOL), "--repo", str(tmp_path), *a], capture_output=True, text=True)


def test_placeholder_email_is_refused(tmp_path):
    make(tmp_path)
    r = tool(tmp_path, "--email", "<USER>@users.noreply.github.com", "--yes")
    assert r.returncode != 0 and "placeholder" in (r.stderr + r.stdout)
    assert "old@example.invalid" in run(tmp_path, "log", "--format=%ae")


def test_dry_run_changes_nothing_and_yes_rewrites_everything_with_a_backup(tmp_path):
    make(tmp_path)
    r = tool(tmp_path, "--email", "me@users.noreply.github.com")
    assert r.returncode == 0 and "dry run" in r.stdout and "old@example.invalid" in run(tmp_path, "log", "--format=%ae")
    r = tool(tmp_path, "--email", "me@users.noreply.github.com", "--yes")
    assert r.returncode == 0, r.stderr
    assert set(run(tmp_path, "log", "--format=%ae%n%ce").split()) == {"me@users.noreply.github.com"}
    bundle = tmp_path.parent / f"{tmp_path.name}.before-author-rewrite.bundle"
    clone = tmp_path.parent / (tmp_path.name + "_clone")
    run(tmp_path.parent, "clone", "-q", str(bundle), str(clone))
    assert "old@example.invalid" in run(clone, "log", "--format=%ae")                                    # the old history is kept in the bundle
    assert run(tmp_path, "rev-list", "--count", "HEAD").strip() == "2"
