"""scripts/hooks/commit-msg removes AI-attribution lines from a commit message and leaves everything else alone (throw-away repository)."""
import shutil
import subprocess

import pytest

from sentinel.config import REPO

HOOK = REPO / "scripts" / "hooks" / "commit-msg"
pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")
# the forbidden lines are built here so that this file's own text is not an attribution line
CO = "Co-Authored-By: " + "Claude Sonnet 5.5 <noreply@anthropic.com>"
SESSION = "Claude-Session: " + "https://claude.ai/code/session_abc123"
GEN = "\U0001F916 Generated with [Claude" + " Code](https://claude.com/claude-code)"
LINK = "https://claude.ai/" + "code/session_xyz"


def git(cwd, *args, check=True, input=None):
    return subprocess.run(["git", "-c", "core.autocrlf=false", *args], cwd=cwd, capture_output=True, text=True, check=check, input=input, encoding="utf-8")


@pytest.fixture()
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "t@example.org")
    git(tmp_path, "config", "user.name", "t")
    hook = tmp_path / ".git" / "hooks" / "commit-msg"
    hook.write_bytes(HOOK.read_bytes().replace(b"\r\n", b"\n"))
    hook.chmod(0o755)
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    git(tmp_path, "add", "f.txt")
    return tmp_path


def commit(repo, message):
    r = git(repo, "commit", "-q", "-F", "-", check=False, input=message)
    return r.returncode, (git(repo, "log", "-1", "--format=%B", check=False).stdout if r.returncode == 0 else r.stderr + r.stdout)


def test_every_attribution_line_is_removed_and_the_message_ends_cleanly(repo):
    code, msg = commit(repo, f"Fix the thing\n\nBody line one.\nBody line two.\n\n{CO}\n{SESSION}\n{GEN}\n{LINK}\n")
    assert code == 0
    assert msg.rstrip("\n") == "Fix the thing\n\nBody line one.\nBody line two."
    for bad in ("claude", "claude.ai", "anthropic", "generated with"):
        assert bad not in msg.lower(), bad


def test_a_message_without_attribution_is_left_as_it_is(repo):
    original = "Subject\n\nParagraph one.\n\nParagraph two with `code` and a list:\n- a\n- b\n"
    code, msg = commit(repo, original)
    assert code == 0 and msg.rstrip("\n") == original.rstrip("\n")


def test_other_lines_that_mention_claude_in_a_different_way_are_kept(repo):
    code, msg = commit(repo, "Document the Claude Code hook\n\nThe hook removes the Co-Authored-By trailer of AI tools.\nSigned-off-by: Someone <s@example.org>\n")
    assert code == 0 and "Document the Claude Code hook" in msg and "Signed-off-by: Someone" in msg and "removes the Co-Authored-By trailer" in msg


def test_other_co_authors_are_not_touched(repo):
    code, msg = commit(repo, f"Pair work\n\nCo-Authored-By: Alice Example <alice@example.org>\n{CO}\n")
    assert code == 0 and "Co-Authored-By: Alice Example <alice@example.org>" in msg and "claude" not in msg.lower()


def test_blank_lines_left_at_the_end_go(repo):
    code, msg = commit(repo, f"S\n\nA\n\n\nB\n\n\n{GEN}\n\n{CO}\n\n\n")
    assert code == 0 and msg.rstrip("\n") == "S\n\nA\n\nB"      # git itself collapses the double blank line; the hook removes the attribution and the blank lines left at the end


def test_amend_and_m_flags_go_through_the_hook_too(repo):
    git(repo, "commit", "-q", "-m", "First", "-m", CO)
    assert "claude" not in git(repo, "log", "-1", "--format=%B").stdout.lower()
    git(repo, "commit", "-q", "--amend", "-m", "Second", "-m", f"Body\n\n{SESSION}")
    out = git(repo, "log", "-1", "--format=%B").stdout
    assert out.rstrip("\n") == "Second\n\nBody" and "claude" not in out.lower()


def test_a_message_that_is_only_attribution_still_fails_as_an_empty_message(repo):
    code, out = commit(repo, f"{CO}\n")
    assert code != 0                       # nothing but the trailer: the hook leaves an empty message and git refuses it


def test_the_hook_script_has_unix_line_endings():
    assert b"\r\n" not in HOOK.read_bytes()
