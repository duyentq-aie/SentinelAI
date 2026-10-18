"""Rewrite author/committer e-mail (and optionally name) in the WHOLE history. Rewrites commit hashes: run only before anything is pushed.

  python tools/rewrite_author.py --email <USER>@users.noreply.github.com [--name NAME] [--repo .] [--yes]

Without --yes it only prints what would change. With --yes it first writes a git bundle of the whole old history next to the repo
(`<repo>.before-author-rewrite.bundle`; nothing is deleted), then runs `git filter-branch`
over all branches and tags. Run from a clean working tree; the old e-mail addresses found in history are all replaced.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def git(*args, cwd, env=None):
    return subprocess.run(["git", *args], cwd=cwd, text=True, encoding="utf-8", capture_output=True, env=env)


def old_identities(cwd):
    out = git("log", "--all", "--format=%an <%ae>%n%cn <%ce>", cwd=cwd).stdout
    return sorted({line for line in out.splitlines() if line})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--email", required=True)
    ap.add_argument("--name")
    ap.add_argument("--repo", default=".")
    ap.add_argument("--yes", action="store_true")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    if "<" in a.email or ">" in a.email or "@" not in a.email:
        sys.exit(f"--email looks like a placeholder or is invalid: {a.email!r} (fill in the real GitHub user name)")
    print("identities in history:", *old_identities(repo), sep="\n  ")
    print(f"-> would become: {a.name or '(name unchanged)'} <{a.email}>")
    if git("status", "--porcelain", "--untracked-files=no", cwd=repo).stdout.strip():
        sys.exit("working tree has uncommitted changes: commit or stash first")
    if not a.yes:
        print("dry run (add --yes to rewrite)")
        return
    bundle = repo.parent / f"{repo.name}.before-author-rewrite.bundle"          # refs inside the repo would be rewritten too: back up to a file
    b = git("bundle", "create", str(bundle), "--all", cwd=repo)
    if b.returncode:
        sys.exit(f"backup failed, nothing rewritten: {b.stderr}")
    print(f"backup of the old history: {bundle}")
    env = {**os.environ, "FILTER_BRANCH_SQUELCH_WARNING": "1", "NEW_EMAIL": a.email, "NEW_NAME": a.name or ""}
    script = ('export GIT_AUTHOR_EMAIL="$NEW_EMAIL" GIT_COMMITTER_EMAIL="$NEW_EMAIL"; '
              'if [ -n "$NEW_NAME" ]; then export GIT_AUTHOR_NAME="$NEW_NAME" GIT_COMMITTER_NAME="$NEW_NAME"; fi')
    r = git("filter-branch", "-f", "--env-filter", script, "--", "--all", cwd=repo, env=env)
    print(r.stdout[-500:], r.stderr[-500:])
    print("after:", *old_identities(repo), sep="\n  ")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
