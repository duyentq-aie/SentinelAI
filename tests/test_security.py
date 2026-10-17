import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from sentinel.security import (SourceRejected, UploadRejected, UploadTooLarge, sanitize_filename, save_upload, validate_source)

EXT = (".mp4", ".avi", ".mov", ".mkv")


def web(**kw):
    base = dict(allowed_source_dirs=["static/uploads", "videos"], allowed_extensions=list(EXT), allow_rtsp=False, allow_camera_index=False)
    base.update(kw)
    return SimpleNamespace(**base)


# ---- file names
@pytest.mark.parametrize("evil", ["../../etc/passwd.mp4", "..\\..\\windows\\system32\\x.mp4", "/abs/path/video.mp4", "C:\\x\\y\\video.mp4",
                                  "a/b/../../c.mp4", "vid eo;rm -rf.mp4", "vidéo $(x).mp4"])
def test_sanitize_strips_directories_and_unsafe_characters(evil):
    out = sanitize_filename(evil, EXT, new_id=lambda: "abcd1234")
    assert out.startswith("abcd1234_") and out.endswith(".mp4")
    assert not any(c in out for c in "/\\ ;$()") and ".." not in out


def test_sanitize_adds_a_unique_prefix_so_uploads_never_overwrite():
    assert sanitize_filename("a.mp4", EXT) != sanitize_filename("a.mp4", EXT)


@pytest.mark.parametrize("bad", ["", "noext", "script.py", "movie.mp4.exe", "x.MP4.php", "run.bat", "\x00.mp4"])
def test_sanitize_rejects_other_extensions_and_empty_names(bad):
    with pytest.raises(UploadRejected):
        sanitize_filename(bad, EXT)


def test_sanitize_accepts_uppercase_extension():
    assert sanitize_filename("CLIP.MP4", EXT, new_id=lambda: "x").endswith("_CLIP.mp4")


# ---- upload size
def test_save_upload_writes_within_limit(tmp_path):
    p = save_upload(io.BytesIO(b"x" * 1000), "../evil.mp4", tmp_path, max_bytes=2000, allowed_ext=EXT)
    assert p.parent == tmp_path and p.read_bytes() == b"x" * 1000 and not list(tmp_path.glob("*.part"))


def test_save_upload_refuses_oversize_and_leaves_nothing(tmp_path):
    with pytest.raises(UploadTooLarge):
        save_upload(io.BytesIO(b"x" * 5000), "big.mp4", tmp_path, max_bytes=2000, allowed_ext=EXT, chunk=512)
    assert list(tmp_path.iterdir()) == []


def test_save_upload_refuses_empty_and_wrong_type(tmp_path):
    with pytest.raises(UploadRejected):
        save_upload(io.BytesIO(b""), "e.mp4", tmp_path, 100, EXT)
    with pytest.raises(UploadRejected):
        save_upload(io.BytesIO(b"abc"), "e.txt", tmp_path, 100, EXT)
    assert list(tmp_path.iterdir()) == []


# ---- source whitelist
def make_tree(tmp_path):
    (tmp_path / "static" / "uploads").mkdir(parents=True)
    (tmp_path / "videos").mkdir()
    (tmp_path / "secret").mkdir()
    for f in ("static/uploads/a.mp4", "videos/b.avi", "secret/c.mp4", "videos/notes.txt"):
        (tmp_path / f).write_bytes(b"x")
    return tmp_path


def test_allowed_files_are_resolved_to_absolute_paths(tmp_path):
    root = make_tree(tmp_path)
    assert validate_source("static/uploads/a.mp4", web(), root) == str((root / "static/uploads/a.mp4").resolve())
    assert validate_source(str(root / "videos" / "b.avi"), web(), root).endswith("b.avi")


@pytest.mark.parametrize("path", ["secret/c.mp4", "../outside.mp4", "static/uploads/../../secret/c.mp4", "videos/notes.txt", "videos/missing.mp4",
                                  "videos/b.avi\x00.txt", "C:/Windows/system32/x.mp4", "//server/share/a.mp4"])
def test_files_outside_whitelist_or_wrong_type_are_rejected(tmp_path, path):
    with pytest.raises(SourceRejected):
        validate_source(path, web(), make_tree(tmp_path))


def test_urls_other_than_rtsp_are_always_rejected(tmp_path):
    for url in ("http://example.com/a.mp4", "file:///etc/passwd", "ftp://x/a.mp4", "rtmp://x/live", "rtsp://cam/1"):
        with pytest.raises(SourceRejected):
            validate_source(url, web(), tmp_path)                                # rtsp rejected too while allow_rtsp is false
    with pytest.raises(SourceRejected):
        validate_source("http://example.com/a.mp4", web(allow_rtsp=True), tmp_path)


def test_rtsp_only_when_enabled(tmp_path):
    assert validate_source("rtsp://10.0.0.5/stream1", web(allow_rtsp=True), tmp_path) == "rtsp://10.0.0.5/stream1"
    with pytest.raises(SourceRejected):
        validate_source("rtsp://10.0.0.5/a b", web(allow_rtsp=True), tmp_path)
    with pytest.raises(SourceRejected):
        validate_source("rtsp://" + "a" * 600, web(allow_rtsp=True), tmp_path)


def test_camera_index_only_when_enabled(tmp_path):
    with pytest.raises(SourceRejected):
        validate_source("0", web(), tmp_path)
    assert validate_source("0", web(allow_camera_index=True), tmp_path) == 0


def test_symlink_escape_is_rejected(tmp_path):
    root = make_tree(tmp_path)
    link = root / "videos" / "link.mp4"
    try:
        link.symlink_to(root / "secret" / "c.mp4")
    except (OSError, NotImplementedError):
        pytest.skip("cannot create symlinks here")
    with pytest.raises(SourceRejected):
        validate_source("videos/link.mp4", web(), root)
