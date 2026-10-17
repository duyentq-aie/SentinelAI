"""Input validation for the demo web server (upload names/sizes, video-source whitelist). Pure functions, no web framework needed."""
import os
import re
import uuid
from pathlib import Path


class SourceRejected(ValueError):
    """The requested video source is not allowed."""


class UploadRejected(ValueError):
    """The upload is not acceptable (bad extension, empty name)."""


class UploadTooLarge(UploadRejected):
    pass


def sanitize_filename(name, allowed_ext, new_id=lambda: uuid.uuid4().hex[:8]):
    """Directory components removed, unsafe characters replaced, extension checked, unique prefix added (no overwrite, no traversal)."""
    if not name or "\x00" in name:
        raise UploadRejected("empty or invalid file name")
    base = os.path.basename(name.replace("\\", "/"))
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in {e.lower() for e in allowed_ext}:
        raise UploadRejected(f"file type {ext or '(none)'} is not allowed; allowed: {', '.join(sorted(allowed_ext))}")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")[:64] or "video"
    return f"{new_id()}_{stem}{ext}"


def save_upload(stream, filename, upload_dir, max_bytes, allowed_ext, chunk=1 << 20):
    """Copy `stream` (file-like) into upload_dir under a sanitised unique name, refusing anything bigger than max_bytes.
    The partial file is removed on any failure. -> Path of the saved file."""
    safe = sanitize_filename(filename, allowed_ext)
    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    final, part = upload_dir / safe, upload_dir / (safe + ".part")
    written = 0
    try:
        with open(part, "wb") as f:
            while True:
                buf = stream.read(chunk)
                if not buf:
                    break
                written += len(buf)
                if written > max_bytes:
                    raise UploadTooLarge(f"upload exceeds the limit of {max_bytes // (1024 * 1024)} MB")
                f.write(buf)
        if written == 0:
            raise UploadRejected("empty upload")
        os.replace(part, final)
        return final
    finally:
        if part.exists():
            part.unlink()


def _inside(path, folder):
    try:
        return os.path.commonpath([str(path), str(folder)]) == str(folder)
    except ValueError:                      # different drives
        return False


def validate_source(value, web, root):
    """Return the source the worker may open (absolute file path, int camera index or rtsp URL) or raise SourceRejected.

    web: object with allowed_source_dirs, allowed_extensions, allow_rtsp, allow_camera_index.
    """
    v = str(value).strip()
    if not v or any(ord(c) < 32 for c in v):
        raise SourceRejected("empty source or control characters")
    if v.isdigit():
        if web.allow_camera_index:
            return int(v)
        raise SourceRejected("camera indexes are disabled (web.allow_camera_index)")
    if "://" in v:
        if v.lower().startswith("rtsp://") and web.allow_rtsp and len(v) <= 512 and " " not in v:
            return v
        raise SourceRejected("only rtsp:// streams are supported, and only when web.allow_rtsp is true")
    root = Path(root).resolve()
    p = Path(v)
    p = (p if p.is_absolute() else root / p).resolve(strict=False)
    if p.suffix.lower() not in {e.lower() for e in web.allowed_extensions}:
        raise SourceRejected(f"file type {p.suffix or '(none)'} is not allowed")
    folders = [(root / d).resolve() if not Path(d).is_absolute() else Path(d).resolve() for d in web.allowed_source_dirs]
    if not any(_inside(p, f) for f in folders):
        raise SourceRejected("file is outside the allowed source folders")
    if not p.is_file():
        raise SourceRejected("file not found")
    return str(p)
