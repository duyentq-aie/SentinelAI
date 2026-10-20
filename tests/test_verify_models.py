"""tools/verify_models.py and the release list."""
import hashlib
import sys

from sentinel.config import REPO

sys.path.insert(0, str(REPO / "tools"))
import verify_models as vm  # noqa: E402


def test_parse_and_verify_ok_missing_mismatch(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"aaa")
    (tmp_path / "b.pt").write_bytes(b"bbb")
    exp = vm.parse(f"{hashlib.sha256(b'aaa').hexdigest()} *a.pt\n{hashlib.sha256(b'XXX').hexdigest()}  b.pt\n{'0' * 64} *c.pt\nnot a line\n")
    assert dict(vm.verify(tmp_path, exp)) == {"a.pt": "ok", "b.pt": "MISMATCH", "c.pt": "MISSING"}


def test_main_exit_codes(tmp_path):
    (tmp_path / "a.pt").write_bytes(b"aaa")
    (tmp_path / "SHA256SUMS").write_text(f"{hashlib.sha256(b'aaa').hexdigest()} *a.pt\n", encoding="utf-8")
    assert vm.main(["x", str(tmp_path)]) == 0
    (tmp_path / "a.pt").write_bytes(b"changed")
    assert vm.main(["x", str(tmp_path)]) == 1


def test_the_release_list_names_the_two_models_with_the_hashes_of_the_experiments_and_the_notes_say_which_is_default():
    exp = vm.parse((REPO / "docs" / "releases" / "models-v1.sha256").read_text(encoding="utf-8"))
    assert exp == {"helmet_3cls_exp015.pt": "20ce0694eb1d8d81d6f579ede9b4d81c1aa1fbdf1d46e9391a63d04960fd7817",
                   "helmet_4cls_exp016.pt": "6452d6354fbd09cd873b7201efd618238e56c6b9034fcf9cd894f73ca02bb19b"}
    notes = (REPO / "docs" / "releases" / "models-v1.md").read_text(encoding="utf-8")
    assert all(h in notes for h in exp.values()) and "DEFAULT helmet model" in notes and "experiments/EXP-015-helmet-grouped-3cls" in notes
    cfg = (REPO / "configs" / "pipeline" / "default.yaml").read_text(encoding="utf-8")
    assert exp["helmet_3cls_exp015.pt"] in cfg                                           # the default config names the same sha256
