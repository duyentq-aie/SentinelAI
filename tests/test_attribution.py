"""CC BY 4.0 asks for: the title, the author, a link to the source, the licence name and link, and an indication of the changes. They are in the README, the data card and the release notes."""
import json
import re

from sentinel.config import REPO

DOCS = {"README.md": REPO / "README.md", "docs/DATA_CARD.md": REPO / "docs" / "DATA_CARD.md", "docs/releases/models-v1.md": REPO / "docs" / "releases" / "models-v1.md"}
HELMET_URL = "https://universe.roboflow.com/rajab/helmet-plate-detection-4"
PLATE_URL = "https://universe.roboflow.com/school-fuhih/vietnamese-license-plate-tptd0"
CC = "https://creativecommons.org/licenses/by/4.0/"


def text(name):
    return DOCS[name].read_text(encoding="utf-8")


def test_helmet_dataset_attribution_is_complete_everywhere():
    for name in DOCS:
        t = text(name)
        for need in ("helmet plate detection 4", "Rajab", HELMET_URL, "CC BY 4.0", CC, "2024"):
            assert need in t, (name, need)
        assert "@misc{ helmet-plate-detection-4_dataset" in t and "author = { Rajab }" in t, name


def test_changes_are_stated():
    for name in ("README.md", "docs/DATA_CARD.md", "docs/releases/models-v1.md"):
        t = text(name).lower()
        assert ("by video" in t or "theo video" in t), name                       # the split was redone by video
        assert "drops the `plate` class" in t or "bỏ lớp `plate`" in t, name
        assert "not redistributed" in t or "không được phân phối lại" in t, name


def test_plate_dataset_is_attributed_and_plate_best_is_still_not_published():
    for name in ("README.md", "docs/DATA_CARD.md"):
        t = text(name)
        assert PLATE_URL in t and "vietnamese license plate" in t and "school" in t and "@misc{ vietnamese-license-plate-tptd0_dataset" in t, name
    readme = text("README.md")
    assert "not verified" in readme and "`plate_best.pt` is not published" in readme
    assert "not verified that it was trained" in text("docs/releases/models-v1.md") or "not verified that `plate_best.pt`" in text("docs/releases/models-v1.md")


def test_the_helmet_dataset_is_no_longer_called_unverified_and_the_readme_is_not_private():
    for name in ("README.md", "docs/DEVELOPER_GUIDE.md", "docs/releases/models-v1.md"):
        t = text(name) if name in DOCS else (REPO / name).read_text(encoding="utf-8")
        assert "not independently verified" not in t and "chưa xác minh)" not in t, name
    assert "private for now" not in text("README.md") and "git clone https://github.com/duyentq-aie/SentinelAI.git; cd SentinelAI\n" in text("README.md")
    card = text("docs/DATA_CARD.md")
    assert "nguồn gốc bộ helmet ĐÃ XÁC MINH" in card and "CHƯA xác minh" in card                # helmet verified, plate_best.pt not


def test_bibtex_blocks_are_well_formed():
    for name in ("README.md", "docs/DATA_CARD.md"):
        blocks = re.findall(r"```bibtex\n(.*?)```", text(name), re.S)
        assert blocks, name
        for b in re.findall(r"@misc\{.*?\n\}", blocks[0], re.S):
            assert b.count("{") == b.count("}") and "year = {" in b and "url = {" in b


def test_tools_that_write_dataset_metadata_name_the_author():
    for f in ("tools/kaggle_train.py", "tools/make_upload_zip.py"):
        t = (REPO / f).read_text(encoding="utf-8")
        assert "Rajab" in t and "CC BY 4.0" in t and "not independently verified" not in t, f
    assert "unknown at the time" in (REPO / "experiments" / "EXP-000-kaggle-original" / "metrics.json").read_text(encoding="utf-8")
    json.loads((REPO / "experiments" / "EXP-000-kaggle-original" / "metrics.json").read_text(encoding="utf-8"))
