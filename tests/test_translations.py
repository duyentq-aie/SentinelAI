"""Checks for the English experiment logs.

Linked experiments (README.md / NOTES.md in English, original kept as *.vi.md):
every number of the Vietnamese original must appear in the English file, the
relative links must resolve, the tables must have the same number of rows.
Other experiments keep their Vietnamese body and get a short English summary.
"""
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads((ROOT / "tests" / "fixtures" / "translations_vi_sha256.json").read_text(encoding="utf-8"))

HEADER = "English translation of {name} (original, Vietnamese)."
README_NOTE = (
    "Experiment logs linked from this README are in English (Vietnamese originals kept as README.vi.md); "
    "other logs have an English summary at the top."
)

VI_FILES = sorted(FIXTURE["translated_originals"])
SUMMARY_FILES = sorted(FIXTURE["summary_only_originals"])

NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
RATIO_RE = re.compile(r"\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?")
PCT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
LINK_RE = re.compile(r"\]\(([^)\s]+)\)")


def _text(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _sha(text):
    return hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def numbers(text):
    """Every integer / decimal, percentage and x/y ratio, normalised (decimal comma -> point, no spaces)."""
    found = set()
    for m in NUM_RE.findall(text):
        found.add(m.replace(",", "."))
    for m in RATIO_RE.findall(text):
        found.add(re.sub(r"\s+", "", m))
    for m in PCT_RE.findall(text):
        found.add(re.sub(r"\s+", "", m).replace(",", "."))
    return found


def table_rows(text):
    return sum(1 for line in text.splitlines() if line.lstrip().startswith("|"))


def relative_links(text):
    out = []
    for target in LINK_RE.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target) or target.startswith("#"):
            continue
        out.append(target)
    return out


def english_of(vi_rel):
    return ROOT / vi_rel.replace(".vi.md", ".md")


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_original_is_unchanged(vi_rel):
    assert _sha(_text(ROOT / vi_rel)) == FIXTURE["translated_originals"][vi_rel]


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_english_first_line(vi_rel):
    vi = ROOT / vi_rel
    first = _text(english_of(vi_rel)).splitlines()[0]
    assert first == HEADER.format(name=vi.name)


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_every_number_of_the_original_is_in_the_translation(vi_rel):
    vi_nums = numbers(_text(ROOT / vi_rel))
    en_nums = numbers(_text(english_of(vi_rel)))
    missing = sorted(vi_nums - en_nums)
    assert not missing, "numbers only in the Vietnamese original: %s" % missing


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_numbers_only_in_translation_are_the_header_or_known(vi_rel):
    # The reverse direction is reported, not required: the translation may add
    # nothing but the words of the header line (which contains no digit).
    vi_nums = numbers(_text(ROOT / vi_rel))
    en_nums = numbers(_text(english_of(vi_rel)))
    extra = sorted(en_nums - vi_nums)
    assert not extra, "numbers only in the English file: %s" % extra


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_links_resolve(vi_rel):
    for rel in (vi_rel, vi_rel.replace(".vi.md", ".md")):
        path = ROOT / rel
        for target in relative_links(_text(path)):
            target_path = target.split("#", 1)[0]
            if not target_path:
                continue
            assert (path.parent / target_path).exists(), "%s: broken link %s" % (rel, target)


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_links_match_the_original(vi_rel):
    assert sorted(relative_links(_text(ROOT / vi_rel))) == sorted(relative_links(_text(english_of(vi_rel))))


@pytest.mark.parametrize("vi_rel", VI_FILES)
def test_table_row_counts_equal(vi_rel):
    assert table_rows(_text(ROOT / vi_rel)) == table_rows(_text(english_of(vi_rel)))


def _strip_summary(text):
    m = re.search(r"\n## Summary \(English\)\n(?:- .*\n)+", text)
    assert m, "no summary block"
    return text[: m.start()] + text[m.end():], text[m.start():m.end()]


@pytest.mark.parametrize("rel", SUMMARY_FILES)
def test_summary_is_at_the_top_and_3_to_5_lines(rel):
    text = _text(ROOT / rel)
    lines = text.splitlines()
    assert lines[0].startswith("# ")
    assert lines[2] == "## Summary (English)"
    bullets = []
    for line in lines[3:]:
        if not line.strip():
            break
        bullets.append(line)
    assert 3 <= len(bullets) <= 5, len(bullets)
    assert text.count("## Summary (English)") == 1
    # purpose, result, conclusion, limits
    joined = " ".join(bullets)
    assert "Limits:" in joined and "Purpose:" in joined and "Result" in joined and "Conclusion:" in joined


@pytest.mark.parametrize("rel", SUMMARY_FILES)
def test_summary_files_keep_the_vietnamese_body(rel):
    body, _ = _strip_summary(_text(ROOT / rel))
    assert _sha(body) == FIXTURE["summary_only_originals"][rel]


@pytest.mark.parametrize("rel", SUMMARY_FILES)
def test_summary_has_numbers_and_honest_limits(rel):
    _, block = _strip_summary(_text(ROOT / rel))
    assert NUM_RE.search(block), "summary has no number"


def test_root_readme_language_note():
    assert README_NOTE in _text(ROOT / "README.md")


def test_root_readme_experiment_links_are_translated():
    """Every experiment linked from the root README must have an English file with a .vi.md original."""
    linked = set()
    for target in relative_links(_text(ROOT / "README.md")):
        m = re.match(r"experiments/(EXP-\d{3}[^/]*)/(README|NOTES)\.md$", target)
        if m:
            linked.add(target)
    assert linked
    for target in linked:
        assert (ROOT / target.replace(".md", ".vi.md")).exists(), target
        assert _text(ROOT / target).startswith("English translation of ")
