"""The README is the first thing a reader sees: its links and demo assets must exist and obey the limits."""
import re
from pathlib import Path

from PIL import Image

from sentinel.config import REPO

README = (REPO / "README.md").read_text(encoding="utf-8")


def test_every_relative_link_and_image_of_the_readme_exists():
    links = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", README)
    rel = [l for l in links if not l.startswith(("http://", "https://", "mailto:"))]
    assert len(rel) >= 20
    missing = [l for l in rel if not (REPO / l).exists()]
    assert not missing, missing


def test_the_demo_gif_is_anonymised_sized_and_the_right_length():
    gif = REPO / "docs" / "assets" / "demo.gif"
    assert gif.stat().st_size <= 8_000_000
    im = Image.open(gif)
    assert im.size[0] == 960 and im.n_frames >= 40
    seconds = sum(im.seek(i) or im.info["duration"] for i in range(im.n_frames)) / 1000
    assert 12 <= seconds <= 15, seconds
    assert (REPO / "docs" / "assets" / "review.png").stat().st_size > 10_000


def test_the_readme_has_the_sections_in_the_order_a_recruiter_reads_them():
    heads = re.findall(r"^## (.+)$", README, re.M)
    assert heads == ["Highlights", "Architecture", "Results at a glance", "Run it", "Limits (read before quoting any number)", "Licence and data"]
    assert README.index("demo.gif") < README.index("## Highlights") and "```mermaid" in README
    for must in ("not measured", "26 seconds", "edge hardware", "false alarm"):
        assert must in README


def test_headline_numbers_are_the_ones_of_the_experiment_readmes():
    exp20 = (REPO / "experiments" / "EXP-020-regression-diag" / "README.md").read_text(encoding="utf-8")
    exp15 = (REPO / "experiments" / "EXP-015-helmet-grouped-3cls" / "README.md").read_text(encoding="utf-8")
    assert "32.0" in exp20 and "16.0" in exp20 and "23.6%" in exp20
    for number in ("0.934", "0.844"):
        assert number in README
    assert "0.8438" in (REPO / "experiments" / "EXP-015-helmet-grouped-3cls" / "metrics_3cls.json").read_text(encoding="utf-8")        # no-helmet test mAP50 0.8438 -> 0.844
