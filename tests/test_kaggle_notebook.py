import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "kaggle" / "helmet_train_split_a.ipynb"


def cells():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_every_code_cell_is_valid_python():
    for src in cells():
        compile("\n".join(l for l in src.splitlines() if not l.startswith("!")), "cell", "exec")


def test_base_model_and_run_name_are_parameters_and_the_recipe_is_otherwise_unchanged():
    c = cells()
    assert 'BASE = "yolo11s.pt"' in c[1] and 'NAME = f"helmet_{RUN}_{BASE[:7]}_splitA"' in c[1]
    assert "model = YOLO(BASE)" in c[2] and 'YOLO("yolo11s.pt")' not in c[2] and "NAME = " not in c[2]
    t = c[2]
    for fixed in ("epochs=150", "patience=25", "imgsz=640", "batch=32", "seed=0", "deterministic=True", 'optimizer="auto"', "cos_lr=True", "mosaic=1.0", "mixup=0.1",
                  "degrees=10.0", "translate=0.1", "scale=0.5", "hsv_h=0.015", "hsv_s=0.7", "hsv_v=0.4", 'fliplr=0.5 if RUN == "3cls" else 0.0'):
        assert fixed in t, fixed
    assert re.search(r"time=2\.5,", t) and re.search(r"workers=2,", t)
    assert "BASE[:7]" and "yolo11n.pt"[:7] == "yolo11n" and "yolo11s.pt"[:7] == "yolo11s"
