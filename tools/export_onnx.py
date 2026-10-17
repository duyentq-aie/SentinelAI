"""Export the three models to ONNX (static 640, fixed opset) into models/onnx/ and record their hashes.

  python tools/export_onnx.py [--models coco helmet plate] [--imgsz 640 [W]] [--opset 17] [--half] [--dynamic]
  --imgsz 640        square 640x640 input (what `imgsz=640` means); a frame is letterboxed to a square
  --imgsz 384 640    static HEIGHT x WIDTH: the shape PyTorch itself uses for the pipeline's 1280x720 frames (rectangular inference), so
                     the ONNX model does the same amount of work and sees the same padding

Output files (gitignored: *.onnx): models/onnx/<name>_<imgsz>[_fp16].onnx, plus models/onnx/export.json (sha256 of source weights and
exported files, versions). --half exports FP16 weights (needs a CUDA device; fails cleanly otherwise).
Select an exported model in a pipeline config by pointing `models.<name>.weights` at it (see configs/pipeline/onnx*.yaml).
"""
import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SOURCES = {"coco": "yolo11s.pt", "helmet": "helmet_best.pt", "plate": "plate_best.pt"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640], help="N (square) or H W")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--dynamic", action="store_true", help="dynamic H/W (needed for the plate model, which runs on crops of any size)")
    ap.add_argument("--out", default="models/onnx")
    a = ap.parse_args()

    import onnx
    import onnxruntime
    import torch
    import ultralytics
    from ultralytics import YOLO

    out = REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    info_path = out / "export.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
    for name in a.models:
        src = REPO / SOURCES[name]
        model = YOLO(str(src))
        t0 = time.time()
        imgsz = a.imgsz[0] if len(a.imgsz) == 1 else list(a.imgsz)
        tag = f"{a.imgsz[0]}" if len(a.imgsz) == 1 else f"{a.imgsz[0]}x{a.imgsz[1]}"
        produced = model.export(format="onnx", imgsz=imgsz, opset=a.opset, simplify=True, dynamic=a.dynamic, half=a.half,
                                device=0 if a.half else "cpu", verbose=False)
        dst = out / f"{name}_{tag}{'_dyn' if a.dynamic else ''}{'_fp16' if a.half else ''}.onnx"
        shutil.move(str(produced), dst)
        m = onnx.load(str(dst))
        onnx.checker.check_model(m)
        inp = m.graph.input[0]
        shape = [d.dim_value or d.dim_param for d in inp.type.tensor_type.shape.dim]
        info[dst.name] = {"source": SOURCES[name], "source_sha256": sha256(src), "sha256": sha256(dst), "bytes": dst.stat().st_size,
                          "input_shape": shape, "opset": a.opset, "half": a.half, "simplify": True, "dynamic": a.dynamic,
                          "export_seconds": round(time.time() - t0, 1)}
        print(f"{dst.name}: {dst.stat().st_size / 2**20:.1f} MiB, input {shape}, checker OK")
    info["_env"] = {"onnx": onnx.__version__, "onnxruntime": onnxruntime.__version__, "providers": onnxruntime.get_available_providers(),
                    "torch": torch.__version__, "ultralytics": ultralytics.__version__}
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
