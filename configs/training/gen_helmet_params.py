"""One-off helper that wrote configs/training/helmet_yolo11s_v1.json from the `train_args` embedded in helmet_best.pt (kept for provenance).

  python configs/training/gen_helmet_params.py helmet_best.pt configs/training/helmet_yolo11s_v1.json

Only arguments that influence training are kept; paths, device, save locations and predict/export-only options are dropped (the Kaggle script sets
its own). Nothing here contacts Kaggle.
"""
import json
import sys

DROP = {"task", "mode", "data", "project", "name", "save_dir", "device", "resume", "exist_ok", "source", "model", "cfg", "tracker", "format", "keras", "optimize",
        "int8", "dynamic", "simplify", "opset", "workspace", "nms", "half", "dnn", "visualize", "augment", "agnostic_nms", "classes", "retina_masks", "embed",
        "show", "save_frames", "save_txt", "save_conf", "save_crop", "show_labels", "show_conf", "show_boxes", "line_width", "vid_stride", "stream_buffer",
        "save_json", "conf", "split", "profile", "time", "end2end", "verbose", "save", "save_period", "compile"}


def main(src, dst):
    import torch
    ck = torch.load(src, map_location="cpu", weights_only=False)
    ta = ck["train_args"]
    params = {k: v for k, v in sorted(ta.items()) if k not in DROP}
    meta = {"source_checkpoint": src, "ultralytics_version": ck.get("version"), "date": ck.get("date"), "names": {int(k): v for k, v in ck["model"].names.items()},
            "base_model": ta.get("model"), "note": "train_args of the checkpoint; optimizer 'auto' resolved to AdamW lr0 0.001429 on Kaggle (EXP-000)"}
    json.dump({"meta": meta, "params": params}, open(dst, "w", encoding="utf-8"), indent=2)


if __name__ == "__main__":
    main(*sys.argv[1:3])
