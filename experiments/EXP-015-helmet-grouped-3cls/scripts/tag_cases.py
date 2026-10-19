import json, collections, csv
d = "experiments/EXP-015-helmet-grouped-3cls/"
cases = json.load(open(d + "val_error_cases.json", encoding="utf-8"))
def tag(c):
    k, v, i = c["kind"], c["video"], c["id"]
    if k.startswith("a") or k == "other":
        if v == "frame_14": return "lỗi model", "đầu trần rõ, GT đúng (hộp GT dài gồm cả lưng); model đoán helmet vì mũ của người lái nằm chồng trong cùng hộp"
        if v == "frame_10": return "không chắc", "người trùm mũ vải/hood, mũ bảo hiểm của người kia chồng trong cùng hộp; GT hợp lý nhưng hai đầu chồng nhau"
        return "không chắc", "khăn trùm đầu (hijab); chưa có quy ước nhãn cho khăn; mũ người lái chồng lên"
    if k.startswith("b"):
        return ("lỗi model", "đầu trần rõ, không có hộp đầu nào") if v == "frame_14" else ("không chắc", "khăn trùm đầu; model không thấy đầu nào")
    if i == "E00": return "nhãn thiếu", "người đội mũ bảo hiểm rõ nhưng KHÔNG có hộp GT helmet; model còn đoán no-helmet nên cũng là lỗi model"
    if i == "E49": return "không chắc", "định vị: hộp GT dài gồm khăn + lưng, hộp dự đoán chỉ phần đầu (IoU < 0.5)"
    if i in ("E07", "E08"): return "lỗi model", "đèn hậu ô tô bị đoán no-helmet"
    return "lỗi model", "dự đoán no-helmet trên thân/lưng dưới một đầu đã có nhãn (thường là helmet): hộp trùng thân người"
for c in cases:
    c["tag"], c["why"] = tag(c)
json.dump(cases, open(d + "val_error_cases.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
with open(d + "val_error_cases.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(["id", "kind", "video", "image", "conf", "iou", "tag", "why"])
    for c in cases:
        w.writerow([c["id"], c["kind"], c["video"], c["img"].split("/")[-1], c["conf"], c["iou"], c["tag"], c["why"]])
cnt = collections.Counter((c["kind"], c["tag"]) for c in cases)
for k, v in sorted(cnt.items()): print(k, v)
print(collections.Counter(c["tag"] for c in cases))
print(sorted(collections.Counter((c["kind"][0], c["video"]) for c in cases).items()))
print("distinct images", len({c["img"] for c in cases}))
