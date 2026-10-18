# EXP-011 – kiểm tra rò rỉ dữ liệu detector (2026-10-18)

Lệnh:
```
python tools/leakage_check.py --data data/raw/helmet --out experiments/EXP-011-data-leakage/helmet_leakage.json
python tools/leakage_check.py --data data/raw/plate  --out experiments/EXP-011-data-leakage/plate_leakage.json
```
Kết quả tóm tắt và diễn giải: `docs/DATA_CARD.md`. Helmet: 100% ảnh valid/test nằm trong nhóm frame liên tiếp có mặt ở split khác; 75.6% ảnh test có frame train của cùng video cách ≤ 1 frame. Plate: không kiểm tra được kề-frame (tên không có số frame); 102 file trùng byte trong cùng split.
Không chia lại, không train, không sửa data/raw.
