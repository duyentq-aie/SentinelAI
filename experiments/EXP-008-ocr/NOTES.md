# EXP-008 – Plate OCR, basic version (đêm 2026-10-18, mục 6)

**Không có claim độ chính xác** (không có ground truth biển số). Kết luận thực tế của thí nghiệm: *chuỗi OCR hoạt động về mặt cơ học, nhưng trên video 720p của dự án biển số quá nhỏ để đọc*.

## Chuỗi xử lý (`sentinel/ocr/`)
1. **Chọn crop tốt nhất của track** (`quality.py`): điểm = 0.5·kích thước + 0.5·độ nét (phương sai Laplacian), mỗi thành phần bị chặn (4× / 3× giá trị tham chiếu). Detector biển tiếp tục chạy cho tới khi gom đủ `ocr.candidates` (5) crop cách nhau ≥ `min_gap` (3) frame, hoặc hết `max_attempts`, hoặc vi phạm kết thúc. `plate.jpg` được ghi đè bằng crop tốt nhất.
2. **Tiền xử lý OpenCV** (`preprocess.py`): crop sát hộp biển (`ocr.margin` = 2 px, thay vì 10 px nền), phóng to bicubic tới cao 128 px (tối đa ×6), CLAHE trên kênh L, **tách biển hai dòng** (xe máy) tại hàng có ít "mực" nhất quanh giữa.
3. **Nhận dạng** (`engine.py`): PaddleOCR 3.4 `TextRecognition(PP-OCRv5_mobile_rec)` (chỉ phần recognition, không detection), từng dòng, ghép lại. Model tải về từ Hugging Face ở lần đầu (cache ngoài repo: `~/.paddlex/`).
4. **Chuẩn hóa + định dạng biển VN** (`normalize.py`): bỏ ký tự phân cách, sửa ký tự dễ nhầm theo vị trí (O→0, B↔8, S↔5, …) với mẫu 8–10 ký tự; chữ cái seri chỉ lấy trong `ABCDEFGHKLMNPSTUVXYZ` (biển VN không dùng I, J, O, Q, R, W); regex `^\d{2}[letters]{1,2}\d?\d{4,5}$`. Chỉ kiểm hình dạng chuỗi, không kiểm biển có tồn tại hay đọc đúng.
5. **Vote** (`vote.py`): đọc `reads` (3) crop tốt nhất; độ dài được ủng hộ nhiều nhất thắng, rồi bỏ phiếu từng ký tự có trọng số theo độ tin cậy.
6. Kết quả ghi vào `events.jsonl` ở `plate_pred` (`docs/EVIDENCE.md`); lỗi PaddleOCR được chặn lại (sự kiện vẫn đầy đủ).
Bật bằng `configs/pipeline/ocr.yaml` (mặc định tắt).

Kiểm thử: test đơn vị cho từng khối (`tests/test_ocr_pure.py`: chất lượng crop, tiền xử lý, chuẩn hóa/sửa ký tự, vote, PlateOCR) và 9 test cho tích hợp trong recorder (`tests/test_evidence_ocr.py`) với recognizer giả (số crop được gom, crop tốt nhất thay `plate.jpg`, OCR chạy một lần/track, lỗi không làm sập pipeline, reset). Các test này **không** dùng PaddleOCR thật.

## Kết quả trên video test (CPU, `record_tracks --config configs/pipeline/ocr.yaml`)
- Sự kiện giống hệt baseline v3 (diff rỗng): bật OCR không đổi luật hay sự kiện.
- **14 sự kiện, 14 có `plate_pred`; 0 có chuỗi đúng định dạng biển số.** Kích thước crop biển lưu: 38–51 × 36–49 px (trung bình 46×44, **gồm viền 10 px nền**; hộp biển do detector trả về đo được ≈ 26×22 px (EXP-010 `mean_plate_box_wh`); 46×44 là kích thước crop lưu, không phải biển).
- Đầu ra thô của recogniser là mảnh vụn hoặc rỗng (`''`, `'0'`, `'M'`, `'BD'`, một số chữ Hán). Khi crop là 64 px cao và có viền 10 px, model PP-OCRv5 gần như chỉ "bịa" ký tự Hán (lần chạy đầu: 0/14 hợp lệ); sau khi cắt sát biển và phóng to tới 128 px, một phần đầu ra chuyển sang chữ/số Latin (vẫn còn vài ký tự Hán) nhưng không thành biển số hợp lệ (0/14).
  (Đã thử vài biến thể tiền xử lý trên 14 crop đã lưu bằng script tạm, không commit; **không** chọn tham số để "tăng số chuỗi hợp lệ" vì không có ground truth, sẽ là tune trên nhiễu. Hai thay đổi được giữ — crop sát biển và phóng to lớn hơn — vì có lý do nguyên tắc: ít điểm ảnh nền hơn và chữ đủ lớn cho recogniser.)
- **Độ trễ OCR / sự kiện** (3 lần đọc, CPU, PaddleOCR mobile): trung bình **209 ms**, trung vị 220 ms, min 68, max 377 ms. Chạy trong luồng suy luận (chặn pipeline ~0.2 s mỗi sự kiện). Nạp model: ~11 s ở lần đầu (tải) / vài giây khi đã cache.
- Ví dụ "không đọc được": toàn bộ 14 sự kiện của clip này. Ảnh ghép các crop: `outputs/ocr_examples/plate_crops_montage.jpg` (thư mục `outputs/` bị gitignore vì chứa biển số thật).

## Probe với biển tổng hợp (`tools/ocr_resolution_probe.py`, `resolution_probe.json`)
Biển hai dòng "29B1 / 12345" vẽ bằng OpenCV, thu nhỏ và làm suy giảm (blur Gaussian, JPEG q25), chạy cả chuỗi OCR. **Đây không phải độ chính xác trên ảnh thật** (một font, nét đậm, sạch):

| cao (px) | clean | blur 0.8 | blur 1.2 | JPEG 25 | blur 0.8 + JPEG 25 |
|---|---|---|---|---|---|
| 100 | đúng | đúng | đúng | đúng | đúng |
| 50 | đúng | đúng | đúng (sau sửa 8→B) | đúng | đúng (sau sửa 8→B) |
| 38 | đúng | đúng | đúng (sau sửa) | đúng (sau sửa) | **sai** (20B112345) |
| 25 | đúng | **sai** | **không hợp lệ** | **sai** | **sai** |

Với biển sạch, chuỗi (tiền xử lý + PaddleOCR + sửa ký tự + vote) đọc đúng; ở ~25 px cao chỉ cần blur 0.8 hoặc JPEG đã làm hỏng. Biển thật ở clip này nhỏ, nhòe, nén JPEG, nghiêng, nên không đọc được là hợp lý.

## Kết luận và việc cần làm
1. **Với camera 720p ở cự ly này OCR không khả thi.** Cần video 1080p+ hoặc camera gần hơn (biển ≥ ~40–50 px cao) — đã ghi vào `docs/RECORDING_GUIDE.md`. Không nên đổi mô hình OCR/hậu xử lý trước khi có dữ liệu đủ độ phân giải và nhãn biển số (`label_events.py` đã có trường `plate_text`).
2. Khi có nhãn: đo exact-match và CER trên crop GT và end-to-end (PLAN mục 3.4), tách ngày/đêm, 1/2 dòng.
3. Độ trễ ~0.2 s/sự kiện nên chuyển OCR sang luồng riêng nếu bật khi demo.
