# EXP-021 – xử lý ở 10 FPS: `test_video.mp4` bước nhảy 3 (30 → 10 FPS), 2026-10-19

Cấu hình mặc định hiện tại (hysteresis bật, GMC tắt), model `helmet_best.pt`; pipeline được báo fps = 30 / 3 = 10 (`Pipeline(fps=10)`): mọi tham số tính bằng giây quy đổi lại (W = 5, K = 2, đóng sự kiện 10 frame, buffer tracker 10 ...; `docs/TIME_BASED_PARAMETERS.md`). Lấy mỗi frame thứ 3 (263 frame). Chưa có ground truth: các so sánh dưới đây nói về **khác biệt** so với chạy 30 FPS, không phải đúng/sai.
Tệp: `run_stride.py` (chạy), `compare.py` (so sánh), `compare.json`, `bench_summary.json`, `bench_stride1|3/` (benchmark `tools/benchmark_pipeline.py --stride`). Ảnh/video không kèm.

## Sự kiện: 10 FPS so với 30 FPS (cùng video, cùng weights)
| | 30 FPS | 10 FPS |
|---|---|---|
| sự kiện | 14 (13 no_helmet + 1 overload) | 10 (9 no_helmet + 1 overload) |
| ghép được (cùng loại, khoảng thời gian trong ±15 frame gốc, box chồng nhau) | 7 | 7 |
| chỉ có ở 30 FPS | 7: M_129 (71–146), M_175 (112–117), M_523 (451–484), M_573 (474–496), M_634 (533–588), M_707 (554–622), M_782 (582–595) | – |
| chỉ có ở 10 FPS | – | 3: M_372 (426–456), M_659 (471–483), M_725 (498–534) |

`diff_events` (dung sai 3 hoặc 6 frame gốc) báo 28 thiếu / 20 thêm / 1 đổi: **không đọc được** vì tracker đổi ID khi đổi fps (buffer, bước nhảy), nên mọi sự kiện lệch tên (`M_153` → `M_127`...). Phép ghép theo khoảng thời gian và box mới có nghĩa.

Khoảng thời gian của 7 sự kiện ghép được (frame gốc, 10 FPS − 30 FPS): bắt đầu −5…+16 frame (6 trong 7 sự kiện lệch ≤ 3 frame, tức ≤ 0.1 s), kết thúc −7…+17 frame:
| loại | 30 FPS | 10 FPS | lệch bắt đầu | lệch kết thúc |
|---|---|---|---|---|
| no_helmet | M_77 35–89 | M_69 36–87 | +1 | −2 |
| no_helmet | M_153 104–118 | M_127 99–135 | −5 | +17 |
| no_helmet | M_299 290–333 | M_392 306–333 | +16 | 0 |
| no_helmet | M_380 374–424 | M_505 375–423 | +1 | −1 |
| no_helmet | M_482 471–474 | M_620 474–474 | +3 | 0 |
| no_helmet | M_681 528–570 | M_869 531–570 | +3 | 0 |
| overload | M_350 328–337 | M_468 330–330 | +2 | −7 |

Đọc đúng: ở 10 FPS **mất 7 sự kiện và thêm 3**. Hai sự kiện chỉ có ở 30 FPS là M_175 và M_707, hai cờ mà phân tích EXP-020 thấy chỉ xuất hiện ở bản hiện tại (không có ở v0) và nhìn bằng mắt là người đội mũ; còn M_129, M_523, M_573, M_634, M_782 là các cờ có ở cả v0. Không có nhãn đúng nên không kết luận được 10 FPS "tốt hơn" hay "tệ hơn". Nguyên nhân (không tách được ở đây): K/W = 2/5 thay vì 1/3 (chặt hơn khi cửa sổ đầy: 3/5 so với 8/15), lượng tử hóa của `exit_ratio`, và tracker đổi ID/đứt track khi mất 2 trong 3 frame.

## Biển số
- Số lần gọi detector biển số cả lượt: **14 (30 FPS) so với 10 (10 FPS)**: mỗi sự kiện tìm thấy biển ngay lần thử đầu (`plate.attempts` = 1 ở cả 7 sự kiện ghép được, ở cả hai tốc độ). Ngân sách tối đa 3 s = 90 lần ở 30 FPS, 30 lần ở 10 FPS không bị chạm.
- Frame tìm thấy biển lệch −5…+16 frame gốc so với 30 FPS (cùng lệch với lúc bắt đầu sự kiện).

## Tốc độ xử lý (không báo FPS tuyệt đối; tỉ lệ so với v0 cùng phiên, 3 vòng xen kẽ, v0 luôn chạy mọi frame)
| phiên | frame xử lý mỗi lượt | thời gian `process` mỗi frame (p50) | v0 `process` (p50) | tỉ lệ stream vs v0 | tỉ lệ headless vs v0 |
|---|---|---|---|---|---|
| 30 FPS (bước 1) | 757 | 48–49 ms | 151–153 ms | **2.39×** (2.38–2.46) | 2.81× |
| 10 FPS (bước 3) | 233 | 48–51 ms | 150–156 ms | **2.46×** (2.29–2.52) | 2.89× |

- **Chi phí mỗi frame được xử lý không đổi** (≈ 49 ms ở cả hai, tỉ lệ so với v0 như nhau trong khoảng nhiễu): bước nhảy không làm pipeline nhanh hơn từng frame, chỉ làm **ít frame hơn** (263 so với 787 mỗi lượt), nên tổng thời gian tính toán mỗi giây video giảm khoảng 3 lần.
- **Phiên đo có tải nền** (`cpu_other` ≈ 35%): tỉ lệ so với v0 ổn định hơn FPS tuyệt đối; xem EXP-009 về độ nhiễu.
- Không suy ra được tốc độ trên thiết bị đích (G7): đây là RTX 3050 Laptop + CPU máy dev.

## Độ ổn định của tracking xe máy: 30 FPS và 10 FPS (`tracking_stability.py`, `tracking_stability.json`; chỉ báo số, chưa sửa gì)
Tracker của model helmet (BoT-SORT, GMC tắt, cấu hình mặc định), track lớp `motorcycle` trên `test_video.mp4`. Ba biến thể: 30 FPS (buffer 30 frame = 1 s); 10 FPS với buffer quy đổi về cùng thời lượng (10 frame, đúng như `Pipeline(fps=10)`); 10 FPS để nguyên buffer 30 frame (= 3 s) để tách ảnh hưởng của buffer.

| | 30 FPS | 10 FPS (buffer 10) | 10 FPS (buffer 30) |
|---|---|---|---|
| track duy nhất | 63 | 49 | 45 |
| độ dài track trung vị (s) | 1.37 | 1.30 | 1.80 |
| track ngắn hơn 1 s | 23 (36.5%) | 15 (30.6%) | 7 (15.6%) |
| ước lượng số lần đổi ID¹ | 2 | 0 | 0 |
| IoU hộp giữa hai frame liên tiếp của một track (trung vị / phân vị 10) | 0.957 / 0.895 | 0.860 / 0.764 | 0.861 / 0.764 |
| dịch chuyển tâm giữa hai frame (px, trung vị / phân vị 90) | 2.8 / 8.5 | 10.6 / 21.7 | 10.5 / 21.7 |
| bước có IoU < 0.3 | 0% | 0% | 0% |
| phát hiện mỗi frame xử lý | 3.3 | 3.2 | 3.2 |
¹ track kết thúc (không phải cuối video) rồi một track MỚI bắt đầu trong ≤ 0.5 s sau, tâm hộp đầu tiên cách tâm hộp cuối của track cũ ≤ 0.5 × cạnh lớn của hộp, ghép một-một; heuristic, **công suất thấp** (chỉ 2 trường hợp ở 30 FPS).

**Giả thuyết "ở 10 FPS xe dịch xa hơn nên tracker ghép kém hơn": không được ủng hộ trên clip này.** Quãng dịch giữa hai frame tăng đúng như dự đoán (≈ 3.8×: 2.8 → 10.6 px) nhưng chỉ làm IoU liên tiếp giảm từ 0.96 xuống 0.86, vẫn xa ngưỡng ghép; không có bước nào dưới 0.3. Số track duy nhất **giảm** (49 so với 63), tỉ lệ track ngắn hơn 1 s không tăng (30.6% so với 36.5%) và không phát hiện đổi ID. Lưu ý: cảnh này phần lớn là hàng xe đứng chờ đèn đỏ rồi chạy chậm, nên chuyển động nhỏ; kết luận không áp dụng cho dòng xe nhanh. Giữ buffer 30 frame ở 10 FPS cho track dài hơn (trung vị 1.80 s) và ít track ngắn hơn (15.6%), nhưng số liệu này không nói buffer dài có làm tăng nhầm ID hay không (không có nhãn ID).
Hệ quả: 7 sự kiện mất và 3 sự kiện thêm ở 10 FPS (phần trên) **không giải thích được bằng tracker bị đứt/đổi ID**; nguyên nhân còn lại (không tách được ở đây): K/W = 2/5, lượng tử hóa của cửa sổ, ít quan sát hơn mỗi xe.
