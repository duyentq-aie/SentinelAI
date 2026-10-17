# EXP-005 – No-helmet và chở quá người theo vote cửa sổ (P2 3.2)

**Chưa kiểm định bằng ground truth.** Mọi so sánh dưới đây là với snapshot của chính pipeline (baseline v2), không phải với sự thật.
Tham số mặc định chưa được tinh chỉnh trên dữ liệu có nhãn: W = 15 quan sát, K = 5 phiếu, r = 0.5 (`configs/pipeline/default.yaml`, `rules.helmet.*` / `rules.overload.*`).

## Luật (`sentinel/rules/windowed.py`)
- Với mỗi track xe máy, một *quan sát* = một frame có ≥ 1 đầu gắn vào xe. Frame không có đầu (che khuất) **không phải quan sát**: không bỏ phiếu và không đẩy phiếu cũ ra khỏi cửa sổ.
- Cửa sổ giữ W quan sát gần nhất. Vi phạm khi `phiếu ≥ K` **và** `phiếu / số quan sát ≥ r`.
  Phiếu no-helmet = có ≥ 1 đầu no-helmet; phiếu overload = số đầu > `max_passengers` (2).
- 1 sự kiện / (track, loại): `evidence.one_event_per_track: true` (mặc định) chặn sự kiện trùng khi kết luận dao động quanh ngưỡng.
- Engine chọn qua `rules.engine` (`windowed` mặc định, `legacy` giữ để tái tạo baseline).

## Unit test (track giả lập, `tests/test_rules_windowed.py`, 17 test)
nhiễu 1–2 frame không bao giờ kích hoạt; che khuất tạm thời không làm loãng tỉ lệ cũng không xóa phiếu; kết luận giữ nguyên qua đoạn che khuất; tỉ lệ thấp (1/3) không kích hoạt dù đủ 5 phiếu; phiếu cũ hết hạn khi người lái đội mũ;
overload cần ≥ 5 quan sát 3 đầu (1–2 frame 3 đầu không đủ); `OVERLOAD & NO-HELMET`; các track độc lập; tham số lấy từ config; reset; dọn cửa sổ của track đã biến mất; chọn engine và kiểm tra giá trị config.

## `diff_events` baseline v2 → windowed (CPU, test_video.mp4, dung sai 0 frame)
Baseline v2: 28 sự kiện (15 nhãn + 13 chụp biển). Windowed (baseline v3): 29 sự kiện (15 nhãn + 14 chụp biển). Sự kiện bằng chứng: 14 sự kiện / 14 track (13 `no_helmet`, 1 `overload`), **mỗi track đúng một sự kiện**, cả 14 track có biển.

| thay đổi | giải thích (bằng số liệu từ fixture tracks) |
|---|---|
| **MẤT** `M_1 NO-HELMET` @23 (và chụp biển M_1) | Luật cũ cờ 1 frame duy nhất vì bộ đếm cộng dồn các phiếu rời rạc (frame 4, 6, 11, 19–23). Windowed: trong 15 quan sát gần nhất có 6 phiếu nhưng tỉ lệ 6/15 = 0.40 < 0.5 → không kết luận. Người lái này thường có đầu đội mũ (2 đầu, 1 đầu no-helmet trong ~5 frame rồi hết). |
| **THÊM** `M_176 NO-HELMET` @112 (+ biển) | Biên giới: ở frame 112 cửa sổ có 5 phiếu / 10 quan sát = 0.50 đúng bằng ngưỡng, sang frame 113 tỉ lệ 5/11 = 0.45 rớt → **sự kiện chỉ dài đúng 1 frame (112–112)**. Đây là hành vi biên đáng nghi (nhiều khả năng dương tính giả); chưa chỉnh luật, ghi nhận để kiểm với ground truth. |
| **THÊM** `M_702 NO-HELMET` @554 (+ biển) | Đầu của xe này chỉ được phát hiện lẻ tẻ (nhiều frame 0 đầu): 4 phiếu no-helmet liên tiếp (536–539) rồi phiếu thứ 5 ở 554, tỉ lệ 5/7 = 0.71. Luật cũ bị bộ đếm giảm mỗi frame không thấy đầu nên không bao giờ đủ 5. Đây đúng là trường hợp "che khuất tạm thời" mà luật mới nhắm tới. |
| **ĐỔI** `M_677`: mất nhãn `OVERLOAD & NO-HELMET` @550 | Luật cũ cờ overload ngay trong 2 frame (550–551) có 3 đầu; windowed cần ≥ 5 quan sát 3 đầu → bỏ qua (nhiễu 2 frame). `NO-HELMET` của M_677 vẫn giữ, nhưng kết thúc sớm hơn (frame 567 thay vì 590) vì tỉ lệ phiếu tụt khi các quan sát đội mũ tích lũy. |
| **ĐỔI** thời điểm kích hoạt | M_153 +1, M_299 −1, M_350 +4 (overload: luật cũ cờ ngay khi >2 đầu trong một frame, luật mới đợi đủ 5 phiếu), M_380 +2, M_476 +3, M_625 −4 (luật cũ có bộ đếm giảm khi frame sạch nên chậm hơn). Chụp biển đi theo thời điểm này. |

16 sự kiện còn lại của baseline v2 không đổi frame nào. Số `chưa kiểm định`: không biết trong các thay đổi trên thay đổi nào đúng hơn.

## Giới hạn / việc cần ground truth
- Không có video dev/test có nhãn: W, K, r chỉ là giá trị mặc định hợp lý, chưa được chọn bằng dữ liệu.
- `M_176` cho thấy ngưỡng tỉ lệ ngay tại biên tạo sự kiện 1 frame; cần hysteresis (ngưỡng vào ≠ ngưỡng ra) hoặc `min` quan sát nếu ground truth xác nhận đó là dương tính giả.
- Luật vượt đèn đỏ vẫn là luật cũ (vùng cố định), thuộc bước 3.3.
