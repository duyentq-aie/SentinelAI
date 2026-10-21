# EXP-005 – No-helmet và chở quá người theo vote cửa sổ (P2 3.2)

## Summary (English)
- Purpose: replace the per-frame counters of the no-helmet / overload rules with a windowed vote (W = 15 observations, K = 5 votes, r = 0.5), later with hysteresis (exit_ratio 0.3).
- Result: 14 events on 14 tracks (13 no_helmet, 1 overload), one event per track, all with a plate; versus baseline v2 one event is lost (M_1), two are added (M_176, M_702) and one overload label disappears (M_677); hysteresis changes no start frame but merges or lengthens intervals (7 of 13 tracks), e.g. M_176 becomes frames 112-117.
- Conclusion: the rules behave as designed in unit tests; M_176 was a 1-frame event at the exact boundary (5/10 = 0.50), caused by the single threshold without hysteresis, not by a code bug.
- Limits: not validated against ground truth; W, K, r and exit_ratio 0.3 are reasonable defaults, not tuned on labelled data; a longer flag may spill into frames where the person already wears a helmet (unknown).

**Chưa kiểm định bằng ground truth.** Mọi so sánh dưới đây là với snapshot của chính pipeline (baseline v2), không phải với sự thật.
Tham số mặc định chưa được tinh chỉnh trên dữ liệu có nhãn: W = 15 quan sát, K = 5 phiếu, r = 0.5 (`configs/pipeline/default.yaml`, `rules.helmet.*` / `rules.overload.*`).

## Luật (`sentinel/rules/windowed.py`)
- Với mỗi track xe máy, một *quan sát* = một frame có ≥ 1 đầu gắn vào xe. Frame không có đầu (che khuất) **không phải quan sát**: không bỏ phiếu và không đẩy phiếu cũ ra khỏi cửa sổ.
- Cửa sổ giữ W quan sát gần nhất. Vi phạm khi `phiếu ≥ K` **và** `phiếu / số quan sát ≥ r`.
  Phiếu no-helmet = có ≥ 1 đầu no-helmet; phiếu overload = số đầu > `max_passengers` (2).
- 1 sự kiện / (track, loại): `evidence.one_event_per_track: true` (mặc định) chặn sự kiện trùng khi kết luận dao động quanh ngưỡng.
- Engine chọn qua `rules.engine` (`windowed` mặc định, `legacy` giữ để tái tạo baseline).

## Unit test (track giả lập, `tests/test_rules_windowed.py`, 16 test)
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

## Phụ lục (2026-10-18): M_176 – lỗi của luật hay của cách tính khoảng sự kiện?
Công cụ: `python tools/rule_trace.py --track 176` (in từng frame: số đầu, phiếu, cửa sổ, tỉ lệ, kết luận). Kết quả cho M_176 (fixture GMC bật):

| frame | đầu / no-helmet | quan sát | phiếu | tỉ lệ | kết luận |
|---|---|---|---|---|---|
| 101–108 | 1 đầu; no-helmet ở 101, 102, 106, 107 | 1…8 | 1…4 | 1.00 → 0.50 | không |
| 109–110 | **0 đầu** (không phải quan sát) | 8 | 4 | 0.50 | không |
| 111 | 1 đầu, mũ | 9 | 4 | 0.44 | không |
| **112** | 1 đầu, **no-helmet** | **10** | **5** | **0.50** | **CÓ** (votes ≥ K=5 và ratio = r) |
| 113 | 1 đầu, mũ | 11 | 5 | 0.46 | không |

**Chẩn đoán.** Cài đặt đúng như đặc tả (không có lỗi code); hiện tượng do hai tính chất ghép lại:
1. **Ngưỡng bao gồm dấu bằng, trùng đúng biên:** phiếu thứ 5 đến cùng lúc `ratio = 0.50 = r` (≥), frame sau chỉ cần một quan sát sạch là rơi xuống 0.46.
2. **Kết luận không có trí nhớ (không hysteresis)** và khoảng sự kiện được định nghĩa là các frame có kết luận ⇒ `frame_start = frame_end = 112`.
Nên đây **vừa là điểm yếu của đặc tả luật (một ngưỡng, không hysteresis), vừa là hệ quả của cách tính khoảng**; nhưng không phải lỗi cài đặt. Bản thân tín hiệu (5 phiếu rải trên 12 frame, 4 phiếu đầu cách xa phiếu thứ 5) yếu: có thể là người không đội mũ bị phát hiện chập chờn, cũng có thể là người đội mũ bị nhận nhầm. **Không có ground truth nên không biết nó có phải vi phạm thật.**

Các phương án, đo trên cả hai fixture (`experiments/EXP-005-windowed-rules/m176_variants.json`; kết quả giống nhau cho GMC bật/tắt):
| phương án | tác động trên clip này |
|---|---|
| ngưỡng chặt `ratio > r` | chỉ M_176 biến mất (13 → 12 track) |
| **hysteresis** (`exit_ratio` 0.3: đã vào thì giữ cờ tới khi tỉ lệ < 0.3) | không track nào thêm/bớt; M_176 thành 112–117 (6 frame), M_153 104–118, M_677 528–570: các khoảng dài và mượt hơn, hết sự kiện 1 frame |
| debounce (kết luận phải đúng ≥ 3 frame mới mở sự kiện) | M_176 và M_476 (471–474) biến mất (13 → 11 track) |

**Quyết định:** không đổi mặc định (đổi theo một clip là tune trên nhiễu). Đã thêm tham số tùy chọn `rules.helmet.exit_ratio` / `rules.overload.exit_ratio` (mặc định `None` = hành vi cũ, có test, `tests/test_rules_windowed.py`) để thử trên video dev có nhãn: khi đó đo F1 sự kiện, độ trễ và số sự kiện ≤ 2 frame cho (a) mặc định, (b) `exit_ratio` 0.3, (c) debounce, và chọn theo ground truth.

## Phụ lục 2 (2026-10-18): hysteresis `exit_ratio: 0.3` bật mặc định – **chưa tinh chỉnh, chờ dev video**
- Ngưỡng 0.3 là giá trị thử trong `m176_variants.json`, không chọn bằng dữ liệu có nhãn.
- `diff_events` baseline v4 (không hysteresis) → v5 (có): **MISSING 0 / ADDED 0 / CHANGED 0** trên 29 sự kiện (`events_v5_hysteresis_cpu.json`): thời điểm bắt đầu và số sự kiện không đổi, vì hysteresis chỉ giữ cờ lâu hơn sau khi đã bật. (`diff_events` so frame bắt đầu, không so frame kết thúc.)
- Thay đổi nằm ở **khoảng sự kiện** (`intervals_default_fixture_plain_vs_hyst.json`, fixture mặc định, 13 track có kết luận, 7 đổi): M_129 (71–120, 126–178 → 71–178: hết tách đôi), M_300 (290–303, 307–307 → 290–333), M_176 (112–112 → 112–117: hết sự kiện 1 frame), M_153 +5 frame, M_571/632/679 +2…4 frame.
- Rủi ro: cờ kéo dài thêm có thể dính sang đoạn đã đội mũ lại (chưa biết trên ground truth). Test: `tests/test_rules_windowed.py`, `tests/test_rules_offline.py` (v5).
