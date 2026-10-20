# EXP-020 – chẩn đoán "báo sai nhiều hơn, ảnh biển mờ hơn" sau refactor (2026-10-19)

Không kèm ảnh (có người và biển số). Ảnh nằm ở `outputs/regression_diag/` (gitignored): `contact4_1..6.jpg` (contact sheet 4 phiên bản, 28 hàng), `contact_sheet_1..5.jpg` (v0 so với hiện tại), `plates_v0_vs_current_1..3.jpg`. Báo cáo gốc: `outputs/regression_diag/REPORT.md`.

## Phát hiện chính
1. **v0 cho model helmet và model biển số nhìn frame đã bị vẽ đè**: `app.py` gốc vẽ trên đúng frame đang đưa vào model (khung đèn giao thông, nhãn, khung xe), nên model helmet thấy khung đèn và model biển số thấy các khung vi phạm. **Bản hiện tại đã sửa** (perception chạy trên frame gốc `raw`, vẽ lên bản sao `display`, từ bước 3.0b). Trên clip này điều đó **không làm đổi số track no-helmet** (bảng 2×2 dưới đây: perception không giải thích khác biệt nào), nhưng nó làm khác ảnh crop biển lưu: crop v0 chứa khung xanh `PLATE` và nhãn vẽ sẵn, crop hiện tại sạch.
2. **Khác biệt về số cờ no-helmet đến từ luật, không từ perception**: bảng 2×2 (phát lại offline các tracks đã ghi; wrong-way tắt):

| perception | luật | track no-helmet | track overload | track-frame bị gắn cờ |
|---|---|---|---|---|
| v0 perception (overlay-contaminated, default tra | legacy | 12 | 2 | 314 |
| v0 perception (overlay-contaminated, default tra | windowed (no hysteresis) | 13 | 1 | 350 |
| v0 perception (overlay-contaminated, default tra | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |
| clean frames + GMC on (3.0b fix, default until 1 | legacy | 12 | 2 | 314 |
| clean frames + GMC on (3.0b fix, default until 1 | windowed (no hysteresis) | 13 | 1 | 350 |
| clean frames + GMC on (3.0b fix, default until 1 | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |
| current default (clean + GMC off) | legacy | 12 | 2 | 314 |
| current default (clean + GMC off) | windowed (no hysteresis) | 13 | 1 | 350 |
| current default (clean + GMC off) | windowed + hysteresis 0.3 (current default) | 13 | 1 | 389 |

   Cột "luật": `legacy` = v0; `windowed (no hysteresis)`; `windowed + hysteresis 0.3` = mặc định hiện tại. Windowed thêm 1 track và +36 track-frame (+11%); hysteresis thêm +39 (+11%): tổng hộp đỏ giữ lâu hơn ≈ 24%.
3. **Mức nền báo sai giống nhau ở mọi bản**: 11 trong 13 sự kiện no-helmet hiện tại cũng có ở v0; nhiều cờ là người đội mũ.
4. **Lặp video**: cả v0 (3 vòng không reset: 12 track và 20 file biển mỗi vòng, id mới mỗi vòng) và bản hiện tại lặp lại cùng các xe mỗi vòng; bản hiện tại nhìn rõ hơn vì có panel sự kiện (đã sửa ở dưới).
5. **Ảnh biển**: cùng kích thước crop (diện tích trung vị 1928 so với 1941 px), cùng JPEG q95, cùng cắt từ frame 1280×720; độ nét trong khung biển: v0 2264, hiện tại 2962 (trung vị Laplacian); không có bằng chứng ảnh lưu mờ hơn. Cái có thể làm biển trông mờ hơn khi xem web là **stream MJPEG chất lượng thấp** (xem dưới) và **CSS phóng to thumbnail**.

## Sự kiện của 4 phiên bản trên cùng video (một lượt, GPU, cùng cấu hình mặc định hiện tại)
| phiên bản | no_helmet | overload | wrong_way | tổng sự kiện (hàng) |
|---|---|---|---|---|
| v0 | 12 | 2 | 8 | 22 |
| current+old model | 13 | 1 | 0 | 14 |
| EXP-015 3cls | 7 | 0 | 0 | 7 |
| EXP-016 4cls | 5 | 0 | 0 | 5 |

"hiện tại + model cũ" = `helmet_best.pt`; "EXP-015 3cls" và "EXP-016 4cls" = pipeline hiện tại với model helmet tương ứng (EXP-016 chỉ dùng cho vai trò helmet; biển số vẫn là `plate_best.pt`). Ghép theo thời gian ±1 s và box chồng nhau; 28 hàng, số hàng cũ giữ nguyên (1–24), hàng mới 25–28. **Chưa có nhãn đúng/sai**: `grading_template.csv` để bạn chấm (đúng / sai / không rõ) cho từng phiên bản; số lượng sự kiện ít hơn không có nghĩa là ít báo sai hơn.

## Sửa đổi trong ứng dụng web (cùng đợt)
- `web.jpeg_quality` mặc định **90** (trước 80). Đo trên 12 frame có biển (1280×720, PSNR so với frame gốc), trung vị:

| chất lượng | kích thước frame | PSNR vùng biển | PSNR cả frame |
|---|---|---|---|
| 80 | 192.7 KB | 35.2 dB | 36.5 dB |
| 90 | 270.3 KB | 38.9 dB | 39.4 dB |
| 95 | 374.0 KB | 42.4 dB | 41.8 dB |

  Mức 90 đổi +40% băng thông so với 80 để được +3.7 dB ở vùng biển; mức 95 gần gấp đôi dung lượng của 80. Chỉ ảnh hưởng tới luồng stream; ảnh bằng chứng luôn ≥ 95.
- **Panel và gallery chỉ hiển thị ảnh bằng chứng đã lưu** (`frame.jpg`, `vehicle.jpg`, `plate.jpg`, q ≥ 95, từ `static/violations`), không lấy từ frame của stream (kiểm bằng code: `/api/violations` và `/api/events` trả URL file; stream chỉ cấp phần tử video). **CSS thumbnail**: trước đây thumbnail biển trong panel dùng `object-fit: cover` (cắt mất một phần crop 46×40 px khi phóng ~2.5×) và gallery phóng ~3× bằng làm mượt của trình duyệt (mờ). Nay biển dùng `object-fit: contain` và `image-rendering: pixelated`. (Đây là thay đổi theo đọc mã và CSS, chưa so ảnh chụp màn hình.)
- **Panel sự kiện**: mỗi thẻ ghi `run N`; mặc định chỉ hiện sự kiện của run hiện tại (một lượt qua video), có nút "show all runs"; API `/api/events?scope=current|all`, `/api/status` có `run`. Có test.

## Cập nhật: phiên bản thứ 5 "hiện tại + model cũ @10 FPS" (2026-10-19)
`contact5_1..7.jpg` (trong `outputs/regression_diag/`, gitignored) và `event_rows_5_versions.json`: cùng 28 hàng cũ (số hàng giữ nguyên) cộng cột thứ 5 là pipeline hiện tại + `helmet_best.pt` xử lý ở 10 FPS (mỗi frame thứ 3, EXP-021); ghép theo thời gian ±1 s + IoU; **3 hàng mới 29–31** (sự kiện của bản 10 FPS không khớp hàng cũ nào: M_372 @14.2 s, M_620 @15.8 s, M_725 @16.6 s). `grading_template.csv` có thêm cột cho phiên bản thứ 5 (cột "verdict" để trống cho bạn chấm). Số sự kiện: v0 22, hiện tại + model cũ 14, EXP-015 7, EXP-016 5, 10 FPS 10.

<!-- grade_rows:start -->
## Precision by eye (`tools/grade_rows.py`)

Graded by the project owner by eye, 23 of 23 rows (no_helmet / overload; wrong_way rows skipped: that rule is off). The verdict of a row applies to every version that has an event in it, unless the row was marked as different people (then each version was graded on its own). Precision = right / (right + wrong), unclear left out.

| version | events graded | right | wrong | unclear | precision | relative coverage |
|---|---|---|---|---|---|---|
| v0 | 14 | 0 | 14 | 0 | 0.00 (0/14) | 0/0 |
| current+old model | 14 | 0 | 14 | 0 | 0.00 (0/14) | 0/0 |
| EXP-015 3cls | 7 | 0 | 7 | 0 | 0.00 (0/7) | 0/0 |
| EXP-016 4cls | 5 | 0 | 5 | 0 | 0.00 (0/5) | 0/0 |
| current+old model @10FPS | 10 | 0 | 10 | 0 | 0.00 (0/10) | 0/0 |

Relative coverage = among the 0 rows where at least one version was graded right, how many each version caught. **It is NOT recall**: a violation that all five versions missed is in nobody's denominator, so the true number of violations in the clip is unknown.

**Read with care:** n is small (every precision above rests on a handful of events), the whole evidence is ONE 26-second clip, and the rules (windows, votes, hysteresis, the ratios) were adjusted on this very clip, so the numbers say little about new footage. The versions differ in model AND settings (see the table above), so a difference between them is not attributed to one cause. The default model is not changed by this table.
<!-- grade_rows:end -->
