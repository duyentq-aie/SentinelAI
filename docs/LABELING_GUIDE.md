# Hướng dẫn gán nhãn sự kiện vi phạm

Công cụ: `python tools/label_events.py --video <clip> --split dev|test` → `eval_sets/events/<video_id>.json`.
Schema và các trường: `docs/EVIDENCE.md` (phần "GT label"); kiểm tra tự động bằng `sentinel.labeling.validate` mỗi khi lưu.

## Một sự kiện là gì
Một **loại vi phạm** của **một xe**. Xe vừa không đội mũ vừa chở 3 người = **2 sự kiện** (`no_helmet` và `overload`), mỗi cái có khoảng frame riêng.

| loại | gán nhãn khi | KHÔNG gán khi |
|---|---|---|
| `no_helmet` | **bất kỳ** người nào trên xe máy không đội mũ bảo hiểm (nhìn rõ đầu trong ≥ 5 frame) | mũ không rõ ràng (khăn, mũ vải, mũ lưỡi trai đội che đầu) → vẫn gán nếu chắc chắn không phải mũ bảo hiểm, đánh `hard` nếu phân vân; người đi bộ, xe đạp |
| `overload` | xe máy chở **> 2 người** (tính cả trẻ em; ghi `notes: "tre em"` vì luật có ngoại lệ) | đúng 2 người; người ngồi xe khác lọt vào khung |
| `red_light` | xe **cắt vạch dừng khi đèn đã đỏ** (đèn đỏ ở thời điểm bánh trước qua vạch) | xe đã qua vạch từ lúc đèn vàng; xe dừng chờ trước/trên vạch; xe rẽ phải khi được phép (nếu chắc chắn đó là luật áp dụng, ghi `notes`) |

## Các trường
- `frame_start`, `frame_end`: khoảng frame mà **vi phạm nhìn thấy rõ ràng** (không phải toàn bộ lúc xe xuất hiện). `red_light`: từ trước vạch ~0.5 s đến sau vạch ~0.5 s.
- `key_frame` và `key_box`: frame **đại diện rõ nhất** và hộp bao **cả xe** (không chỉ người/đầu) ở frame đó. Kéo chuột vẽ hộp rồi phím `k`.
  Tọa độ ở **pixel gốc của video** (độ phân giải ghi trong `width`/`height` của file nhãn); công cụ đánh giá sẽ quy đổi từ khung 1280×720 của pipeline.
- `plate_text`: biển số chuẩn hóa **không dấu gạch/chấm/khoảng trắng, chữ hoa** (`29B112345`); để trống nếu không đọc được bằng mắt. Công cụ tự chuẩn hóa.
- `difficulty`: `normal` hoặc `hard` (xe xa, bị che, mờ, đêm tối, nhãn không chắc). Báo cáo sẽ có số liệu có và không có sự kiện `hard`.
- `notes`: ghi chú tự do (trẻ em, rẽ phải, v.v.).

## Vùng được gán nhãn đầy đủ (`annotated_ranges`)
Dương tính giả chỉ được **đếm trong các vùng này** (phím `[` mở, `[` lần nữa đóng). Chỉ đánh dấu đoạn video bạn đã xem kỹ **toàn bộ** và gán **mọi** vi phạm trong đó.
Một sự kiện nằm ngoài mọi vùng sẽ bị báo lỗi khi lưu (vì không xác định được báo nhầm).
Vùng nhìn không rõ (xa, cây che): có thể khai báo `ignore_regions` thủ công trong file JSON (đa giác), pipeline sẽ không bị chấm ở đó.

## Quy tắc làm việc
1. Gán nhãn **trước** khi xem kết quả của pipeline (tránh bị ảnh hưởng). Không dùng pipeline để gợi ý nhãn cho tập test.
2. Mỗi video gán **một lần, một người**; nếu có người thứ hai, gán độc lập cho khoảng 10% rồi so (độ đồng thuận) trước khi dùng.
3. Phân loại `split` đúng ngay từ đầu (`dev` để chỉnh luật; `test` khóa). Không chuyển clip giữa hai tập sau khi đã xem kết quả.
4. Lưu thường xuyên (`w`); file được ghi nguyên tử và kiểm tra hợp lệ nên không bị hỏng giữa chừng.

## Phím tắt
`space` phát/dừng · `j`/`l` ±1 frame · `J`/`L` ±30 frame · `s` đầu sự kiện · `e` cuối sự kiện · kéo chuột + `k` = key frame và hộp ·
`t` đổi loại · `h` đánh dấu `hard` · `Enter` lưu sự kiện (hỏi biển số ở terminal) · `u` hoàn tác · `[` mở/đóng vùng đã gán đầy đủ · `w` lưu · `q` lưu và thoát.


## Định nghĩa class đầu người (nhãn cho detector, **quy tắc mới 2026-10-19**)
Áp dụng khi gán/sửa nhãn ảnh tĩnh cho detector (`helmet`, `no-helmet`), không phải khi gán sự kiện ở trên.
- **`helmet` = chỉ mũ bảo hiểm** (mũ cứng/nửa đầu/trùm đầu có vỏ cứng). Đầu đội mũ bảo hiểm → `helmet`.
- **Khăn trùm đầu, mũ vải, mũ lưỡi trai/nón (kể cả nón lá), hood, tóc trần → `no-helmet`.** Chỉ mũ bảo hiểm mới là `helmet`.
- **Box chỉ bao vùng đầu: từ đỉnh đầu (đỉnh mũ) tới cằm; không bao vai, lưng hay thân.** Nhìn từ trên xuống (camera hiện tại) thì hộp gần vuông; hộp cao gấp rõ ràng lần rộng thường là đã bao cả vai/lưng.
- Hai đầu chồng nhau (người ngồi sau thò mũ lên trên vai người trước): mỗi đầu một box riêng, phần chồng được phép; đừng nới box của người phía trước để bao cả mũ người sau.
- Không chắc (đầu bị che, quá nhỏ): không gán box hoặc ghi chú để duyệt; đừng đoán.
- **Đây là quy tắc mới và dữ liệu hiện tại chưa tuân theo hoàn toàn.** Đo trên split A (`experiments/EXP-015-helmet-grouped-3cls/box_consistency/`): box đầu thường bao cả vai (trung vị cao/rộng 1.34 helmet, 1.39 no-helmet), một phần nhỏ bao tới thắt lưng; khăn trùm đầu hiện bị gán lẫn lộn. **Chưa có quyết định gán nhãn lại**: quy tắc này là chuẩn cho nhãn mới và cho khi rà soát; việc sửa nhãn cũ cần số đo và duyệt riêng.
- **Khác với luật sự kiện ở trên:** dòng `no_helmet` của bảng sự kiện vẫn *không gán* khi mũ không rõ ràng (khăn, mũ vải…) vì ground truth sự kiện phải thận trọng; quy tắc lớp ở đây nói khăn/mũ vải là `no-helmet` ở mức *ảnh tĩnh*. Hai quy tắc có chủ đích khác nhau và **cần bạn xác nhận có muốn thống nhất** (nếu khăn trùm đầu là vi phạm thì dòng sự kiện phải sửa theo).
