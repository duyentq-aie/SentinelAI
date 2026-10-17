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
