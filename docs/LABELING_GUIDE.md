# Hướng dẫn gán nhãn sự kiện vi phạm

Công cụ: `python tools/label_events.py --video <clip> --split dev|test` → `eval_sets/events/<video_id>.json`.
Schema và các trường: `docs/EVIDENCE.md` (phần "GT label"); kiểm tra tự động bằng `sentinel.labeling.validate` mỗi khi lưu.

## Một sự kiện là gì
Một **loại vi phạm** của **một xe**. Xe vừa không đội mũ vừa chở 3 người = **2 sự kiện** (`no_helmet` và `overload`), mỗi cái có khoảng frame riêng.

| loại | gán nhãn khi | KHÔNG gán khi |
|---|---|---|
| `no_helmet` | **bất kỳ** người nào trên xe máy mà **nhìn rõ đầu và đó không phải mũ bảo hiểm** (≥ 5 frame): đầu trần, khăn trùm, mũ vải, mũ lưỡi trai, nón lá, hood đều là vi phạm | (ngoại lệ) **không nhìn rõ** đầu (mờ, bị che, quá xa): **vẫn tạo sự kiện nhưng đặt `ambiguous: true`** (phím `a`) – sự kiện không bị chấm điểm (xem mục `ambiguous` ở dưới); người đi bộ, xe đạp: không gán. `difficulty: hard` chỉ dành cho ca nhìn rõ nhưng khó (xa, đêm, một phần bị che) |
| `overload` | xe máy chở **> 2 người** (tính cả trẻ em; ghi `notes: "tre em"` vì luật có ngoại lệ) | đúng 2 người; người ngồi xe khác lọt vào khung |
| `red_light` | xe **cắt vạch dừng khi đèn đã đỏ** (đèn đỏ ở thời điểm bánh trước qua vạch) | xe đã qua vạch từ lúc đèn vàng; xe dừng chờ trước/trên vạch; xe rẽ phải khi được phép (nếu chắc chắn đó là luật áp dụng, ghi `notes`) |

## Các trường
- `frame_start`, `frame_end`: khoảng frame mà **vi phạm nhìn thấy rõ ràng** (không phải toàn bộ lúc xe xuất hiện). `red_light`: từ trước vạch ~0.5 s đến sau vạch ~0.5 s.
- `key_frame` và `key_box`: frame **đại diện rõ nhất** và hộp bao **cả xe** (không chỉ người/đầu) ở frame đó. Kéo chuột vẽ hộp rồi phím `k`.
  Tọa độ ở **pixel gốc của video** (độ phân giải ghi trong `width`/`height` của file nhãn); công cụ đánh giá sẽ quy đổi từ khung 1280×720 của pipeline.
- `plate_text`: biển số chuẩn hóa **không dấu gạch/chấm/khoảng trắng, chữ hoa** (`29B112345`); để trống nếu không đọc được bằng mắt. Công cụ tự chuẩn hóa.
- `difficulty`: `normal` hoặc `hard` (xe xa, bị che, mờ, đêm tối, nhãn không chắc). Báo cáo sẽ có số liệu có và không có sự kiện `hard`.
- `ambiguous`: `true`/`false` (mặc định `false`, cấp **sự kiện**). Đặt `true` khi **không thể biết** vi phạm có xảy ra không (đầu mờ, bị che, quá xa). Sự kiện vẫn có `frame_start`/`frame_end`/`key_frame`/`key_box` như bình thường để dự đoán của pipeline còn ghép được với nó; bộ đánh giá xử lý như cờ `difficult` của VOC / `iscrowd` của COCO: dự đoán trùng sự kiện ambiguous **không tính TP cũng không tính FP**, ambiguous không được dự đoán **không tính FN**; báo cáo ghi `n_ambiguous_matched / n_ambiguous_total`. Khác `difficulty: hard` (hard vẫn được chấm trừ khi dùng `--exclude-hard`). Đừng lạm dụng: ambiguous là "không biết", không phải "khó".
- `notes`: ghi chú tự do (trẻ em, rẽ phải, v.v.).

## Vùng được gán nhãn đầy đủ (`annotated_ranges`)
Dương tính giả chỉ được **đếm trong các vùng này** (phím `[` mở, `[` lần nữa đóng). Chỉ đánh dấu đoạn video bạn đã xem kỹ **toàn bộ** và gán **mọi** vi phạm trong đó.
Một sự kiện nằm ngoài mọi vùng sẽ bị báo lỗi khi lưu (vì không xác định được báo nhầm).
`ignore_regions` (đa giác khai báo thủ công trong file JSON) **chỉ dành cho vùng cố định của khung hình** luôn nhìn không rõ (cây che cả đoạn đường, đoạn quá xa): pipeline không bị chấm FP ở đó. Một xe/người cụ thể nhìn không rõ thì dùng `ambiguous: true` ở sự kiện, không dùng `ignore_regions`.

## Quy tắc làm việc
1. Gán nhãn **trước** khi xem kết quả của pipeline (tránh bị ảnh hưởng). Không dùng pipeline để gợi ý nhãn cho tập test.
2. Mỗi video gán **một lần, một người**; nếu có người thứ hai, gán độc lập cho khoảng 10% rồi so (độ đồng thuận) trước khi dùng.
3. Phân loại `split` đúng ngay từ đầu (`dev` để chỉnh luật; `test` khóa). Không chuyển clip giữa hai tập sau khi đã xem kết quả.
4. Lưu thường xuyên (`w`); file được ghi nguyên tử và kiểm tra hợp lệ nên không bị hỏng giữa chừng.

## Phím tắt
`space` phát/dừng · `j`/`l` ±1 frame · `J`/`L` ±30 frame · `s` đầu sự kiện · `e` cuối sự kiện · kéo chuột + `k` = key frame và hộp ·
`t` đổi loại · `h` đánh dấu `hard` · `a` đánh dấu `ambiguous` (cho sự kiện kế tiếp; HUD hiện `next ambiguous`, sự kiện ambiguous vẽ màu cam) · `Enter` lưu sự kiện (hỏi biển số ở terminal) · `u` hoàn tác · `[` mở/đóng vùng đã gán đầy đủ · `w` lưu · `q` lưu và thoát.


## Định nghĩa class đầu người (nhãn cho detector, **quyết định 2026-10-19**, sau khi đo dữ liệu hiện có)
Áp dụng khi gán/rà soát nhãn ảnh tĩnh cho detector (`helmet`, `no-helmet`); cùng một định nghĩa "đội mũ bảo hiểm hay không" với bảng sự kiện ở trên.
- **`helmet` = chỉ mũ bảo hiểm.** Khăn, mũ vải, mũ lưỡi trai, nón lá, hood, đầu trần → **`no-helmet`**.
- **Box đầu = vùng đầu; cho phép bao tới vai.** Lý do: đó đúng là cách dữ liệu hiện có được gán. Đo trên split A (`experiments/EXP-015-helmet-grouped-3cls/box_consistency/`): chiều cao box đầu / chiều cao xe máy có **trung vị 0.22–0.23** ở cả hai lớp, mọi split (helmet 0.234 / 0.224 / 0.230, no-helmet 0.223 / 0.219 / 0.220), tức khoảng một phần tư xe, lớn hơn chỉ cái đầu.
- **Box bao tới thắt lưng hoặc ba lô là sai** (đuôi của phân phối: chiều cao tương đối > ≈ 0.30 hoặc cao/rộng > ≈ 1.5 là ngưỡng *rà soát*, không phải luật cứng; ~18% box đầu vượt ngưỡng). Với nhãn mới: giữ box trong "đầu + vai". **Không sửa ngược dữ liệu cũ** (quyết định "không gán lại", xem README EXP-015).
- Hai đầu chồng nhau: mỗi đầu một box riêng, phần chồng được phép; đừng nới box người phía trước để bao mũ người sau.
- **Hai trường hợp cho `no_helmet` (thống nhất với bảng sự kiện):**
  (a) *nhìn rõ không phải mũ bảo hiểm* (khăn, mũ vải, nón lá, đầu trần) → `no-helmet` và **là vi phạm** (sự kiện `no_helmet`, `ambiguous: false`);
  (b) *không nhìn rõ* (mờ, bị che, quá xa) → sự kiện với **`ambiguous: true`** (cùng track/khoảng frame/hộp như sự kiện thường, không bị chấm). `ignore_regions` chỉ dùng cho vùng cố định, không dùng cho từng xe.
  Cờ `ambiguous` thuộc **nhãn sự kiện (video)**. Nhãn ảnh tĩnh cho detector (file YOLO `.txt`) không có trường này: với ảnh tĩnh, đầu không nhìn rõ thì không vẽ box (chưa có cách đánh dấu riêng).
- Dữ liệu hiện tại **chưa tuân theo hoàn toàn** (khăn trùm đầu bị gán lẫn lộn; một phần box bao thân): xem số đo trong EXP-015.
