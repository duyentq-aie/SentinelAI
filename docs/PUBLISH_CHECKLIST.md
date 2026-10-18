# Chuẩn bị đưa lên GitHub (CHƯA push) – danh sách cần chủ repo duyệt

Quét bằng `python tools/preflight_publish.py --history` (chỉ đọc, không xóa gì; kết quả 2026-10-18: **0 SECRET**, 197 file làm việc + toàn bộ lịch sử). Quét không phải là bảo đảm: false negative có thể có.

## Đã làm
- `LICENSE`: văn bản AGPL-3.0 chuẩn (661 dòng, sao từ bản đi kèm gói Ultralytics 8.4.26; Ultralytics là AGPL-3.0 nên dự án chọn AGPL-3.0). **Chưa có dòng bản quyền/tác giả riêng** – chủ repo thêm nếu muốn.
- Không secret/token/khóa riêng trong cây làm việc lẫn lịch sử (các giá trị giả trong `tests/test_preflight_publish.py` mang nhãn `preflight-ignore`).
- Không file lớn (> 5 MB) hay weights/video **được theo dõi** bởi git (`*.pt`, `*.onnx`, `*.mp4`... trong `.gitignore`); `venv-edge/` đã được thêm vào `.gitignore`.
- License dataset ghi là **"chưa xác minh"** (README, AUDIT): hai dataset Roboflow/Kaggle chưa kiểm được điều khoản.

## Cần bạn duyệt / quyết định
| # | vấn đề | vị trí | gợi ý (chưa làm) |
|---|---|---|---|
| 1 | Tên tài khoản Kaggle `tranquangduyenk17ct` nằm trong tài liệu và metadata thí nghiệm | `docs/AUDIT.md` (4 chỗ), `docs/PLAN.md`, `experiments/EXP-000-kaggle-original/*` (+ ~10 chỗ nữa, xem `outputs/preflight.json`) | giữ (là nguồn gốc dữ liệu, thông tin công khai trên Kaggle) hoặc thay bằng `<kaggle-user>` |
| 2 | Tên Kaggle **nằm trong chính các weights** (`helmet_best.pt`, `plate_best.pt` và 4 file `models/onnx/*.onnx`: 1 lần/file, trong metadata `data`/train args) | file cục bộ, **không được commit** | không đăng weights nguyên bản; nếu cần đăng, xuất lại state_dict/ONNX đã xóa metadata (cần bạn đồng ý, công cụ chưa viết) – và xem lại license dataset trước khi đăng weights |
| 3 | Email trong lịch sử commit: `<gmail>` (tác giả `<author>`) | mọi commit | dùng email `...@users.noreply.github.com` cho các commit mới; viết lại lịch sử chỉ khi bạn yêu cầu |
| 4 | Đường dẫn cá nhân `<HOME>`, `<REPO>` | `experiments/EXP-009|010/**/env.json` (do `sysprobe` ghi), `tests/test_preflight_publish.py` (dữ liệu thử) | thay bằng `~`/`<repo>` trong env.json trước khi push, hoặc chấp nhận |
| 5 | AGPL-3.0 ép phát hành mã nguồn cho dịch vụ mạng dùng code này | `LICENSE` | xác nhận chấp nhận (README đã ghi) |
| 6 | Video mẫu `test_video.mp4`, ảnh bằng chứng, biển số thật | gitignored | giữ ngoài git; nếu đăng ví dụ phải làm mờ biển/mặt |
| 7 | Dữ liệu gốc helmet/plate (`data/raw/`) | gitignored | không đăng khi license chưa xác minh |
| 8 | `.gitattributes`/kết thúc dòng: cảnh báo LF→CRLF khi commit trên Windows | toàn repo | tùy chọn: thêm `.gitattributes` `* text=auto` |

Không có lệnh push nào đã được chạy; remote chưa được cấu hình hay dùng.
