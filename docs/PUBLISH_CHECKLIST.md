# Chuẩn bị đưa lên GitHub (CHƯA push) – danh sách cần chủ repo duyệt

Quét bằng `python tools/preflight_publish.py --history` (chỉ đọc, không xóa gì; kết quả 2026-10-18: **0 SECRET**, 197 file làm việc + toàn bộ lịch sử). Quét không phải là bảo đảm: false negative có thể có.

## Đã làm
- `LICENSE`: văn bản AGPL-3.0 chuẩn (661 dòng, sao từ bản đi kèm gói Ultralytics 8.4.26; Ultralytics là AGPL-3.0 nên dự án chọn AGPL-3.0). **Chưa có dòng bản quyền/tác giả riêng** – chủ repo thêm nếu muốn.
- Không secret/token/khóa riêng trong cây làm việc lẫn lịch sử (các giá trị giả trong `tests/test_preflight_publish.py` mang nhãn `preflight-ignore`).
- Không file lớn (> 5 MB) hay weights/video **được theo dõi** bởi git (`*.pt`, `*.onnx`, `*.mp4`... trong `.gitignore`); `venv-edge/` đã được thêm vào `.gitignore`.
- License dataset ghi là **"chưa xác minh"** (README, DATA_CARD): cả hai export Roboflow *khai báo* CC BY 4.0 trong `data.yaml` nhưng chưa kiểm độc lập; hai dataset thuộc workspace Roboflow của người khác (`trans-workspace`, `school-fuhih`), nên cần ghi công tác giả nếu phân phối lại.

## Cần bạn duyệt / quyết định
| # | vấn đề | vị trí | gợi ý (chưa làm) |
|---|---|---|---|
| 1 | Tên tài khoản Kaggle `tranquangduyenk17ct` nằm trong tài liệu và metadata thí nghiệm | `docs/AUDIT.md` (4 chỗ), `docs/PLAN.md`, `experiments/EXP-000-kaggle-original/*` (+ ~10 chỗ nữa, xem `outputs/preflight.json`) | giữ (là nguồn gốc dữ liệu, thông tin công khai trên Kaggle) hoặc thay bằng `<kaggle-user>` |
| 2 | Tên Kaggle **nằm trong chính các weights** (`helmet_best.pt`, `plate_best.pt` và 4 file `models/onnx/*.onnx`: 1 lần/file, trong metadata `data`/train args) | file cục bộ, **không được commit** | không đăng weights nguyên bản; nếu cần đăng, xuất lại state_dict/ONNX đã xóa metadata (cần bạn đồng ý, công cụ chưa viết) – và xem lại license dataset trước khi đăng weights |
| 3 | Email cá nhân (`<gmail>`) và tên tác giả cũ (`<author>`) trong metadata commit | mọi commit | đã thay bằng email noreply của GitHub; `tools/rewrite_author.py` (dry-run mặc định, backup bundle ngoài repo) |
| 4 | Đường dẫn cá nhân: thư mục home (`<HOME>`) và đường dẫn dự án (`<REPO>`) | `experiments/EXP-009|010|012/**/env.json` (do `sysprobe` ghi) | ghi `<HOME>` thay cho thư mục home (benchmark_pipeline làm tự động); dữ liệu thử của scanner được dựng lúc chạy, không ghi nguyên văn |
| 5 | AGPL-3.0 ép phát hành mã nguồn cho dịch vụ mạng dùng code này | `LICENSE` | xác nhận chấp nhận (README đã ghi) |
| 6 | Video mẫu `test_video.mp4`, ảnh bằng chứng, biển số thật | gitignored | giữ ngoài git; nếu đăng ví dụ phải làm mờ biển/mặt |
| 7 | Dữ liệu gốc helmet/plate (`data/raw/`) | gitignored | không đăng khi license chưa xác minh |
| 8 | `.gitattributes`/kết thúc dòng: cảnh báo LF→CRLF khi commit trên Windows | toàn repo | tùy chọn: thêm `.gitattributes` `* text=auto` |

Không có lệnh push nào đã được chạy; remote chưa được cấu hình hay dùng.

## Cập nhật 2026-10-18 (sau khi bạn duyệt)
- **Đường dẫn home:** đã thay bằng `<HOME>` trong mọi `env.json` đang được theo dõi; `benchmark_pipeline.py` tự thay từ nay. Còn lại 2 chỗ trong `tests/test_preflight_publish.py` là dữ liệu thử cố ý.
- **Email tác giả:** `tools/rewrite_author.py --email <USER>@users.noreply.github.com [--name ...]` (dry-run mặc định; `--yes` mới viết lại; tự tạo bundle sao lưu lịch sử cũ ngoài repo; từ chối chuỗi giữ chỗ `<USER>`). **CHƯA chạy** vì cần tên GitHub thật của bạn. Nó đổi toàn bộ hash commit: chạy một lần, trước khi push. Cần xóa luôn commit cũ trong reflog/bundle nếu muốn email cũ biến mất hẳn khỏi máy.
- Vẫn chưa push.

## Kết quả lần chạy 2026-10-19 (sau khi rewrite)
- Metadata: 41 commit, một danh tính duy nhất (email noreply của GitHub); không còn email cũ hay tên cũ trong tên, email, commit message.
- Nội dung mọi revision: `<gmail>`, `<HOME>`, `<REPO>`, `<author>` thay cho email cá nhân, thư mục home, đường dẫn dự án và tên tác giả cũ; grep bốn loại chuỗi đó (email Gmail, đường dẫn home, đường dẫn dự án, tên tác giả cũ) trên mọi revision = 0 kết quả.
- Backup (ngoài repo, chứa dữ liệu cũ, không đăng): `SentinelAI.before-author-rewrite.bundle` (trước rewrite danh tính), `SentinelAI.before-content-rewrite.bundle` (trước rewrite nội dung). Reflog cục bộ vẫn giữ commit cũ (không gc).
- Giữ nguyên theo quyết định chủ repo: tên tài khoản Kaggle (nguồn gốc dữ liệu). Weights/ONNX cục bộ có tên này nhưng không được track.
- Chưa cấu hình remote, chưa push.
