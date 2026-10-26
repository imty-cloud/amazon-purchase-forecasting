# Danh sách kiểm tra bài nộp

Bản nộp là một bản `git clone` đầy đủ của repository gốc (`~/Documents/Đồ án dự kiến/amazon_purchase_forecasting/`), commit gốc `d7dbb22`, cộng
**một commit mới** chỉ thêm `README.md` (tiếng Việt, thay bản cũ đã lỗi thời), `SUBMISSION_CHECKLIST.md` và vài dòng `.gitignore`.
Không sửa file nào khác. Repository gốc không bị thay đổi.

## Đã đưa vào
- [x] `src/`, `scripts/`, `tests/` (136 test), `pyproject.toml`, `requirements.in`, `requirements.lock.txt`
- [x] `reports/` (protocol v1.2, báo cáo Stage A–D) và `outputs/` (kết quả tổng hợp, hai hình Phase 2, file khóa)
- [x] Toàn bộ lịch sử Git gốc: 11 commit, 4 tag (`stage-b`, `stage-c`, `stage-d-lock`, `stage-d-lock-v1.2`), nhánh `main`
- [x] README tiếng Việt

## Không đưa vào (có chủ ý)
- Dữ liệu gốc `amazon-purchases.csv`, `survey.csv` (người chấm tải từ Dataverse, MD5 ghi trong README)
- File hàng-từng-người đã bị `.gitignore` chặn: `outputs/*.csv.gz` (nhãn, dự đoán, bootstrap, mã người trả lời)
- `.venv`, cache, `.DS_Store`, remote `origin` (trỏ về đường dẫn máy cục bộ)
- Gói tài liệu viết báo cáo (`Amazon Purchase Forecasting/`) vì là tài liệu cho phiên viết báo cáo, không phải mã nguồn

## Đã kiểm tra bằng cách chạy
- [x] Tag và commit của bản nộp trùng với repository gốc; `git fsck --full` sạch; 11 commit ở cả hai nơi
- [x] `pytest`: 128 passed, 8 skipped (không có dữ liệu gốc) và 136 passed (có dữ liệu gốc, chỉ đọc); cây làm việc sạch sau đó
- [x] `stage_b_validate.py` và `stage_c_features.py` trên bản clone tạm: thoát mã 0; chỉ thời gian chạy, đường dẫn dữ liệu và thời gian bắt đầu khác trong hai file JSON tóm tắt. Bản clone tạm đã xóa
- [x] Quét mã nguồn và lịch sử tìm khóa, token, mật khẩu: không thấy. Tác giả commit dùng địa chỉ noreply của GitHub

## Chưa chạy (chỉ đọc code)
- [ ] Cài đặt `requirements.lock.txt` vào môi trường mới
- [ ] `stage_d_phase1.py`, `stage_d_write_lock.py`, `mutation_check.py`, `independent_*check.py`, `build_stage_*_report.py`
- [ ] **`stage_d_phase2.py` không chạy, cố ý** (đánh giá test chạy một lần)

## Hạn chế còn lại
- `check_stage_d_report.py` không chạy trọn vẹn trong bản nộp: cần ba file hàng-từng-người bị loại khỏi Git (thiếu file thì báo `FileNotFoundError`). Đã chạy thành công trên repository gốc. Không sửa checker để không làm yếu kiểm tra.
- Đường dẫn tuyệt đối của máy tác giả đã được thay bằng ký hiệu chung (`<THU_MUC_DU_AN>`, `<DUONG_DAN_DU_LIEU>`) trong các báo cáo, hai script build Stage B/C và `stage_b_validation_summary.json`; các commit cũ vẫn giữ nguyên, không viết lại lịch sử. Mã định danh người trả lời không bị lộ trong repository.
- Giữ nguyên tag và `.git`: provenance của báo cáo dùng `stage-d-lock-v1.2^{commit}`.
- Python 3.14 mới; người chấm cần cài Python 3.14. Chưa kiểm tra cài mới trên máy khác.
- Các điểm không nhất quán đã biết trong giao thức (§8 so với §11, §9) được ghi nhận, không sửa.
- Chưa đẩy lên GitHub, chưa tạo repository trực tuyến.
