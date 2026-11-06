# Dự báo hoạt động mua hàng trong tương lai từ dữ liệu giao dịch thương mại điện tử

*So sánh các mô hình học máy và đánh giá độ ổn định theo thời gian ngắn hạn*

## 1. Mục tiêu và câu hỏi nghiên cứu

**Câu hỏi chính.** Lịch sử giao dịch có sẵn trước một mốc thời gian (cutoff) có dự báo được việc một người trả lời khảo sát
có ghi nhận ít nhất một ngày mua hàng trong 90 ngày tiếp theo hay không?

Các lưu ý bắt buộc khi đọc kết quả:
- `Survey ResponseID` là mã người trả lời khảo sát, **không** phải mã khách hàng Amazon đã được xác minh. Tài khoản có thể dùng chung.
- Nhãn y = 0 nghĩa là *không ghi nhận ngày mua nào trong dữ liệu trong 90 ngày sau cutoff*. Đây **không** phải churn đã được xác nhận.
- Chỉ có hai quý test liền kề, nên không thể kết luận độ ổn định dài hạn.
- Các so sánh được chuẩn bị trước (P1–P3) không cho phép kết luận mô hình nào vượt trội khi khoảng bất định chứa 0; "không có khác biệt rõ ràng" không có nghĩa là "tương đương".
- Mối liên hệ dự báo không phải quan hệ nhân quả.

Kết quả đầy đủ: `reports/stage_d_results.md`. Giao thức đã khóa: `reports/stage_d_protocol.md` (v1.2).

## 2. Cấu trúc repository

| Đường dẫn | Vai trò |
|---|---|
| `src/apf/config.py` | Hằng số: đường dẫn dữ liệu (`APF_DATA_DIR`), MD5 chuẩn, 15 cutoff, chân trời 90 ngày |
| `src/apf/data.py`, `target.py`, `features.py` | Đọc dữ liệu, tạo nhãn y, 19 đặc trưng tại thời điểm cutoff |
| `src/apf/splits.py`, `models.py`, `evaluation.py` | Chia train/tune/test và hold-out theo người trả lời, mô hình, chỉ số và bootstrap |
| `src/apf/lock.py`, `phase2.py` | Cơ chế khóa cấu hình và đánh giá test chạy một lần |
| `scripts/stage_b_validate.py` | Giai đoạn B: kiểm tra dữ liệu và nhãn |
| `scripts/stage_c_features.py` | Giai đoạn C: đặc trưng, kiểm tra rò rỉ, thiết kế chia dữ liệu |
| `scripts/stage_d_phase1.py` | Giai đoạn D, Phase 1: chọn mô hình (chỉ dùng hàng trước cutoff test đầu tiên) |
| `scripts/stage_d_write_lock.py` | Ghi file khóa cấu hình |
| `scripts/stage_d_phase2.py` | Giai đoạn D, Phase 2: đánh giá test, một lần |
| `scripts/build_stage_*_report.py`, `check_stage_d_report.py` | Sinh báo cáo từ `outputs/` và kiểm tra từng con số |
| `scripts/independent_check.py`, `independent_feature_check.py`, `mutation_check.py` | Kiểm tra độc lập và kiểm tra đột biến |
| `tests/` | 136 test (8 test cần file dữ liệu gốc) |
| `reports/`, `outputs/` | Báo cáo và kết quả tổng hợp đã commit |
| `pyproject.toml`, `requirements.in`, `requirements.lock.txt` | Cấu hình và phiên bản thư viện |

## 3. Môi trường

- Python **3.14** (`requires-python = ">=3.14"`; đã chạy với 3.14.7 trên macOS arm64).
- Thư viện cố định trong `requirements.lock.txt` (numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, LightGBM 4.7.0, ...).

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

## 4. Dữ liệu
Dữ liệu **không** nằm trong repository này. Bộ dữ liệu công khai: [Open e-commerce 1.0](https://doi.org/10.7910/DVN/YGLYDY) (Berke và cs., 2024, *Scientific Data* 11, 491, [doi:10.1038/s41597-024-03329-6](https://doi.org/10.1038/s41597-024-03329-6); Harvard Dataverse [doi:10.7910/DVN/YGLYDY](https://doi.org/10.7910/DVN/YGLYDY), phiên bản 1.0). Tải hai file `amazon-purchases.csv` và `survey.csv` về cùng một thư mục (giấy phép và điều khoản theo trang Dataverse), rồi trỏ tới thư mục đó:

## 5. Thứ tự chạy

Repository đã chứa kết quả của tất cả các bước. **Không chạy lại ở thư mục này**, vì các bước B, C và Phase 1 ghi đè file đã commit. Nếu muốn
chạy lại, hãy làm trên một bản clone tạm và so sánh bằng `git diff`.

| Bước | Lệnh | Ghi vào |
|---|---|---|
| 1 | `.venv/bin/python -m pytest` | không ghi gì |
| 2 | `.venv/bin/python scripts/stage_b_validate.py` | `outputs/stage_b_*`, file hàng-từng-người `*.csv.gz` (bị git bỏ qua) |
| 3 | `.venv/bin/python scripts/stage_c_features.py` | `outputs/stage_c_*`, `model_table_*.csv.gz` |
| 4 | `.venv/bin/python scripts/stage_d_phase1.py [--out-dir DIR]` | chọn mô hình; cũng ghi `reports/stage_d_phase1_results.md` |
| 5 | `.venv/bin/python scripts/stage_d_write_lock.py` | khóa cấu hình; từ chối nếu đã có khóa trừ khi có `--supersede "lý do"` |
| 6 | `.venv/bin/python scripts/stage_d_phase2.py` | **không chạy**; xem mục 7 |
| 7 | `scripts/build_stage_d_report.py`, rồi `scripts/check_stage_d_report.py` | báo cáo kết quả và kiểm tra |

## 6. Chạy test

```bash
.venv/bin/python -m pytest                       # 136 test nếu có dữ liệu gốc
.venv/bin/python -m pytest -m "not realdata"     # bỏ các test cần dữ liệu gốc
```

## 7. Yêu cầu Git và cơ chế chạy test một lần

- `stage_d_write_lock.py` và `stage_d_phase2.py` gọi `git status` trong thư mục dự án (`src/apf/lock.py`), nên phải chạy trong một **repository Git**. Giữ thư mục `.git`.
- Phase 2 (`src/apf/phase2.py`) chỉ chạy khi: (a) có file khóa; (b) SHA-256 của `reports/stage_d_protocol.md` khớp file khóa; (c) `src/`, `scripts/`, giao thức và file khóa đã commit và không bị sửa; (d) file selection Phase 1 khớp checksum trong khóa; (e) `outputs/stage_d_test_metrics.csv` **chưa tồn tại**.
- Điều kiện (e) nghĩa là Phase 2 đã chạy đúng một lần và sẽ từ chối chạy lại. **Đừng xóa file kết quả hay sửa code để vượt qua chốt này**: làm vậy làm mất giá trị của đánh giá test.
- Các tag ghi nhận mốc kiểm toán: `stage-b`, `stage-c`, `stage-d-lock`, `stage-d-lock-v1.2` (khóa dùng cho Phase 2 nằm ở commit `8d904e6`; kết quả ở commit `099ab37`). Code không đọc tag khi chạy, nhưng `build_stage_d_report.py` và file provenance dùng `stage-d-lock-v1.2^{commit}`. Hãy giữ nguyên tag và lịch sử.
- Ba file hàng-từng-người của Phase 2 (`stage_d_predictions.csv.gz`, hai file bootstrap) bị git bỏ qua vì chứa mã người trả lời, nên **không có** trong repository. Vì vậy `check_stage_d_report.py` không chạy trọn vẹn ở đây (xem mục 9).

## 8. Đã chạy thực tế và chỉ kiểm tra tĩnh

**Đã chạy thực tế (trong lúc chuẩn bị bài nộp):**
- `pytest` không có dữ liệu gốc: 128 passed, 8 skipped. `pytest` có dữ liệu gốc (chỉ đọc): 136 passed.
- `stage_b_validate.py` và `stage_c_features.py` trên một bản clone tạm với dữ liệu gốc: cả hai thoát với mã 0; mọi file kết quả giống hệt bản đã commit, trừ hai file JSON tóm tắt khác ở thời điểm chạy, đường dẫn dữ liệu và thời gian chạy.
- Cấu trúc Git của bản nộp: 11 commit, 4 tag, `git fsck` sạch, trùng với repository gốc.

**Chỉ kiểm tra tĩnh (đọc code, chưa chạy):** `pip install -r requirements.lock.txt` trên môi trường mới, `stage_d_phase1.py`, `stage_d_write_lock.py`, `stage_d_phase2.py`, `mutation_check.py`, `independent_check.py`, `independent_feature_check.py`, `build_stage_*_report.py`. Phase 2 cố ý không chạy lại.

## 9. Hạn chế và xử lý sự cố

- `check_stage_d_report.py` kết thúc bằng `FileNotFoundError` nếu thiếu `outputs/stage_d_predictions.csv.gz` (file provenance ghi SHA-256 của ba file hàng-từng-người). Đây là hạn chế đã biết của bản nộp; việc kiểm tra này đã được chạy thành công trên repository gốc.
- Các báo cáo cũ và vài script build có chứa đường dẫn tuyệt đối của máy tác giả. Lịch sử Git không bị viết lại để giữ tính toàn vẹn.
- `stage_d_protocol.md` §11 (mục D-Q2) nêu một so sánh chính duy nhất, trong khi §8 nêu ba (P1–P3); giao thức §9 dự kiến mô hình dự báo vượt mức y = 0, nhưng kết quả cho thấy dự báo thấp hơn thực tế. Hai điểm này được ghi lại, không sửa.
- Xác suất của các mô hình không được hiệu chỉnh tốt tại các cutoff test (dự báo thấp y = 0).
- `ModuleNotFoundError: apf`: chạy từ thư mục gốc repository, hoặc dùng `python -m pytest` (đã đặt `pythonpath = ["src"]`).
- `LockError ... uncommitted changes in guarded paths`: bạn đã sửa `src/`, `scripts/`, giao thức hoặc file khóa. Dùng `git status` và `git restore`.
- Test báo skip "raw CSV not available": chưa đặt đúng `APF_DATA_DIR`.
