# Phân tích bổ sung hậu nghiệm (post-hoc / exploratory)

> **POST-HOC, không dùng để đổi kết luận P1–P3.** Giao thức khóa `stage-d-lock-v1.2` và mọi kết quả khóa giữ nguyên. Không file nào trong `src/`, `outputs/` (gốc), `reports/`, `tests/` bị sửa; không commit, không push, không đụng tag.

## Bước 0 – Kiểm tra tái lập (đạt)

- MD5 `amazon-purchases.csv` = `3549d72735b1211ab009ff0177b94bd9`, `survey.csv` = `8340a44a5691c3763702683155cb5cc7` (khớp bản chính thức và `data_md5` trong cấu hình khóa).
- `pytest`: 136 passed (venv riêng ngoài dự án).
- `src/`, `tests/`, `stage_d_locked_config.json`, `stage_d_test_metrics.csv` trong "Code nop bai" giống hệt bản git `amazon-purchase-forecasting-submission` (tag `stage-d-lock-v1.2` → `dc88321…`); thư mục "Code nop bai" không phải repo git. Bản submission chỉ khác ở hai báo cáo `stage_b/c_*_validation.md` (không có trong bản git, và thư mục đó cũng đang báo xóa 3 file — chỉ đọc, không đụng).
- Dựng lại đặc trưng + Refit (B0, B1, B2, M1, M2, M3, S1, S2) và so AP/ROC-AUC/Brier pooled trên 2 tập kiểm định: **48/48 chỉ số khớp đến 4 chữ số; chênh lệch lớn nhất 1,1e-16**. (`post_step0_check.csv`)

## A. Macro-AP (không huấn luyện lại)

Trung bình không trọng số AP của hai mốc 2022-03-31 và 2022-06-30, cạnh AP gộp.

| Tập | Mô hình | AP 03-31 | AP 06-30 | Macro-AP | AP gộp | Macro − gộp |
|---|---|---|---|---|---|---|
| temporal | B2 | 0,5860 | 0,5206 | 0,5533 | 0,5502 | +0,0031 |
| temporal | M1 | 0,5907 | 0,5412 | 0,5660 | 0,5649 | +0,0011 |
| temporal | M2 | 0,6310 | 0,5735 | 0,6023 | 0,5966 | +0,0056 |
| temporal | M3 | 0,6350 | 0,5784 | 0,6067 | 0,6045 | +0,0021 |
| holdout | B2 | 0,6485 | 0,5236 | 0,5861 | 0,5743 | +0,0117 |
| holdout | M1 | 0,6316 | 0,5487 | 0,5902 | 0,5812 | +0,0090 |
| holdout | M2 | 0,6668 | 0,5743 | 0,6205 | 0,6110 | +0,0096 |
| holdout | M3 | 0,6544 | 0,5885 | 0,6215 | 0,6157 | +0,0058 |

Nhận xét: macro-AP luôn cao hơn AP gộp một chút (≤ 0,012); mọi mô hình đều thấp hơn ở mốc 06-30 so với 03-31. Thứ hạng mô hình không đổi (B2 < M1 < M2 < M3).

## B. Precision/recall top-k, bootstrap theo người tham gia (B=1000, seed 20261024, percentile 95%)

Mô hình Refit đã khớp; top-k theo từng mốc rồi gộp, ties chia theo kỳ vọng (`top_fraction_by_group`); điểm ước lượng được assert trùng bảng khóa.

| Tập | Mô hình | Prec@10% | Recall@10% | Prec@20% | Recall@20% |
|---|---|---|---|---|---|
| temporal | B2 | 0,587 [0,548; 0,631] | 0,462 [0,438; 0,493] | 0,440 [0,411; 0,471] | 0,693 [0,668; 0,724] |
| temporal | M1 | 0,597 [0,551; 0,640] | 0,470 [0,442; 0,497] | 0,471 [0,440; 0,499] | 0,741 [0,712; 0,769] |
| temporal | M2 | 0,613 [0,576; 0,648] | 0,483 [0,454; 0,514] | 0,489 [0,459; 0,520] | 0,771 [0,743; 0,798] |
| temporal | M3 | 0,626 [0,589; 0,662] | 0,493 [0,464; 0,523] | 0,494 [0,463; 0,523] | 0,778 [0,748; 0,804] |
| holdout | B2 | 0,612 [0,530; 0,691] | 0,497 [0,439; 0,558] | 0,440 [0,377; 0,503] | 0,710 [0,652; 0,767] |
| holdout | M1 | 0,599 [0,535; 0,683] | 0,486 [0,441; 0,553] | 0,460 [0,403; 0,520] | 0,743 [0,689; 0,806] |
| holdout | M2 | 0,629 [0,550; 0,708] | 0,510 [0,460; 0,578] | 0,465 [0,406; 0,525] | 0,751 [0,697; 0,805] |
| holdout | M3 | 0,639 [0,555; 0,708] | 0,518 [0,462; 0,577] | 0,475 [0,418; 0,537] | 0,767 [0,711; 0,826] |

Hiệu số ghép cặp, top-20%:

| Tập | So sánh | Precision | Recall |
|---|---|---|---|
| temporal | M2 − B2 | +0,049 [0,035; 0,062] (loại 0) | +0,078 [0,055; 0,097] (loại 0) |
| temporal | M3 − M2 | +0,004 [−0,004; 0,012] | +0,007 [−0,006; 0,018] |
| holdout | M2 − B2 | +0,026 [−0,005; 0,053] | +0,041 [−0,009; 0,088] |
| holdout | M3 − M2 | +0,010 [−0,005; 0,027] | +0,016 [−0,008; 0,047] |

Khoảng 95% chưa hiệu chỉnh đa so sánh.

## C. Bỏ biến tích lũy – CHỈ Train → Tune

Bỏ `frequency_days, monetary_total, categories_total, tenure_days` (danh sách cố định trước). Siêu tham số khóa; LightGBM dùng cố định 114 cây (không early stopping). Không chạm tập kiểm định. Điểm 19 đặc trưng khớp `stage_d_tune_results.csv` (M2 0,5977; M3 0,5953).

| Mô hình | Cấu hình | AP | ROC-AUC | Brier | Hệ số chặn HC | Độ dốc | TB dự báo / quan sát |
|---|---|---|---|---|---|---|---|
| M2 | 19 đặc trưng | 0,5977 | 0,9209 | 0,0675 | 0,211 | 1,081 | 0,1050 / 0,1189 (0,883) |
| M2 | bỏ 4 (15) | 0,5925 | 0,9198 | 0,0674 | 0,060 | 1,074 | 0,1148 / 0,1189 (0,966) |
| M3 | 19 đặc trưng | 0,5953 | 0,9196 | 0,0676 | 0,108 | 1,175 | 0,1112 / 0,1189 (0,935) |
| M3 | bỏ 4 (15) | 0,5856 | 0,9188 | 0,0678 | 0,021 | 1,163 | 0,1174 / 0,1189 (0,987) |

Bỏ 4 biến làm AP giảm nhẹ (M2 −0,005; M3 −0,010), ROC-AUC gần như không đổi, Brier không đổi, còn hiệu chuẩn tổng thể (hệ số chặn) tốt hơn. Lưu ý: 114 cây được chọn trên Tune với đủ 19 đặc trưng, nên cấu hình đủ có lợi thế nhỏ ở M3.

## D. Hiệu chuẩn lại (hệ số chặn, Platt)

Mô hình huấn luyện trên Train; bộ hiệu chuẩn khớp trên Tune; áp lên dự báo của chính mô hình đó ở hai tập kiểm định. Tham số: M2 hệ số chặn a = 0,211, Platt (a, b) = (0,313; 1,081); M3 a = 0,108, Platt = (0,310; 1,175). AP và ROC-AUC được assert không đổi (sai lệch < 1e-9).

| Tập | Mô hình | Biến thể | Brier | Hệ số chặn HC | Độ dốc | TB dự báo / quan sát |
|---|---|---|---|---|---|---|
| temporal | M2 | không | 0,0723 | 0,324 | 1,067 | 0,1051 / 0,1270 |
| temporal | M2 | chỉ hệ số chặn | 0,0710 | 0,114 | 1,067 | 0,1191 / 0,1270 |
| temporal | M2 | Platt | 0,0710 | 0,120 | 0,986 | 0,1190 / 0,1270 |
| temporal | M3 | không | 0,0715 | 0,235 | 1,188 | 0,1099 / 0,1270 |
| temporal | M3 | chỉ hệ số chặn | 0,0708 | 0,128 | 1,188 | 0,1176 / 0,1270 |
| temporal | M3 | Platt | 0,0704 | 0,145 | 1,012 | 0,1172 / 0,1270 |
| holdout | M2 | không | 0,0704 | 0,341 | 1,065 | 0,1019 / 0,1243 |
| holdout | M2 | chỉ hệ số chặn | 0,0692 | 0,130 | 1,065 | 0,1154 / 0,1243 |
| holdout | M2 | Platt | 0,0693 | 0,138 | 0,985 | 0,1153 / 0,1243 |
| holdout | M3 | không | 0,0695 | 0,263 | 1,200 | 0,1057 / 0,1243 |
| holdout | M3 | chỉ hệ số chặn | 0,0687 | 0,155 | 1,199 | 0,1131 / 0,1243 |
| holdout | M3 | Platt | 0,0683 | 0,181 | 1,021 | 0,1125 / 0,1243 |

Mô hình dự báo thấp hơn tỷ lệ thực (trôi theo thời gian từ giai đoạn Train/Tune sang 2022); hiệu chuẩn lại chỉ giải quyết được một phần (hệ số chặn còn 0,11–0,18) và giảm Brier ≤ 0,0013. Platt đưa độ dốc về ≈ 1 ở M3. Biểu đồ độ tin cậy theo decile: `post_d_reliability.png`, số liệu `post_d_reliability_deciles.csv`.

## E. Độ nhạy Monte Carlo (tùy chọn): P1–P3 với B=10000, seed 20270101, Bonferroni 98,33%

Điểm ước lượng không đổi (cùng mô hình, cùng dữ liệu); chỉ khác các lượt bootstrap (số lượt và hạt giống). Cận Bonferroni dưới (B=1000 → B=10000), AP:

| Tập | So sánh | Cận dưới 1000 | Cận dưới 10000 | Loại 0? |
|---|---|---|---|---|
| temporal | P1 | 0,0140 | 0,0134 | có → có |
| temporal | P2 | −0,0043 | −0,0039 | không → không |
| temporal | P3 | 0,0280 | 0,0281 | có → có |
| holdout | P1 | −0,0053 | −0,0034 | không → không |
| holdout | P2 | −0,0186 | −0,0179 | không → không |
| holdout | P3 | −0,0032 | −0,0033 | không → không |

**Không cận nào (dưới hay trên) đổi dấu và không kết luận nào đổi, trên cả 18 tổ hợp tập × so sánh × chỉ số** (AP, ROC-AUC, Brier; xem `post_e_mc_sensitivity.csv`). Cận gần 0 nhất là cận trên Brier P2 (temporal): 0,000029 → 0,000045, vẫn dương ở cả hai.

## Hạn chế

Đây là phân tích hậu nghiệm: được thiết kế sau khi đã thấy kết quả kiểm định, nên chỉ mang tính thăm dò, không điều chỉnh đa so sánh ngoài E, và không thay thế P1–P3 khóa. B và D dùng lại hai tập kiểm định vốn đã chấm một lần (E cũng vậy), vì thế không còn là bằng chứng độc lập. D khớp bộ hiệu chuẩn trên Tune rồi áp ở 2022, nên bị ảnh hưởng bởi trôi phân phối theo thời gian. C chỉ dùng Tune (7 915 dòng, hai mốc) và M3 dùng số cây chọn từ cấu hình đủ biến. Bootstrap theo người tham gia chỉ phản ánh biến thiên lấy mẫu, không phản ánh sai lệch phân phối. A, B so sánh chủ yếu trên cùng mô hình nên các khoảng tin cậy ở B không hiệu chỉnh đa so sánh.

## Lệnh chạy lại

```bash
cd "<Code nop bai>"
export APF_DATA_DIR="/Users/quynhtien/Documents/Mô hình dự báo kinh doanh/du_lieu_tho"
export APF_POST_CACHE=/tmp   # nơi lưu cache bảng đặc trưng (ngoài dự án); PYTHONDONTWRITEBYTECODE=1
PY=<venv>/bin/python         # venv từ requirements.lock.txt
cd scripts/exploratory
for s in step0_check a_macro_ap b_topk_ci c_drop_cumulative d_recalibration e_mc_sensitivity; do $PY post_$s.py; done
```
(E mất khoảng vài chục phút.) Môi trường: venv dựng từ `requirements.lock.txt`.

## File đã tạo

`scripts/exploratory/`: `post_common.py`, `post_step0_check.py`, `post_a_macro_ap.py`, `post_b_topk_ci.py`, `post_c_drop_cumulative.py`, `post_d_recalibration.py`, `post_e_mc_sensitivity.py`.

`outputs/exploratory/`: `post_step0_check.csv`, `post_a_macro_ap.csv`, `post_b_topk_ci.csv`, `post_b_topk_paired_diff.csv`, `post_c_drop_cumulative_tune.csv`, `post_d_recalibration.csv`, `post_d_calibrator_params.csv`, `post_d_reliability_deciles.csv`, `post_d_reliability.png`, `post_e_mc_sensitivity.csv`, `post_summary.md`.

Ngoài dự án: venv `/private/tmp/claude-501/venv`, cache `/private/tmp/claude-501/post_table.pkl`. Mỗi CSV có dòng đầu `# POST-HOC, không dùng để đổi kết luận P1–P3`.
