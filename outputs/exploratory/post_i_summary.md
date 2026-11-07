# POST-HOC, không dùng để đổi kết luận P1–P3

# Tổng kết nhóm post_i (C, E, F, D)

Bước 0 đạt: dựng lại Refit khớp `stage_d_test_metrics.csv` 48/48 chỉ số đến 4 chữ số (chênh lệch lớn nhất 1,1e-16). Trong C, B2, M1, M2, M3 còn khớp bảng khóa trên cả 9 chỉ số. Hiệu số "a − b" đều là bootstrap cụm ghép cặp, B = 1000, hạt giống 20261024, mô tả và không hiệu chỉnh bội. Cách đọc theo giao thức §8: khoảng chứa 0 là "không có khác biệt rõ", không phải tương đương. Hệ số chặn > 0 nghĩa là dự báo thiếu y = 0; độ dốc < 1 nghĩa là xác suất quá cực đoan.

## C. Baseline còn thiếu

- **Đã làm:** quy tắc theo `recency_to_median_gap` (C1a) và `median_gap_days` (C1b); logistic spline trên `recency_days` (C2, chọn C = 1 trên Tune); BG/NBD và Pareto/NBD tự cài bằng scipy (`lifetimes` chưa có, không cài thêm), ước lượng tại từng mốc từ lịch sử đến mốc đó (`_roll`) và một lần tại mốc huấn luyện cuối (`_fixed`). Các số dưới đây dùng `_roll`.
- **Số chính (AP temporal / hold-out):** B2 0,5502 / 0,5743; M2 0,5966 / 0,6110; M3 0,6045 / 0,6157; C1a 0,3781 / 0,4241; C1b 0,4194 / 0,4164; C2 0,5526 / 0,5764; BG/NBD 0,5415 / 0,5844; Pareto/NBD 0,5476 / 0,5940. ROC-AUC của BG/NBD và Pareto/NBD khoảng 0,90 so với 0,87 của B2 (khoảng hiệu số loại 0 ở cả hai tập), nhưng độ dốc hiệu chuẩn chỉ 0,54–0,63.
- **Khác kết luận hiện có:** P1–P3 không đổi; B2 vẫn là baseline mạnh nhất theo AP trên temporal. Điểm cần nói rõ: B2 được chốt là "mạnh nhất" chỉ trong tập B0, B1, B2. Pareto/NBD có AP không khác rõ so với B2 (temporal −0,0026 [−0,0271; +0,0198]; hold-out +0,0197 [−0,0305; +0,0613]) và ROC-AUC cao hơn. So với M2: Pareto/NBD thấp hơn rõ trên temporal (AP −0,0490 [−0,0734; −0,0263]) nhưng không khác rõ về AP trên hold-out (−0,0170 [−0,0617; +0,0208]); ROC-AUC của M2 vẫn cao hơn rõ ở cả hai tập, Brier cũng tốt hơn nhưng trên hold-out cận sát 0.
- **Hạn chế:** ngày mua đầu quan sát được bị cắt trái tại 2018-01-01 nhưng BG/NBD và Pareto/NBD coi đó là ngày bắt đầu; tham số s, β của Pareto/NBD gần như không xác định được riêng (s/β ≈ 1,5e-4 mỗi tuần, tức gần như không rời bỏ); C2 có thứ hạng gần như B2 nên không thêm tín hiệu mới.
- **Thời gian:** 1,0 phút (lần chạy cuối).

## E. Rolling-origin (thăm dò)

- **Đã làm:** M2 và M3 với siêu tham số khóa (LightGBM 114 cây), cửa sổ mở rộng, 8 mốc đánh giá từ 2020-03-31 đến 2021-12-31, chỉ người ngoài hold-out; B2 để tham chiếu.
- **Số chính:** AP trung bình (SD): B2 0,5298 (0,0210); M2 0,5709 (0,0386); M3 0,5763 (0,0342). M2 > B2 ở 8/8 mốc; M3 > M2 ở 6/8 mốc, hiệu số từ −0,0063 đến +0,0154. Hệ số chặn M2 từ −0,295 đến +0,309 (trung bình +0,068), M3 từ −0,240 đến +0,312 (+0,027); độ dốc M2 0,95–1,13, M3 1,08–1,25.
- **Khác kết luận hiện có:** không đổi P1–P3. Hệ số chặn khoảng +0,20 ở tập kiểm định nằm trong biên độ dao động giữa các mốc trước đó, nên hai quý kiểm định không có vẻ bất thường. Hệ số chặn có xu hướng dương hơn ở 4 mốc sau (M2 trung bình −0,061 rồi +0,196), nhưng chỉ có 8 điểm.
- **Hạn chế:** siêu tham số khóa được chọn trên Tune nên các mốc đầu có nhìn trước về siêu tham số, và 2 mốc đánh giá trùng Tune; không có khoảng tin cậy theo mốc; không đánh giá trên người hold-out.
- **Thời gian:** dưới 0,1 phút theo log.

## F. Giả thuyết "mẫu chỉ gồm người còn dùng Amazon vào cuối 2022"

- **Đã làm:** (1) hiệu chuẩn theo tam phân vị `tenure_days` trên Train (533 và 821 ngày) và theo số mốc có mặt (15 / 10–14 / 1–9); (2) M2 chuẩn hóa theo thâm niên, giữ hoặc bỏ `tenure_days`; (3) M2, M3 cửa sổ trượt 4 và 6 quý. Không dùng cột cuối Bảng A.1 làm bằng chứng.
- **Số chính:** hệ số chặn M2 theo số mốc có mặt (15 / 10–14 / 1–9): temporal +0,156 / +0,320 / +0,400; hold-out +0,168 / +0,352 / +0,419; Tune cùng chiều. Cửa sổ trượt giảm hệ số chặn: M2 từ +0,204 / +0,225 (temporal / hold-out) còn +0,074 / +0,097 (4 quý) và +0,073 / +0,100 (6 quý); M3 từ +0,174 / +0,204 còn +0,114 / +0,157 và +0,078 / +0,118. Bỏ `tenure_days` đưa M2 về +0,068 / +0,094. Thay đổi AP đều nhỏ: M2 từ −0,0051 đến +0,0036, tất cả chứa 0; riêng M3 4 quý trên temporal +0,0073 [+0,00001; +0,0135] sát 0, và trên Tune lại ngược dấu (−0,0124 [−0,0217; −0,0014]), nên không nên coi là cải thiện.
- **Khác kết luận hiện có:** không đổi P1–P3. Bổ sung cho §9 của giao thức: giao thức dự kiến dự báo thừa y = 0 vì tỷ lệ y = 0 giảm theo thời gian, nhưng hệ số chặn thực tế dương (dự báo thiếu). Kết quả phù hợp định tính với giả thuyết: người mới bị dự báo thiếu nhiều hơn, dữ liệu gần đây giảm sai lệch, và mô hình ước lượng gần như không có rời bỏ. **Nhưng các kết quả này không phân biệt được giả thuyết với các giải thích cạnh tranh** như trôi theo thời gian, khác biệt giữa các nhóm gia nhập, hay ngoại suy `tenure_days` vượt miền của Train. Đây là chẩn đoán, không phải bằng chứng cho giả thuyết.
- **Hạn chế:** ngưỡng tính trên Train nên nhóm T3 chiếm 94 % dòng temporal (T1 hold-out chỉ 59 dòng, 18 sự kiện); hệ số chặn theo nhóm không có khoảng tin cậy; số mốc có mặt chỉ là hàm của ngày mua đầu tiên nên trùng phần lớn với thâm niên; dữ liệu không có người đã rời mẫu nên không kiểm định trực tiếp được giả thuyết.
- **Thời gian:** 0,8 phút.

## D. Tinh chỉnh LightGBM

- **Đã làm:** random search 36 cấu hình (gồm ràng buộc đơn điệu như một chiều tìm kiếm), rolling-origin CV 3 lần chia tại 2021-03-31, 2021-06-30, 2021-09-30, dừng sớm trong từng lần chia. Chốt cfg 23 (`num_leaves` 15, `min_child_samples` 63, `learning_rate` 0,059, `reg_lambda` 0,92, `colsample_bytree` 0,79, `subsample` 0,55, không đơn điệu, 42 cây = trung vị số cây 3 lần chia), rồi Refit và chấm một lần trên hai tập kiểm định.
- **Số chính:** AP D 0,6018 / 0,6113 so với M3 khóa 0,6045 / 0,6157 (temporal / hold-out). D − M3: temporal −0,0027 [−0,0076; +0,0022]; hold-out −0,0044 [−0,0123; +0,0063]. ROC-AUC và Brier cũng không khác rõ. Hệ số chặn của D giảm còn +0,105 / +0,132 (khoảng hiệu số loại 0), nhưng độ dốc xấu đi: 1,28 / 1,29 so với 1,17 / 1,18.
- **Khác kết luận hiện có:** không đổi P2 và không có lý do đổi cấu hình M3. Tìm kiếm rộng hơn không cải thiện AP: AP CV của 36 cấu hình chỉ từ 0,5730 đến 0,5807, trong khi SD giữa các lần chia khoảng 0,05, và cấu hình khóa dừng sớm đạt 0,5818, cao hơn cả 36 cấu hình. Ràng buộc đơn điệu không tạo khác biệt đáng kể (tốt nhất 0,5802 so với 0,5807).
- **Hạn chế:** dừng sớm dùng chính mốc đánh giá nên AP CV lạc quan; thứ hạng cấu hình gần như là nhiễu nên cfg 23 không có gì đặc biệt; số cây không tính lại theo cỡ mẫu Refit (giống §5 giao thức).
- **Thời gian:** 1,0 phút cho toàn nhóm (tìm kiếm 0,7 phút), xa dưới giới hạn 90 phút.

## Lỗi, cắt giảm và điểm lệch khỏi yêu cầu

- **Không nhóm nào bị lỗi ở lần chạy cuối, không nhóm nào bị cắt giảm** (36/36 cấu hình). Không có `ConvergenceWarning`; không cấu hình nào chạm trần 3000 cây.
- **C được chấm trên tập kiểm định hai lần.** Lần đầu phát hiện 13 giá trị log-likelihood NaN/inf của Pareto/NBD (hàm siêu bội tràn số với x ≈ 860–960), bị gán P0 ≈ 0 âm thầm. Việc sửa xuất phát từ lỗi số học, không phải từ chỉ số kiểm định, và không có lựa chọn cấu hình nào đổi. AP Pareto/NBD `_roll` đổi từ 0,5480 / 0,5935 sang 0,5476 / 0,5940; các mô hình khác không đổi. E, F, D chấm một lần.
- **Kiểm chứng BG/NBD và Pareto/NBD:** `python3 post_i_clv.py` so dạng đóng với tích phân số (kể cả x ≈ 950) và khôi phục tham số trên dữ liệu mô phỏng. Không có đối chiếu với thư viện `lifetimes`.
- **`post_step0_check.csv` bị ghi lại** khi chạy Bước 0, vì script đó tự ghi đè file này (kết quả vẫn 48/48). Không có bản sao để so từng byte với file cũ.
- **File đã khóa:** theo thời điểm sửa đổi, không file nào trong `src/`, `tests/`, `reports/`, `scripts/` gốc và `outputs/` gốc bị đổi; các `post_*` khác không bị đụng. Hai thư mục `__pycache__` do lần chạy của tôi sinh ra đã được xóa. Không commit, không push, không sửa README hay file Word.
- **Không tìm thấy** `README.md` và `SUBMISSION_CHECKLIST.md` trong thư mục; giao thức nằm ở `reports/stage_d_protocol.md` và chỉ được đọc đầy đủ sau khi đã chạy xong (đối chiếu lại không có gì cần sửa). Đường dẫn dữ liệu lấy từ `src/apf/config.py`.
- **Danh sách quy tắc trong yêu cầu không có mục 4** (nhảy từ 3 sang 5).
- **Chạy lại:** đặt `APF_DATA_DIR=../du_lieu_tho` (tính từ thư mục dự án) và `APF_POST_CACHE`. Điểm dừng nằm ở `APF_POST_CACHE/post_i_ckpt`, ngoài dự án. Nếu mất điểm dừng, chạy lại D sẽ chấm lại tập kiểm định; kết quả cần được coi là lần chấm thứ hai.

## File đã tạo

- **`scripts/exploratory/`:** `post_i_common.py`, `post_i_clv.py`, `post_i_c_baselines.py`, `post_i_e_rolling_origin.py`, `post_i_f_survivor.py`, `post_i_d_lgbm_search.py`
- **C:** `post_i_c_baselines.csv`, `post_i_c_paired_diff.csv`, `post_i_c_spline_selection.csv`, `post_i_c_clv_params.csv`, `post_i_c.log`
- **E:** `post_i_e_rolling_origin.csv`, `post_i_e_rolling_origin_agg.csv`, `post_i_e.log`
- **F:** `post_i_f_variants.csv`, `post_i_f_calibration_by_group.csv`, `post_i_f_paired_diff.csv`, `post_i_f.log`
- **D:** `post_i_d_search.csv`, `post_i_d_final.csv`, `post_i_d_paired_diff.csv`, `post_i_d.log`
- **Tổng kết:** `post_i_summary.md` (tất cả file kết quả nằm trong `outputs/exploratory/`)
