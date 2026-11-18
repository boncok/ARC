

## 1. Mô hình huấn luyện trước ĐÃ dùng

### 1.1. Teacher distillation (ảnh hưởng tới MỌI submission detection)

- **Nguồn**: weight chính thức YOLO11m huấn luyện trên COCO (ultralytics).
- **Đường dẫn**: `models/teacher_q_m320_best.pt` (YOLO11m fine-tune tiếp trên
  6.000 ảnh được cấp).
- **Cách dùng**: chỉ làm teacher trong knowledge distillation (feature L2 +
  class-logit BCE tại P3/P4, và cosine P2 ở `kd_v20`). Teacher **không** nằm
  trong model nộp; model nộp là student liteC 197.433 tham số.
- **Run liên quan**: mọi run `q_v20*` (args.yaml ghi `distill_model`).

### 1.2. Chuỗi khởi tạo vòng loại cũ (v19–v21, KHÔNG dùng cho v22)

- **Nguồn**: YOLO11s / YOLO11m COCO → các run `qual_work`
  (`qual_work/REPORT_QUALIFIER.md` mục bảng kết quả).
- **Transfer**: `qual_work/p2/transfer_p2.py` chuyển weight P3-P5 sang P2-P4
  (tầng P2 mới khởi tạo ngẫu nhiên) → `kd6_p2_hp32_init.pt` → `q_p2a` →
  `models/p2s_a.pt` → các run liteA…liteM → `soup_v18` → submission v19/v20/v21.
- **v22**: student khởi tạo **ngẫu nhiên** từ YAML
  (`work/build_lite_init.py`, `configs/yolo11_p2_liteC.yaml`); tri thức COCO
  đi vào duy nhất qua teacher mục 1.1.

### 1.3. Thư viện framework (không phải weight pretrained)

- ultralytics 8.4.198 stock (không fork), torch 2.9.1+xpu, TensorFlow converter
  + onnx2tf. Khai báo để minh bạch chuỗi công cụ.

## 2. Dữ liệu bên ngoài: KHÔNG

- 100% ảnh train/val/calibration = **6.000 ảnh ban tổ chức cấp**
  (`work/ds_local`, `work/ds_all` là phân hoạch lại; xem
  `report_v19/04_du_lieu_va_giao_thuc_do.md`).
- 100% câu hỏi intent = 6.000 câu được cấp.
- **Không** dùng ảnh COCO hay bất kỳ ảnh ngoài nào; COCO chỉ hiện diện qua
  weight teacher (mục 1.1).
- Calibration PTQ/QAT: `export/out/calib_250.npy` trích từ ảnh được cấp.

## 3. Phần tự train from scratch (không pretrained)

- Student detection liteC: 197.433 tham số, init ngẫu nhiên, train trên ảnh
  được cấp (KD từ teacher mục 1.1).
- Intent TextCNN: 41.429 tham số, `nn.Embedding` init ngẫu nhiên
  (`analysis/textcnn_intent.py:67`), train trên câu hỏi được cấp, val_acc 1.0.

## 4. Lượng tử hóa

- PTQ: TF converter, representative dataset từ ảnh được cấp.
- QAT v2 (`export/qat/qat_v2.py`): fake-quant int8 trong vòng lặp, clamp SiLU
  [−0.2785, 8], distill từ twin FP32 của chính student; fine-tune trên ảnh được
  cấp. Không dữ liệu ngoài.

## 5. Minh chứng lưu giữ (điều 7)

| Hạng mục | Vị trí |
|---|---|
| Cấu hình mọi run train | `work/runs/*/args.yaml`, `qual_work/` |
| Metric từng epoch | `work/runs/*/results.csv` |
| Weight teacher + student + soup | `models/`, `work/runs/*/weights/`, `work/soups/` |
| Code train/KD/export/QAT | `work/`, `solution/`, `export/`, `analysis/`, `qual_work/` |
| Submission + validator | `submission_kd_v*.csv`, `work/v20_check.py` |
| Phân tích nguồn gốc + audit luật | `report_v19/01…10` |

## 6. Diễn giải điều 6 và cam kết

- Slide trang 5 cho phép pretrained kèm khai báo, không giới hạn vòng; trang
  luật điều 6 ghi "(chỉ được dùng ở VÒNG CHUNG KẾT)". Hai văn bản khác nhau.
- Đội khai báo đầy đủ ở mục 1–4 và nộp v22 cho vòng loại theo diễn giải slide;
  **chấp nhận phán quyết của ban tổ chức** nếu diễn giải trang luật được áp
  dụng.
- Phương án dự phòng đã chuẩn bị: train from-scratch không teacher COCO
  (`report_v19/10_tuan_thu_luat.md` mục 5A) để thay thế nếu bị yêu cầu.
- Ở chung kết, mọi hạng mục mục 1 tiếp tục được khai báo kèm hồ sơ này.