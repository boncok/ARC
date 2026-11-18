# submission_v22 — Toàn bộ code tạo bài nộp Edge AI Challenge 2026 (vòng loại)

Gói này chứa **đúng các file code đã tạo ra** `submission_kd_v22.csv` (file đã nộp Kaggle),
kèm model Detect + Intent và tài liệu tuân thủ. Code được xuất ở **2 dạng**:

- **`original_source/`** — bản **GỐC verbatim** (copy từng byte từ máy train): đúng các file
  đã chạy để tạo v22, giữ nguyên cấu trúc thư mục và đường dẫn gốc. Đây là bằng chứng nguồn gốc.
- **`scripts/`** — bản **chạy được ngay** (portable): cùng logic, đã sửa đường dẫn sang
  `PKG/DATA/WORK` qua biến môi trường. Đã kiểm chứng tái tạo CSV **giống hệt từng byte**.

**Kiểm chứng 2 lần độc lập** (xem mục 8): chạy lại từ script gốc → identical; chạy lại chỉ bằng
gói này → `PACKAGE_CSV_IDENTICAL`, validator `PASS` (8000 dòng, 185.505 box, 0 lỗi cấu trúc).

---

## 1. Cấu trúc thư mục

```
submission_v22/
├── original_source/            # ★ CODE GỐC verbatim đã tạo v22 (giữ nguyên path máy train)
│   ├── work/                   #   chuỗi train KD + soup + dựng submission
│   ├── solution/lib/           #   detlib.py, xpu_patch.py
│   ├── solution/scripts/       #   01_build_dataset, 02_check_leak, 04_eval_q
│   ├── analysis/               #   intent_q, intent_oov, textcnn_intent
│   └── configs/                #   yolo11_p2_liteC.yaml
├── scripts/                    # ★ Bản portable chạy được ngay (đã kiểm chứng byte-identical)
├── ultralytics/                # Fork ultralytics 8.4.198 ĐÃ VÁ (MuSGD, distill_model, dis…)
├── configs/yolo11_p2_liteC.yaml# Kiến trúc detector liteC (strides 4/8/16, P2)
├── manifests/                  # val_clean.txt (320 ảnh vc held-out) + mon.txt (92 ảnh monitor)
├── models/
│   ├── detect_q_v20e_best.pt   # ★ Champion detector FP32 (197.433 params) — tạo submission
│   ├── detect_soup_v24.pt      # Soup trung gian (0.9896 vc) — điểm khởi đầu của q_v20e
│   ├── detect_init_liteC.pt    # Khởi tạo ngẫu nhiên liteC
│   ├── teacher_q_m320_best.pt  # Teacher KD (chỉ dùng lúc train)
│   ├── intent_tfidf_oov.joblib # ★ Model intent (TF-IDF + LogReg C=20) fit 6000 câu
│   └── textcnn_intent.pt       # TextCNN cho firmware ESP32 (không thuộc submission)
├── reference/
│   ├── submission_kd_v22.csv   # ★ File đã nộp Kaggle — chuẩn đối chiếu byte-identical
│   └── intent_test_predictions.csv
├── runs_args/                  # args.yaml thật 4 lần train + log soup (bằng chứng tham số)
├── declaration/                # 10_tuan_thu_luat.md + 11_khai_bao_pretrained.md
├── notebooks/                  # (bổ sung tuỳ chọn — xem mục 10)
├── requirements.txt
└── work/                       # SINH RA khi chạy (dataset/runs/cache/CSV) — đã .gitignore
```

## 2. Các file code gốc (`original_source/`) và vai trò

| File | Vai trò trong chuỗi tạo v22 |
|------|------------------------------|
| `work/build_lite_init.py` | Khởi tạo liteC ngẫu nhiên từ YAML (env `BOX=40 CLS=80 BOXDW=1 YAML=…liteC.yaml OUT=…init_liteC.pt`) → 197.433 params |
| `work/train_v20.py` | Train KD liteC (env `MODEL EPOCHS NAME LR0 SEED DATA DIS DISCLS DISP2 MOSAIC MIXUP CLOSE SCALE…`) |
| `work/kd_v20.py` | `KdV20`: feature L2 + BCE class-logit P3/P4 + cosine P2 chuẩn hoá kênh |
| `work/kd_aligned.py` / `kd_aligned2.py` | `FeatureHook` / `KdModel2` (nền của KdV20) |
| `work/lite_head.py` | `DetectLite` (box_ch=40, cls_ch=80, box_dw, reg_max=16) + `patch_parse_model` |
| `work/thread_loader.py` | Loader luồng nền (thay workers của ultralytics) |
| `work/soup_search.py` | Greedy model soup + tinh chỉnh trọng số nguyên trên vc |
| `work/v20_post.py` | Sweep hậu xử lý → khoá conf 0.0001 / IoU 0.70 |
| `work/v20_build.py` | **Dựng submission v22** (raw conf .0001/iou .9/maxdet 300 → post conf .0001/NMS .70/maxdet 100) |
| `work/v20_check.py` | Validator cấu trúc nghiêm ngặt so với sample_submission |
| `work/make_ds_all.py` | Dựng ds_all (6000 train) từ ds_local |
| `solution/lib/detlib.py` | Inference / NMS / fusion / ghi submission |
| `solution/lib/xpu_patch.py` | NMS chạy CPU (nhanh trên Intel Arc; vô hại trên CUDA/CPU) |
| `solution/scripts/01_build_dataset.py` | Dựng dataset YOLO + kiểm tra nhãn |
| `solution/scripts/02_check_leak.py` | Tạo val sạch `val_clean.txt` (average-hash, loại gần-trùng) |
| `solution/scripts/04_eval_q.py` | Đánh giá mAP50 / mAP50-95 |
| `analysis/intent_q.py` | Intent TF-IDF (word+char) + LogReg, 5-fold CV |
| `analysis/intent_oov.py` | **Intent + lớp luật OOV** → `intent_test_predictions.csv` (3 cột) |
| `analysis/textcnn_intent.py` | TextCNN intent cho firmware (không dùng trong submission) |
| `configs/yolo11_p2_liteC.yaml` | Kiến trúc detector liteC |

**Đường dẫn gốc (hardcoded) trong `original_source/`:** các file `work/*` dùng
`OUT = F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI` (import `OUT/solution/lib`, `OUT/work`;
dữ liệu `OUT/data`; model `OUT/models`; intent `OUT/results`). `soup_search.py` và `analysis/intent_*`
tham chiếu thêm `…/egde-ai-challenge-2026-lan-2-qualifiers`, `…/qual_ds`, `…/qual_work`.
→ Muốn chạy trên máy khác: sửa các hằng `OUT/ROOT/SOL/Q/WORK` về đúng bố cục mới, hoặc dùng
bản `scripts/` (mục 5) đã portable sẵn.

## 3. Dữ liệu BTC (chỉ đọc, không nằm trong gói)

Đặt `data/` của BTC **ngang hàng** thư mục gói, hoặc trỏ bằng `LAN2_DATA`:

```
<thư mục cha>/
├── data/                  # hoặc LAN2_DATA=/đường/dẫn/khác
│   ├── train/images/      # 6000 ảnh .png (320×240)
│   ├── train/labels/      # 6000 nhãn YOLO (7 lớp)
│   ├── test/images/       # 4000 ảnh .png
│   ├── intent_train.csv   # 6000 câu (question_id, question, intent)
│   ├── intent_test.csv    # 4000 câu (question_id, question)
│   ├── sample_submission.csv
│   └── classes.txt
└── submission_v22/
```

Mọi script **chỉ đọc** `data/`; thứ sinh ra nằm trong `work/` của gói (đổi bằng `LAN2_WORK`).
Split dùng cho v22: `ds_local` = train 5680 (6000 − vc320) / vc 320 (`manifests/val_clean.txt`) /
mon 92 (`manifests/mon.txt`); `ds_all` = 6000 train / val mon (chỉ polish, không dùng chấm chính).

## 4. Cài môi trường

Python ≥ 3.10 (đã kiểm chứng 3.12). Cài torch theo backend trước:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124   # CUDA
# hoặc .../whl/cpu  (CPU thuần)   hoặc .../whl/xpu  (Intel Arc — máy gốc)
pip install -r requirements.txt
```

**Quan trọng:** dùng fork `ultralytics/` đóng gói kèm (đã vá MuSGD + KD), **không** cài ultralytics
từ pip. Đặt `PYTHONPATH` trỏ vào thư mục gói để fork được ưu tiên:

```bash
cd submission_v22
export PYTHONPATH=$PWD                 # Linux/macOS
# Windows cmd:        set PYTHONPATH=%CD%
# Windows PowerShell: $env:PYTHONPATH = (Get-Location).Path
```

Thiết bị tự chọn XPU → CUDA → CPU (ghi đè bằng `DEVICE`). MuSGD + patch device-neutral nên
chạy được trên máy CUDA/CPU của BTC.

## 5. Chạy bằng `scripts/` (portable, đã kiểm chứng byte-identical)

**Cách nhanh — xuất từ model đóng gói (~5 phút, giống từng byte file đã nộp):**

```bash
python scripts/intent_oov.py --from-model models/intent_tfidf_oov.joblib
python scripts/build_submission.py --weights models/detect_q_v20e_best.pt \
       --out work/submission.csv --cache work/predcache_export.npz
python scripts/check_submission.py --path work/submission.csv
# đối chiếu: work/submission.csv == reference/submission_kd_v22.csv (từng byte)
```

**Cách đầy đủ — train lại end-to-end từ đầu (~7 giờ GPU):**

```bash
python scripts/build_datasets.py        # ds_local + ds_all từ data BTC + manifests
python scripts/build_lite_init.py       # init liteC 197.433 params
# chuỗi train (đặt env đúng runs_args/*.args.yaml): q_v20a → soup_v22 → q_v20b
#   → soup_v23 → q_v20c → soup_v24 → q_v20e  (xem mục 6)
python scripts/train_v20.py             # mỗi giai đoạn 1 lần, env khác nhau
python scripts/soup_search.py <ckpt…>   # soup trên vc
python scripts/intent_oov.py            # intent TF-IDF + OOV (fit 6000 câu)
python scripts/build_submission.py --weights work/runs/q_v20e/weights/best.pt \
       --out work/submission.csv --cache work/predcache_retrain.npz
python scripts/check_submission.py --path work/submission.csv
```

Inference 4000 ảnh test ~3–5 phút (GPU) / ~15–25 phút (CPU). Cache raw kèm sha256 weights +
cấu hình; cache khác model/cấu hình bị từ chối tự động. Train đặt `deterministic=False` nên
chạy lại trên phần cứng khác có thể lệch số rất nhỏ; file nộp chính thức là CSV từ **cách nhanh**
(đảm bảo giống từng byte), BTC chấm lại trong ngưỡng lệch 0,5 điểm so với Kaggle.

## 6. Chuỗi tạo v22 (recipe tóm tắt)

**Detect** — liteC 197.433 params, KD nhiều giai đoạn + model soup (teacher `teacher_q_m320_best.pt`):

| # | Giai đoạn | Tham số kunci | vc mAP50 |
|---|-----------|----------------|----------|
| 1 | init liteC | ngẫu nhiên từ YAML, 197.433 params | — |
| 2 | `q_v20a` | 300 ep, lr0 .01, seed 2024, DIS 2 / DISCLS 2 / DISP2 0.5, mosaic .5 / mixup .1 / close 30 / scale .7 | — |
| 3 | `soup_v22` | greedy → epoch290 | 0.9851 |
| 4 | `q_v20b` | 160 ep, lr0 .005, seed 777, DIS 4 / DISCLS 1, mosaic .3 / mixup .05 / close 20 / scale .5 | — |
| 5 | `soup_v23` | [last, epoch150] trọng số [1,1] | — |
| 6 | `q_v20c` | 160 ep, lr0 .003, seed 4242, DIS 4 / DISCLS 1, mosaic .2 / mixup .03 / close 20 / scale .4 | — |
| 7 | `soup_v24` | [v20b/last ×1, v20c/epoch120 ×2] | 0.9896 |
| 8 | `q_v20e` | 160 ep, lr0 .004, seed 31415, DIS 4 / DISCLS 8 / DISP2 2, mosaic .2 / mixup .03 / close 20 / scale .4 | **0.9897** |

Tham số chính xác từng lần: `runs_args/*.args.yaml`. `models/detect_soup_v24.pt` là checkpoint
giữa chuỗi (có thể train thẳng `q_v20e` từ nó ~1,5 h thay vì chạy lại từ đầu).

**Intent** — `intent_oov.py`: chuẩn hoá NFD bỏ dấu / đ→d; TF-IDF union word(1–2) + char_wb(2–5),
`min_df=2`, `sublinear_tf`; `LogisticRegression(C=20, max_iter=3000)` fit **toàn bộ 6000 câu**;
lớp luật OOV (vật thể ngoài 7 lớp → OUT_OF_SCOPE, trừ LIST) ghi đè 74 câu test.
Phân phối test: LIST 800 · EXIST 800 · COUNT 800 · OUT_OF_SCOPE 798 · LOCATION 802.

**Công thức submission khoá (KHÔNG chỉnh nếu muốn giống file đã nộp):** 1 model FP32 duy nhất,
không TTA/ensemble; input 320×240 (letterbox 256×320), `imgsz=320`; raw `conf=0.0001, iou=0.9,
max_det=300`; post `conf=0.0001`, NMS theo lớp `IoU=0.70` (so sánh `>=`), `max_det=100`, loại box
cạnh ≤ 1 px, kẹp biên [0,320]×[0,240], làm tròn score 5 chữ số / bbox 2 chữ số.

## 7. Tuân thủ luật & khai báo

- Detector **197.433 tham số ≤ 200.000** (assert trong `build_lite_init.py`).
- Suy luận **1 model, không TTA, không ensemble**; ảnh đầu vào ≤ 320×240.
- Khai báo nguồn gốc pretrained: `declaration/11_khai_bao_pretrained.md` (đã chốt: **giữ v22 +
  khai báo pretrained**). Đối chiếu luật 4–7: `declaration/10_tuan_thu_luat.md`.
- LB = 70·mAP50(test) + 30·Accuracy(intent). Mốc hiệu chuẩn: v19 = 98.49687, v20 = 98.82758,
  v21 = 98.81534 (offset val→test ≈ 0.0064). Điểm LB của v22: xem lịch sử nộp Kaggle của đội.

## 8. Bằng chứng kiểm chứng (máy gốc Intel Arc B580)

```
PACKAGE_CSV_IDENTICAL            # work/submission.csv == reference/submission_kd_v22.csv (từng byte)
rows 8000 det 4000 intent 4000
boxes 185505 >=0.5 25384 oob 0 degenerate 0 empty_img 0 max_per_img 100
intent dist {'LIST': 800, 'EXIST': 800, 'COUNT': 800, 'OUT_OF_SCOPE': 798, 'LOCATION': 802}
PASS: all structural checks ok
```

Raw inference 332.195 detection / 4000 ảnh → hậu xử lý còn 185.505 (46,4 box/ảnh). Intent train
lại từ đầu trong gói cho CSV giống hệt `reference/intent_test_predictions.csv`.

## 9. Đưa lên GitHub

- `.gitignore` loại `/work/` (sinh ra), `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`;
  **không** loại `original_source/work/` (code gốc cần nộp).
- File lớn nhất `models/teacher_q_m320_best.pt` ≈ 40,5 MB < 100 MB → **không cần Git LFS**.
- Không commit `data/ của BTC` (dữ liệu cuộc thi, chỉ đọc) — đặt riêng theo mục 3.
- Sau khi clone: cài môi trường (mục 4) → chạy mục 5 (cách nhanh) để tái tạo file nộp.

## 10. Notebook (bổ sung tuỳ chọn)

`notebooks/` chứa bản notebook bọc các script trên (01 train end-to-end, 02 xuất submission) —
phần bổ sung nếu BTC yêu cầu dạng notebook; **trọng tâm gói là các file code** ở
`original_source/` (gốc) và `scripts/` (chạy được). Có thể xoá `notebooks/` nếu chỉ nộp code.