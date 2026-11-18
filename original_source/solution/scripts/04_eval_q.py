# -*- coding: utf-8 -*-
"""BƯỚC 04: Đánh giá detector: mAP50 / mAP50-95 trên val và val sạch

Pipeline Edge AI Challenge 2026 - vòng loại (7 lớp, ảnh 320x240).
Chạy:  <python> QUALIFIER_SOLUTION/scripts/04_eval_q.py
"""
"""Evaluation for the qualifier round: mAP50/mAP50-95 on the 600-image val,
optionally restricted to the 320 images with no near-duplicate in train."""
import argparse, sys
from pathlib import Path

import numpy as np

# --- đường dẫn (portable: tính từ vị trí file, không hard-code) -------------
SOL = Path(__file__).resolve().parents[1]      # thư mục QUALIFIER_SOLUTION
DATA = Path(__import__("os").environ.get("LAN2_DATA") or (SOL.parent / "data"))                              # thư mục ...-qualifiers (train/, test/)
WORK = SOL                                     # nơi ghi dataset / runs / results
DS = SOL / "dataset"                              # dataset YOLO đã chia (bước 01)
sys.path.insert(0, str(SOL / "lib"))
import detlib  # noqa: F401  (CPU NMS patch)

CLASSES = (DATA / "classes.txt").read_text().split()
W, H = 320, 240


def load_gt(img_dir, lab_dir):
    gt = {}
    for lab in sorted(Path(lab_dir).glob("*.txt")):
        boxes, cls = [], []
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            c, cx, cy, w, h = [float(v) for v in line.split()]
            boxes.append([(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H])
            cls.append(int(c))
        gt[lab.stem + ".png"] = {"boxes": np.array(boxes).reshape(-1, 4),
                                 "cls": np.array(cls, dtype=int)}
    return gt


def ap_at_iou(pred, gt, iou_thr, n_classes=7):
    aps = {}
    for c in range(n_classes):
        dets, npos = [], 0
        for name, g in gt.items():
            npos += int((g["cls"] == c).sum())
            p = pred.get(name)
            if p is None:
                continue
            m = p["cls"] == c
            for b, s in zip(p["boxes"][m], p["scores"][m]):
                dets.append((name, float(s), b))
        if npos == 0:
            aps[c] = None
            continue
        if not dets:
            aps[c] = 0.0
            continue
        dets.sort(key=lambda t: -t[1])
        used = {n: np.zeros(len(g["boxes"]), bool) for n, g in gt.items()}
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for i, (name, s, b) in enumerate(dets):
            g = gt[name]
            m = g["cls"] == c
            gb = g["boxes"][m]
            if len(gb) == 0:
                fp[i] = 1
                continue
            ov = detlib.iou_matrix(np.array([b]), gb)[0]
            j = int(np.argmax(ov))
            if ov[j] >= iou_thr and not used[name][m][j]:
                tp[i] = 1
                used[name][np.where(m)[0][j]] = True
            else:
                fp[i] = 1
        tpc, fpc = np.cumsum(tp), np.cumsum(fp)
        rec = tpc / npos
        prec = tpc / np.clip(tpc + fpc, 1e-9, None)
        mrec = np.concatenate([[0.0], rec, [1.0]])
        mpre = np.concatenate([[1.0], prec, [0.0]])
        for k in range(len(mpre) - 2, -1, -1):
            mpre[k] = max(mpre[k], mpre[k + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        aps[c] = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    vals = [v for v in aps.values() if v is not None]
    return (float(np.mean(vals)) if vals else 0.0), aps


def evaluate(pred, gt, label=""):
    ap50, per50 = ap_at_iou(pred, gt, 0.5)
    ap5095 = float(np.mean([ap_at_iou(pred, gt, t)[0] for t in np.arange(0.5, 1.0, 0.05)]))
    if label:
        print(f"{label:34s} mAP50={ap50:.4f}  mAP50-95={ap5095:.4f}")
    return ap50, ap5095, per50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--imgsz", type=int, nargs="+", default=[320])
    ap.add_argument("--tta", type=int, default=1)
    ap.add_argument("--clean", type=int, default=0)
    ap.add_argument("--fusion", default="none", choices=["none", "nms"])
    ap.add_argument("--per_class", type=int, default=1)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="ngưỡng conf khi dựng box; cấu hình nộp dùng 0.0005")
    ap.add_argument("--iou", type=float, default=0.7,
                    help="NMS IoU; cấu hình nộp dùng 0.65")
    a = ap.parse_args()

    img_dir = DS / "images/val"
    gt = load_gt(img_dir, DS / "labels/val")
    if a.clean:
        keep = {l.strip() for l in (SOL / "data/val_clean.txt").read_text().splitlines() if l.strip()}
        gt = {k: v for k, v in gt.items() if k in keep}
    print(f"val: {len(gt)} images, {sum(len(g['boxes']) for g in gt.values())} boxes"
          f"{' (clean subset)' if a.clean else ''}")

    preds = []
    for i, w in enumerate(a.models):
        ms = a.imgsz[i] if len(a.imgsz) > 1 else a.imgsz[0]
        print(f"  predict {Path(w).parent.parent.name} @{ms} tta={bool(a.tta)}", flush=True)
        preds.append(detlib.predict_dir(w, img_dir, imgsz=ms, conf=a.conf, iou=a.iou,
                                        tta=bool(a.tta), max_det=100))
    if a.fusion == "none" or len(preds) == 1:
        for w, p in zip(a.models, preds):
            evaluate(p, gt, f"  {Path(w).parent.parent.name}")
    else:
        fused = {}
        for k in gt:
            ps = [p[k] for p in preds if k in p]
            bs = np.concatenate([p["boxes"] for p in ps])
            sc = np.concatenate([p["scores"] for p in ps])
            cl = np.concatenate([p["cls"] for p in ps])
            fused[k] = detlib.nms(bs, sc, cl, iou_thr=0.55)
        ap50, ap5095, per = evaluate(fused, gt, f"  FUSION({len(preds)})")
        if a.per_class:
            print("   per-class AP50:", {CLASSES[c]: round(v, 3) for c, v in per.items() if v is not None})


if __name__ == "__main__":
    main()
