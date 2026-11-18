# -*- coding: utf-8 -*-
"""Evaluation: mAP50/mAP50-95 (all-points AP, per-class) trên held-out vc 320.

Portable từ solution/scripts/04_eval_q.py; khoá ảnh = tên file có .png.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import PKG, DATA, WORK, default_device
import detlib

DEVICE = os.environ.get("DEVICE") or default_device()

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
    ap.add_argument("--conf", type=float, default=0.0001)
    ap.add_argument("--iou", type=float, default=0.7)
    a = ap.parse_args()

    gt = load_gt(WORK / "ds_local/images/vc", WORK / "ds_local/labels/vc")
    print(f"val: {len(gt)} images (clean held-out)")
    for i, w in enumerate(a.models):
        ms = a.imgsz[i] if len(a.imgsz) > 1 else a.imgsz[0]
        p = detlib.predict_dir(w, WORK / "ds_local/images/vc", imgsz=ms, conf=a.conf, iou=a.iou,
                               tta=False, max_det=100, device=DEVICE)
        evaluate(p, gt, f"  {Path(w).parent.parent.name}/{Path(w).name}")


if __name__ == "__main__":
    main()