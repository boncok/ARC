# -*- coding: utf-8 -*-
"""Post-processing sweep on the HONEST holdout (vc 320), using q_liteM/last.pt.

Two letterbox regimes are compared:
  mode0 = imgsz [240,320], rect=True  -> tensor 256x320, 8+8 grey rows (current pipeline)
  mode1 = imgsz 320,       rect=False -> tensor 320x320, 40+40 grey rows (matches TRAIN canvas)
Then a grid over conf / NMS IoU / max_det / min box size / class-agnostic NMS.
Writes nothing; prints a ranked table.
"""
import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(r"F:/tele/edge-ai-challenge-2026-lan-2")
SOL = ROOT / "egde-ai-challenge-2026-lan-2-qualifiers" / "QUALIFIER_SOLUTION"
OUT = ROOT / "FINAL_VONG_LOAI"
sys.path.insert(0, str(SOL / "lib"))
sys.path.insert(0, str(OUT / "work"))
spec = importlib.util.spec_from_file_location("evalq", SOL / "scripts" / "04_eval_q.py")
m = importlib.util.module_from_spec(spec)
sys.modules["evalq"] = m
spec.loader.exec_module(m)
import detlib
from ultralytics import YOLO

W, H = 320, 240
IMG = ROOT / "qual_ds/images/val"
LAB = ROOT / "qual_ds/labels/val"
gt = m.load_gt(IMG, LAB)
keep = {l.strip() for l in (SOL / "data/val_clean.txt").read_text().splitlines() if l.strip()}
gt = {k: v for k, v in gt.items() if k in keep}
files = sorted(gt)
A, B = files[0::2], files[1::2]
gA = {k: gt[k] for k in A}
gB = {k: gt[k] for k in B}

WEIGHT = sys.argv[1] if len(sys.argv) > 1 else str(OUT / "work/runs/q_liteM/weights/last.pt")


def raw_predict(imgsz, rect):
    model = YOLO(WEIGHT)
    acc = {}
    for r in model.predict(source=str(IMG), imgsz=imgsz, conf=0.00005, iou=0.9, max_det=300,
                           augment=False, device="xpu:0", stream=True, verbose=False,
                           batch=8, save=False, rect=rect):
        name = Path(r.path).name
        if name not in gt:
            continue
        if r.boxes is None or len(r.boxes) == 0:
            acc[name] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, int)}
        else:
            acc[name] = {"boxes": r.boxes.xyxy.cpu().numpy().astype(np.float64),
                         "scores": r.boxes.conf.cpu().numpy().astype(np.float64),
                         "cls": r.boxes.cls.cpu().numpy().astype(np.int64)}
    return acc


def post(pred, conf, iou, max_det, min_size, agn):
    out = {}
    for name, d in pred.items():
        chunks = []
        for c in range(7):
            mk = (d["cls"] == c) & (d["scores"] >= conf)
            if not mk.any():
                continue
            r = detlib.nms(d["boxes"][mk], d["scores"][mk], np.full(int(mk.sum()), c), iou_thr=iou)
            if len(r["scores"]):
                bb = r["boxes"].copy()
                bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, W)
                bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, H)
                ok = (bb[:, 2] - bb[:, 0] > min_size) & (bb[:, 3] - bb[:, 1] > min_size)
                if ok.any():
                    chunks.append((bb[ok], r["scores"][ok], np.full(int(ok.sum()), c)))
        if not chunks:
            out[name] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, int)}
            continue
        bb = np.concatenate([k[0] for k in chunks])
        ss = np.concatenate([k[1] for k in chunks])
        cc = np.concatenate([k[2] for k in chunks])
        if agn:
            r = detlib.nms(bb, ss, cc, iou_thr=agn)
            bb, ss, cc = r["boxes"], r["scores"], r["cls"]
        if len(ss) > max_det:
            o = np.argsort(-ss)[:max_det]
            bb, ss, cc = bb[o], ss[o], cc[o]
        out[name] = {"boxes": bb, "scores": ss, "cls": cc}
    return out


rows = []
for mode, imgsz, rect in (("pad8", [240, 320], True), ("pad40", 320, False)):
    pred = raw_predict(imgsz, rect)
    print(f"[{mode}] raw dets = {sum(len(v['scores']) for v in pred.values())}", flush=True)
    for conf, iou, md, ms, agn in itertools.product(
            (0.0001, 0.0005, 0.002, 0.008), (0.6, 0.65, 0.7), (100, 300), (1, 3), (0, 0.9)):
        p = post(pred, conf, iou, md, ms, agn)
        ap50, _ = m.ap_at_iou(p, gt, 0.5)
        a50 = m.ap_at_iou({k: p[k] for k in A}, gA, 0.5)[0]
        b50 = m.ap_at_iou({k: p[k] for k in B}, gB, 0.5)[0]
        rows.append((ap50, a50, b50, mode, conf, iou, md, ms, agn))

rows.sort(key=lambda r: -r[0])
print(f"\n{'mAP50':>7} {'A':>7} {'B':>7}  mode   conf     iou  maxdet minsz agn")
for r in rows[:20]:
    print(f"{r[0]:7.4f} {r[1]:7.4f} {r[2]:7.4f}  {r[3]:5s} {r[4]:.4f} {r[5]:.2f} {r[6]:5d} {r[7]:5d} {r[8]:.1f}")
base = [r for r in rows if r[3] == "pad8" and r[4] == 0.0005 and r[5] == 0.65 and r[6] == 100 and r[7] == 1 and r[8] == 0]
print("\ncurrent pipeline config:", base[0][:3] if base else "n/a")