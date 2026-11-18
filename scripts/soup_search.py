# -*- coding: utf-8 -*-
"""Greedy model-soup search trên held-out vc 320 (protocol khoá: conf 0.0001, IoU 0.70).

Usage: python scripts/soup_search.py ckpt1.pt ckpt2.pt ... [--conf 0.0001 --iou 0.70]
       [--imgsz 240x320] [--out work/soups/soup_vXX.pt]
Mỗi subset được trung bình trọng số (EMA state dicts), chấm mAP50 trên vc,
giữ subset tốt nhất; sau đó tinh chỉnh trọng số nguyên 1..3 (2 pass).
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import PKG, WORK, default_device
import eval_q as em
import detlib

DEVICE = os.environ.get("DEVICE") or default_device()

ap = argparse.ArgumentParser()
ap.add_argument("ckpts", nargs="+")
ap.add_argument("--conf", type=float, default=0.0001)
ap.add_argument("--iou", type=float, default=0.70)
ap.add_argument("--imgsz", type=str, default="240x320")
ap.add_argument("--out", type=str, default=str(WORK / "soups/soup_best.pt"))
a = ap.parse_args()
H, W = [int(x) for x in a.imgsz.split("x")]

IMG = WORK / "ds_local/images/vc"
LAB = WORK / "ds_local/labels/vc"
gt = em.load_gt(IMG, LAB)
files = sorted(gt)
A, B = files[0::2], files[1::2]
gA = {k: gt[k] for k in A}
gB = {k: gt[k] for k in B}


def unwrap(m):
    """Return the bare student DetectionModel (strips KD wrapper / EMA)."""
    for _ in range(10):
        if isinstance(m, torch.nn.Module):
            if hasattr(m, "student_model"):
                m = m.student_model
                continue
            if hasattr(m, "yaml"):
                return m
        if isinstance(m, dict):
            m = m.get("ema") if m.get("ema") is not None else m.get("model")
        else:
            m = getattr(m, "model", None)
        if m is None:
            raise RuntimeError("cannot unwrap checkpoint")
    raise RuntimeError("unwrap too deep")


def clean_sd(net):
    sd = {}
    for k, v in net.state_dict().items():
        if k.startswith("student_model."):
            k = k[len("student_model."):]
        elif k.startswith("teacher_model.") or k.startswith("projector."):
            continue
        sd[k] = v.detach().float()
    return sd


nets, sds = [], []
for ck in a.ckpts:
    obj = torch.load(ck, map_location="cpu", weights_only=False)
    net = unwrap(obj)
    nets.append(net)
    sds.append(clean_sd(net))

tmp = WORK / "soups/_tmp.pt"
tmp.parent.mkdir(parents=True, exist_ok=True)


def blend(idx_weights):
    tot = sum(w for _, w in idx_weights)
    acc = None
    for i, w in idx_weights:
        sd = sds[i]
        if acc is None:
            acc = {k: v * (w / tot) for k, v in sd.items()}
        else:
            for k, v in sd.items():
                if k in acc and acc[k].shape == v.shape:
                    acc[k] += v * (w / tot)
    net = nets[0]
    net.load_state_dict({k: v.to(next(net.parameters()).dtype) for k, v in acc.items()})
    torch.save({"model": net}, tmp)
    return str(tmp)


def score(idx_weights, label):
    t0 = time.time()
    path = blend(idx_weights)
    p = detlib.predict_dir(path, IMG, imgsz=[H, W], conf=a.conf, iou=a.iou, tta=False, max_det=100,
                           device=DEVICE)
    p = {k: v for k, v in p.items() if k in gt}
    ap50, ap5095, _ = em.evaluate(p, gt)
    apA = em.evaluate({k: p[k] for k in A}, gA)[0]
    apB = em.evaluate({k: p[k] for k in B}, gB)[0]
    print(f"{label:44s} {ap50:7.4f} {ap5095:8.4f} A {apA:7.4f} B {apB:7.4f} {time.time()-t0:4.0f}s", flush=True)
    return ap50, ap5095, apA, apB


names = [Path(c).name for c in a.ckpts]
singles = []
for i, n in enumerate(names):
    singles.append(score([(i, 1.0)], n))
order = sorted(range(len(names)), key=lambda i: -singles[i][0])
best_i = order[0]
best = singles[best_i]
sel = [(best_i, 1.0)]
print(f"start best single: {names[best_i]} {best[0]:.4f}", flush=True)
for i in order[1:]:
    cand = sel + [(i, 1.0)]
    s = score(cand, "+".join(names[j] for j, _ in cand))
    if s[0] > best[0] + 1e-5:
        sel, best = cand, s
        print(f"  -> keep {names[i]} ({s[0]:.4f})", flush=True)
# weight refinement: two-pass integer weights 1..3 on the selected subset
weights = [1] * len(sel)
for rnd in range(2):
    for pos in range(len(sel)):
        for wv in (0, 1, 2, 3):
            if wv == weights[pos]:
                continue
            trial = [(sel[k][0], wv if k == pos else weights[k]) for k in range(len(sel))]
            if sum(w for _, w in trial) == 0:
                continue
            s = score(trial, f"w{tuple(w for _, w in trial)}")
            if s[0] > best[0] + 1e-5:
                weights, best = [w for _, w in trial], s
                print(f"  -> weights {weights} ({s[0]:.4f})", flush=True)

path = blend([(sel[k][0], weights[k]) for k in range(len(sel))])
out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": nets[0]}, out)
print("BEST", [names[j] for j, _ in sel], weights, f"{best[0]:.4f}", f"{best[1]:.4f}", f"A {best[2]:.4f} B {best[3]:.4f}", flush=True)
print("SAVED", out, flush=True)