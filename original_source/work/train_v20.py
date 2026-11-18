# -*- coding: utf-8 -*-
"""v20 training driver: liteC architecture + kd_v20 (feature + cls-logit + normalised P2 KD).

Env: MODEL EPOCHS NAME LR0 SEED DATA BOX CLS BOXDW DIS DISCLS DISP2 TP2
     MOSAIC MIXUP CLOSE MSCALE SCALE SAVE_PERIOD SMOKE
SMOKE=1 runs 2 epochs on 15% of the data to sanity-check loss magnitudes.
"""
import os
import sys
from pathlib import Path

OUT = Path(r"F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI")
sys.path.insert(0, str(OUT / "solution/lib"))
sys.path.insert(0, str(OUT / "work"))
os.chdir(OUT)
import xpu_patch  # noqa: F401
import thread_loader

thread_loader.patch_infinite_loader(workers=int(os.environ.get("WORKERS", "6")), prefetch=4)
import lite_head

BOX = int(os.environ.get("BOX", "40"))
CLS = int(os.environ.get("CLS", "80"))
BOXDW = os.environ.get("BOXDW", "1") == "1"
lite_head.patch_parse_model(BOX, CLS, BOXDW)

import kd_v20

DIS = float(os.environ.get("DIS", "2.0"))
DISCLS = float(os.environ.get("DISCLS", "2.0"))
DISP2 = float(os.environ.get("DISP2", "0.5"))
kd_v20.install(offset=1, dis=DIS, dis_cls=DISCLS, dis_p2=DISP2, t_p2_idx=int(os.environ.get("TP2", "2")))

from ultralytics import YOLO

MODEL = os.environ.get("MODEL", str(OUT / "work/init_liteC.pt"))
TEACHER = os.environ.get("TEACHER", str(OUT / "models/teacher_q_m320_best.pt"))
EPOCHS = int(os.environ.get("EPOCHS", "320"))
NAME = os.environ.get("NAME", "q_v20a")
DATA = os.environ.get("DATA", str(OUT / "work/ds_local/data.yaml"))
LR0 = float(os.environ.get("LR0", "0.01"))
SEED = int(os.environ.get("SEED", "2024"))
SMOKE = os.environ.get("SMOKE", "0") == "1"

m = YOLO(MODEL)
p = sum(x.numel() for x in m.model.parameters())
print(f"[v20] init params={p} head={type(m.model.model[-1]).__name__}", flush=True)
assert p <= 200_000, p
assert isinstance(m.model.model[-1], lite_head.DetectLite), "head is not DetectLite"

kw = {}
ep = EPOCHS
name = NAME
save = int(os.environ.get("SAVE_PERIOD", "10"))
if SMOKE:
    ep, name, save = 2, NAME + "_smoke", -1
    kw.update(fraction=0.15)
r = m.train(
    data=DATA, imgsz=320, epochs=ep, batch=64, device="xpu:0", workers=0,
    project=str(OUT / "work/runs"), name=name, exist_ok=True,
    seed=SEED, patience=0, plots=False, val=True, cache="ram",
    optimizer="MuSGD", lr0=LR0, lrf=0.01, cos_lr=True, warmup_epochs=2.0,
    scale=float(os.environ.get("SCALE", "0.7")), mosaic=float(os.environ.get("MOSAIC", "0.5")),
    mixup=float(os.environ.get("MIXUP", "0.1")), close_mosaic=int(os.environ.get("CLOSE", "30")),
    multi_scale=float(os.environ.get("MSCALE", "0.0")),
    degrees=5.0, translate=0.15, fliplr=0.5, conf=0.01, max_det=100,
    save_period=save, deterministic=False,
    distill_model=TEACHER, dis=DIS,
    **kw,
)
print("TRAIN_DONE", r.save_dir)
print("RESULTS", r.results_dict)