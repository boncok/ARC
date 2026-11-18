# -*- coding: utf-8 -*-
"""v20 training driver: liteC architecture + kd_v20 (feature + cls-logit + normalised P2 KD).

Env: MODEL EPOCHS NAME LR0 SEED DATA BOX CLS BOXDW DIS DISCLS DISP2 TP2
     MOSAIC MIXUP CLOSE MSCALE SCALE SAVE_PERIOD SMOKE DEVICE WORKERS
SMOKE=1 runs 2 epochs on 15% of the data to sanity-check loss magnitudes.

Chuỗi v22 (xem runs_args/*.args.yaml và README):
  q_v20a: MODEL=init_liteC.pt            EPOCHS=300 LR0=0.01  SEED=2024  DISCLS=2 DISP2=0.5 MOSAIC=0.5 MIXUP=0.1 CLOSE=30 SCALE=0.7
  q_v20b: MODEL=soups/soup_v22.pt        EPOCHS=160 LR0=0.005 SEED=777   DISCLS=4 DISP2=1   MOSAIC=0.3 MIXUP=0.05 CLOSE=20 SCALE=0.5
  q_v20c: MODEL=soups/soup_v23.pt        EPOCHS=160 LR0=0.003 SEED=4242  DISCLS=4 DISP2=1   MOSAIC=0.2 MIXUP=0.03 CLOSE=20 SCALE=0.4
  q_v20e: MODEL=soups/soup_v24.pt        EPOCHS=160 LR0=0.004 SEED=31415 DISCLS=8 DISP2=2   MOSAIC=0.2 MIXUP=0.03 CLOSE=20 SCALE=0.4
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import PKG, WORK, default_device
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

MODEL = os.environ.get("MODEL", str(WORK / "init_liteC.pt"))
TEACHER = os.environ.get("TEACHER", str(PKG / "models/teacher_q_m320_best.pt"))
EPOCHS = int(os.environ.get("EPOCHS", "320"))
NAME = os.environ.get("NAME", "q_v20a")
DATA = os.environ.get("DATA", str(WORK / "ds_local/data.yaml"))
LR0 = float(os.environ.get("LR0", "0.01"))
SEED = int(os.environ.get("SEED", "2024"))
DEVICE = os.environ.get("DEVICE") or default_device()
SMOKE = os.environ.get("SMOKE", "0") == "1"

m = YOLO(MODEL)
p = sum(x.numel() for x in m.model.parameters())
print(f"[v20] init params={p} head={type(m.model.model[-1]).__name__} device={DEVICE}", flush=True)
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
    data=DATA, imgsz=320, epochs=ep, batch=64, device=DEVICE, workers=0,
    project=str(WORK / "runs"), name=name, exist_ok=True,
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