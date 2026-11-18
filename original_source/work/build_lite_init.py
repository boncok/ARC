# -*- coding: utf-8 -*-
"""Build a P2-P4 model with a lite (capacity-rebalanced) head and save a fresh init checkpoint."""
import os, sys
from pathlib import Path

ROOT = Path(r"F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI")
sys.path.insert(0, str(ROOT / "work"))
import lite_head  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BOX = int(os.environ.get("BOX", "24"))
CLS = int(os.environ.get("CLS", "48"))
BOXDW = os.environ.get("BOXDW", "0") == "1"
OUT = Path(os.environ.get("OUT", str(ROOT / "work/init_liteA.pt")))

lite_head.patch_parse_model(BOX, CLS, BOXDW)
m = YOLO(os.environ.get("YAML", str(ROOT / "configs/yolo11_p2_lite.yaml")))
net = m.model
head = net.model[-1]
p = sum(x.numel() for x in net.parameters())
b = sum(x.numel() for x in net.buffers())
hp = sum(x.numel() for x in head.parameters())
print("head", type(head).__name__, "det_ch", [x[0].conv.in_channels for x in head.cv2], "params", p, "buffers", b, "head_p", hp, "stride", torch.as_tensor(net.stride).tolist())
assert isinstance(head, lite_head.DetectLite)
assert p <= 200_000, p
for i, l in enumerate(net.model):
    s = sum(x.numel() for x in l.parameters())
    if s:
        print("  ", i, type(l).__name__, s)
net.args = {"imgsz": 320}
torch.save({"model": net, "date": "lite-build", "train_args": {"imgsz": 320, "nc": 7}}, OUT)
print("saved", OUT, os.path.getsize(OUT))
