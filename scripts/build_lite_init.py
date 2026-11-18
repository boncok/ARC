# -*- coding: utf-8 -*-
"""Dựng checkpoint khởi tạo NGẪU NHIÊN từ YAML (liteC: 197.433 params, head DetectLite).

Mặc định đúng cấu hình v22: YAML=configs/yolo11_p2_liteC.yaml BOX=40 CLS=80 BOXDW=1.
Chạy: python scripts/build_lite_init.py   (env BOX/CLS/BOXDW/YAML/OUT để đổi)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import PKG, WORK
import lite_head  # noqa: E402
import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

BOX = int(os.environ.get("BOX", "40"))
CLS = int(os.environ.get("CLS", "80"))
BOXDW = os.environ.get("BOXDW", "1") == "1"
OUT = Path(os.environ.get("OUT", str(WORK / "init_liteC.pt")))

lite_head.patch_parse_model(BOX, CLS, BOXDW)
m = YOLO(os.environ.get("YAML", str(PKG / "configs/yolo11_p2_liteC.yaml")))
net = m.model
head = net.model[-1]
p = sum(x.numel() for x in net.parameters())
b = sum(x.numel() for x in net.buffers())
hp = sum(x.numel() for x in head.parameters())
print("head", type(head).__name__, "det_ch", [x[0].conv.in_channels for x in head.cv2],
      "params", p, "buffers", b, "head_p", hp, "stride", torch.as_tensor(net.stride).tolist())
assert isinstance(head, lite_head.DetectLite)
assert p <= 200_000, p
net.args = {"imgsz": 320}
OUT.parent.mkdir(parents=True, exist_ok=True)
torch.save({"model": net, "date": "lite-build", "train_args": {"imgsz": 320, "nc": 7}}, OUT)
print("saved", OUT, os.path.getsize(OUT))