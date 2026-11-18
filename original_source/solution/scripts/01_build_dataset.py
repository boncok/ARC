# -*- coding: utf-8 -*-
"""BƯỚC 01: dựng dataset YOLO + kiểm tra nhãn (label) + thống kê.

Nhãn do BTC cung cấp sẵn ở dạng YOLO chuẩn hoá:  "<class_id> <cx> <cy> <w> <h>"
(cx,cy,w,h ∈ [0,1], gốc trên-trái). Script này KHÔNG sửa nhãn — nó *kiểm tra* nhãn
và chuyển thành dataset chuẩn cho Ultralytics, đồng thời in thống kê để quyết định
kích thước huấn luyện / độ phân giải suy luận.

Việc kiểm tra gồm:
  - class_id có nằm trong 0..6 không
  - toạ độ có nằm trong [0,1] không, w/h > 0
  - box có tràn ra ngoài ảnh không (sau khi nhân với 320x240)
  - file nhãn rỗng (ảnh không có vật thể) — bộ này không có
  - số box mỗi lớp, số box mỗi ảnh, phân bố diện tích box (px)

Chạy: python QUALIFIER_SOLUTION/scripts/01_build_dataset.py
"""
import random
import shutil
from pathlib import Path

import numpy as np

SOL = Path(__file__).resolve().parents[1]
PKG = SOL.parent
DATA = Path(__import__("os").environ.get("LAN2_DATA") or (PKG / "data"))
W, H = 320, 240
VAL_SIZE = 600
SEED = 0

classes = (DATA / "classes.txt").read_text(encoding="utf-8").split()
print(f"lớp ({len(classes)}):", classes)


def link(src: Path, dst: Path):
    """Hardlink để không tốn dung lượng; fallback copy nếu khác ổ đĩa."""
    if dst.exists():
        dst.unlink()
    try:
        dst.hardlink_to(src)
    except Exception:
        shutil.copy2(src, dst)


def main():
    img_dir, lab_dir = DATA / "train/images", DATA / "train/labels"
    imgs = sorted(img_dir.glob("*.png"))
    print(f"train: {len(imgs)} ảnh | test: {len(list((DATA / 'test/images').glob('*.png')))} ảnh")

    counts = {c: 0 for c in range(len(classes))}
    per_img, areas = [], []
    bad_class = bad_range = bad_size = 0
    empty = 0
    for p in imgs:
        lab = lab_dir / f"{p.stem}.txt"
        lines = [l for l in lab.read_text().splitlines() if l.strip()]
        if not lines:
            empty += 1
        per_img.append(len(lines))
        for line in lines:
            parts = line.split()
            c = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            if c not in counts:
                bad_class += 1
                continue
            counts[c] += 1
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                bad_range += 1
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            if x1 < -1 or y1 < -1 or x2 > W + 1 or y2 > H + 1:
                bad_size += 1
            areas.append(w * W * h * H)

    areas = np.array(areas)
    print(f"\nkiểm tra nhãn: class sai {bad_class} | toạ độ sai {bad_range} | box tràn ảnh {bad_size}"
          f" | file rỗng {empty}")
    print("box/lớp:", {classes[c]: counts[c] for c in counts}, "| tổng:", sum(counts.values()))
    print("box/ảnh: TB %.2f  max %d" % (np.mean(per_img), max(per_img)))
    print("diện tích box (px): p10=%.0f p50=%.0f p90=%.0f  | nhỏ hơn 32x32: %.1f%%"
          % (*np.percentile(areas, [10, 50, 90]), 100 * (areas < 1024).mean()))
    print("→ 40% box nhỏ hơn 32x32 ⇒ nên suy luận ở 448 (upscale) dù ảnh gốc 320x240")

    # ---- chia train/val ----
    rng = random.Random(SEED)
    order = imgs[:]
    rng.shuffle(order)
    val, train = order[:VAL_SIZE], order[VAL_SIZE:]
    out = SOL / "dataset"
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (out / sub).mkdir(parents=True, exist_ok=True)
        for f in (out / sub).glob("*"):
            f.unlink()
    for split, files in (("train", train), ("val", val)):
        for p in files:
            link(p, out / "images" / split / p.name)
            link(lab_dir / f"{p.stem}.txt", out / "labels" / split / f"{p.stem}.txt")
    print(f"\ndataset: train {len(train)} / val {len(val)} (seed={SEED})")

    # ---- data.yaml ----
    lines = [f"path: {out.as_posix()}", "train: images/train", "val: images/val", "names:"]
    lines += [f"  {i}: {c}" for i, c in enumerate(classes)]
    (out / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out / "data.yaml").read_text(encoding="utf-8"))

    # ---- manifest ----
    (SOL / "data/splits").mkdir(parents=True, exist_ok=True)
    for name, files in (("train", train), ("val", val)):
        (SOL / "data/splits" / f"{name}.txt").write_text(
            "\n".join(p.stem for p in files) + "\n", encoding="utf-8")
    print("đã ghi data/splits/{train,val}.txt và dataset/data.yaml")


if __name__ == "__main__":
    main()
