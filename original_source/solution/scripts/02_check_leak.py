# -*- coding: utf-8 -*-
"""BƯỚC 02: kiểm tra rò rỉ train/val + tạo "val sạch".

Vì sao cần: ảnh trong bộ này là render tổng hợp, nhiều ảnh cùng cảnh chỉ khác góc nhìn
rất nhỏ. Nếu chia train/val ngẫu nhiên, val sẽ chứa ảnh gần-trùng với train ⇒ mAP val
bị "ảo". Script đo mức trùng bằng average-hash 8x8 (Hamming distance) và tạo tập
val sạch (không có ảnh nào gần-trùng với train) để báo cáo con số trung thực.

Chạy: python QUALIFIER_SOLUTION/scripts/02_check_leak.py
"""
from pathlib import Path

import numpy as np
from PIL import Image

SOL = Path(__file__).resolve().parents[1]
DS = SOL / "dataset"
THR = 6          # Hamming distance <= 6 coi là gần trùng


def ahash(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("L").resize((8, 8), Image.BILINEAR), np.float32)
    return (a > a.mean()).flatten()


def main():
    train = sorted((DS / "images/train").glob("*.png"))
    val = sorted((DS / "images/val").glob("*.png"))
    tr = {p.stem: ahash(p) for p in train}
    print(f"train {len(train)} | val {len(val)}")

    dists, clean = [], []
    for p in val:
        h = ahash(p)
        d = min(int((h != t).sum()) for t in tr.values())
        dists.append(d)
        if d > THR:
            clean.append(p.name)   # ghi kèm .png để khớp key của ground truth
    d = np.array(dists)
    print("khoảng cách val→train: p10=%d p50=%d p90=%d" % tuple(np.percentile(d, [10, 50, 90])))
    print(f"val có gần-trùng (≤{THR}): {int((d <= THR).sum())}/{len(val)}")
    print(f"val sạch: {len(clean)}/{len(val)}")

    out = SOL / "data/val_clean.txt"
    out.write_text("\n".join(clean) + "\n", encoding="utf-8")
    print("đã ghi", out)
    print("→ hai script 04/05 dùng danh sách này với cờ --clean 1")


if __name__ == "__main__":
    main()
