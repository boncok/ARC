# -*- coding: utf-8 -*-
"""Dựng dataset dẫn xuất ds_local / ds_all từ dữ liệu BTC (chỉ ĐỌC data/, không sửa).

ds_local: train = 6.000 - 320 (val_clean) = 5.680 ; val = vc (320, held-out thật) ; mon = 92 ⊂ train.
ds_all  : train = 6.000 (train + vc) ; val = mon (92, leak có chủ đích — chỉ dùng cho polish cuối).

Manifests đóng gói sẵn: manifests/val_clean.txt (320 tên file), manifests/mon.txt (92 tên file).
Chạy: python scripts/build_datasets.py   (env LAN2_DATA / LAN2_WORK nếu cần đổi chỗ)
"""
import os
import shutil
from pathlib import Path

from pkg_paths import PKG, DATA, WORK

NAMES = ["bottle", "cup", "phone", "book", "mouse", "box", "toy_car"]


def link(src: Path, dst: Path):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    imgs = sorted((DATA / "train/images").glob("*.png"))
    assert len(imgs) == 6000, f"train phải 6.000 ảnh, thấy {len(imgs)}"
    vc = {l.strip() for l in (PKG / "manifests/val_clean.txt").read_text().splitlines() if l.strip()}
    mon = {l.strip() for l in (PKG / "manifests/mon.txt").read_text().splitlines() if l.strip()}
    assert len(vc) == 320 and len(mon) == 92
    all_names = {p.name for p in imgs}
    assert vc <= all_names and mon <= all_names and not (vc & mon)

    for ds, splits in (("ds_local", {"train": all_names - vc, "vc": vc, "mon": mon}),
                       ("ds_all", {"train": all_names, "mon": mon})):
        root = WORK / ds
        for split, names in splits.items():
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            for n in sorted(names):
                link(DATA / "train/images" / n, root / "images" / split / n)
                link(DATA / "train/labels" / (Path(n).stem + ".txt"),
                     root / "labels" / split / (Path(n).stem + ".txt"))
        val = "vc" if ds == "ds_local" else "mon"
        yaml = [f"path: {root.as_posix()}", "train: images/train", f"val: images/{val}", "names:"]
        yaml += [f"  {i}: {c}" for i, c in enumerate(NAMES)]
        (root / "data.yaml").write_text("\n".join(yaml) + "\n", encoding="utf-8")
        print(ds, {s: len(n) for s, n in splits.items()}, "val =", val)
    print("OK")


if __name__ == "__main__":
    main()