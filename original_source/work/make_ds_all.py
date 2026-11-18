import os, shutil
from pathlib import Path
SRC = Path(r"F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI/work/ds_local")
DST = Path(r"F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI/work/ds_all")
for sub in ["images/train", "images/mon", "labels/train", "labels/mon"]:
    (DST / sub).mkdir(parents=True, exist_ok=True)
n = 0
for split in ["train", "vc"]:
    for f in (SRC / f"images/{split}").iterdir():
        if f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        dst = DST / "images/train" / f.name
        if not dst.exists():
            try:
                os.link(f, dst)
            except OSError:
                shutil.copy2(f, dst)
            n += 1
        lab = SRC / f"labels/{split}" / (f.stem + ".txt")
        dl = DST / "labels/train" / (f.stem + ".txt")
        if lab.exists() and not dl.exists():
            try:
                os.link(lab, dl)
            except OSError:
                shutil.copy2(lab, dl)
for f in (SRC / "images/mon").iterdir():
    if f.suffix.lower() in (".png", ".jpg", ".jpeg"):
        d = DST / "images/mon" / f.name
        if not d.exists():
            try: os.link(f, d)
            except OSError: shutil.copy2(f, d)
        lab = SRC / "labels/mon" / (f.stem + ".txt")
        if lab.exists() and not (DST / "labels/mon" / lab.name).exists():
            try: os.link(lab, DST / "labels/mon" / lab.name)
            except OSError: shutil.copy2(lab, DST / "labels/mon" / lab.name)
print("linked", n)
y = """path: F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI/work/ds_all
train: images/train
val: images/mon
names:
  0: bottle
  1: cup
  2: phone
  3: book
  4: mouse
  5: box
  6: toy_car
"""
(DST / "data.yaml").write_text(y)
print("train imgs", len(list((DST/"images/train").iterdir())), "labels", len(list((DST/"labels/train").iterdir())))
