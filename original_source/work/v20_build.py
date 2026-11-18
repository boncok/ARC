# -*- coding: utf-8 -*-
"""v20 submission builder.

Post-processing locked by the honest-holdout sweep (work/v20_post.py on q_liteM + vc320):
  conf 0.0001, NMS IoU 0.70, max_det 100, min box side > 1 px, letterbox 256x320.
That combo scored 0.9830 (A 0.9808 / B 0.9851) vs 0.98217 for the v19 pipeline
(conf 0.0005 / IoU 0.65), same sign on both halves.

Unlike 09_build_sub_final.py, the prediction cache stores a sha256 of the weights plus
the inference config, and refuses to reuse a cache produced by a different model.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(r"F:/tele/edge-ai-challenge-2026-lan-2/FINAL_VONG_LOAI")
sys.path.insert(0, str(OUT / "solution/lib"))
sys.path.insert(0, str(OUT / "work"))
import lite_head  # noqa: F401  (unpickle DetectLite)
import xpu_patch  # noqa: F401
import detlib

W, H = 320, 240
CONF = 0.0001
IOU = 0.70
MAX_DET = 100
MIN_SIDE = 1


def predict_test(weights, cache_path):
    sig = hashlib.sha256(Path(weights).read_bytes()).hexdigest()[:16]
    cfg = f"{sig}|imgsz320|conf0.0001|iou0.9|maxdet300"
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        if str(z["cfg"]) != cfg:
            raise SystemExit(f"cache {cache_path} was built from a different model/config "
                             f"({z['cfg']} != {cfg}); delete it or pass a new --cache")
        names = [str(n) for n in z["names"]]
        return {n: {"boxes": z[f"{n}|b"], "scores": z[f"{n}|s"], "cls": z[f"{n}|c"]} for n in names}
    pred = detlib.predict_dir(weights, OUT / "data/test/images", imgsz=320, conf=CONF, iou=0.9,
                              tta=False, max_det=300)
    names = sorted(pred)
    np.savez_compressed(cache_path, cfg=np.array(cfg), names=np.array(names),
                        **{f"{n}|b": pred[n]["boxes"] for n in names},
                        **{f"{n}|s": pred[n]["scores"] for n in names},
                        **{f"{n}|c": pred[n]["cls"] for n in names})
    return pred


def apply_global(pred):
    out = {}
    for name, d in pred.items():
        chunks = []
        for c in range(7):
            mk = (d["cls"] == c) & (d["scores"] >= CONF)
            if not mk.any():
                continue
            r = detlib.nms(d["boxes"][mk], d["scores"][mk], np.full(int(mk.sum()), c), iou_thr=IOU)
            if len(r["scores"]):
                bb = r["boxes"].copy()
                bb[:, [0, 2]] = bb[:, [0, 2]].clip(0, W)
                bb[:, [1, 3]] = bb[:, [1, 3]].clip(0, H)
                ok = (bb[:, 2] - bb[:, 0] > MIN_SIDE) & (bb[:, 3] - bb[:, 1] > MIN_SIDE)
                if ok.any():
                    chunks.append((bb[ok], r["scores"][ok], np.full(int(ok.sum()), c)))
        if not chunks:
            out[name] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, int)}
            continue
        bb = np.concatenate([k[0] for k in chunks])
        ss = np.concatenate([k[1] for k in chunks])
        cc = np.concatenate([k[2] for k in chunks])
        if len(ss) > MAX_DET:
            o = np.argsort(-ss)[:MAX_DET]
            bb, ss, cc = bb[o], ss[o], cc[o]
        out[name] = {"boxes": bb, "scores": ss, "cls": cc}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()

    pred = predict_test(a.weights, Path(a.cache))
    print(f"raw: {sum(len(v['scores']) for v in pred.values())} dets / {len(pred)} imgs")
    post = apply_global(pred)
    n = sum(len(v["scores"]) for v in post.values())
    print(f"post: {n} dets ({n / len(post):.1f}/img) @ conf={CONF} iou={IOU} max_det={MAX_DET}")

    intent = {}
    for row in csv.DictReader(open(OUT / "results/intent_test_predictions.csv", encoding="utf-8")):
        intent[row["question_id"]] = row["intent"]

    sample = list(csv.DictReader(open(OUT / "data/sample_submission.csv", encoding="utf-8")))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["task_type", "sample_id", "prediction"])
        for row in sample:
            sid = row["sample_id"]
            if row["task_type"] == "DETECTION":
                d = post.get(sid + ".png") or post.get(sid)
                payload = []
                if d is not None and len(d["scores"]):
                    for b, s, c in zip(d["boxes"], d["scores"], d["cls"]):
                        payload.append({"class_id": int(c), "score": round(float(s), 5),
                                        "bbox": [round(float(b[0]), 2), round(float(b[1]), 2),
                                                 round(float(b[2]), 2), round(float(b[3]), 2)]})
                wr.writerow(["DETECTION", sid, json.dumps(payload, separators=(",", ":"))])
            else:
                wr.writerow(["INTENT", sid, intent[sid]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()