"""Detection inference + fusion + submission writer for Edge AI Challenge 2026 LAN 2."""
import json
import sys
from pathlib import Path

import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK))
try:
    import xpu_patch  # noqa: F401  CPU NMS (much faster than device NMS on Intel Arc)
except Exception:
    pass


def predict_dir(weights, source, imgsz=1280, conf=0.001, iou=0.6, max_det=300, tta=False,
                device="xpu:0", batch=8):
    """Run one or more models over a directory; return {fname: {'boxes','scores','cls'}}."""
    from ultralytics import YOLO

    if isinstance(weights, (str, Path)):
        weights = [weights]
    acc = {}
    for w in weights:
        model = YOLO(str(w))
        for r in model.predict(source=str(source), imgsz=imgsz, conf=conf, iou=iou,
                               max_det=max_det, augment=tta, device=device,
                               stream=True, verbose=False, batch=batch, save=False):
            name = Path(r.path).name
            if r.boxes is None or len(r.boxes) == 0:
                det = acc.setdefault(name, {"boxes": [], "scores": [], "cls": [], "ntask": 0})
                det["ntask"] += 1
                continue
            xyxy = r.boxes.xyxy.cpu().numpy().astype(np.float64)
            sc = r.boxes.conf.cpu().numpy().astype(np.float64)
            cl = r.boxes.cls.cpu().numpy().astype(np.int64)
            det = acc.setdefault(name, {"boxes": [], "scores": [], "cls": [], "ntask": 0})
            det["boxes"].append(xyxy)
            det["scores"].append(sc)
            det["cls"].append(cl)
            det["ntask"] += 1
    out = {}
    for name, d in acc.items():
        if d["boxes"]:
            out[name] = {
                "boxes": np.concatenate(d["boxes"]),
                "scores": np.concatenate(d["scores"]),
                "cls": np.concatenate(d["cls"]),
            }
        else:
            out[name] = {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, dtype=int)}
    return out


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.clip(union, 1e-9, None)


def wbf(per_model, conf_thr=0.0, iou_thr=0.55, weights=None, n_models=None, iou_type="iou"):
    """Weighted Box Fusion of a list of {boxes,scores,cls} dicts -> fused dict."""
    boxes = [d["boxes"] for d in per_model if len(d["boxes"])]
    scores = [d["scores"] for d in per_model if len(d["boxes"])]
    clss = [d["cls"] for d in per_model if len(d["boxes"])]
    nm = len(per_model) if n_models is None else n_models
    if weights is None:
        weights = np.ones(len(boxes))
    weights = np.asarray(weights, dtype=np.float64)[: len(boxes)]

    if not boxes:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, dtype=int)}

    b = np.concatenate(boxes)
    s = np.concatenate(scores)
    c = np.concatenate(clss)
    src = np.concatenate([np.full(len(x), i) for i, x in enumerate(boxes)])
    keep = s >= conf_thr
    b, s, c, src = b[keep], s[keep], c[keep], src[keep]

    out_b, out_s, out_c = [], [], []
    for cls in np.unique(c):
        idx = np.where(c == cls)[0]
        order = idx[np.argsort(-s[idx])]
        bb, ss, sr = b[order], s[order], src[order]
        used = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            if used[i]:
                continue
            ov = iou_matrix(bb[i:i + 1], bb)[0]
            # cluster = same model? no: any overlapping, but limit per-model duplicates
            cand = (ov >= iou_thr) & (~used)
            cand[i] = True
            # keep at most one box per source model in a cluster (the best scored)
            sel_list, seen = [], set()
            for j in np.where(cand)[0]:
                if sr[j] in seen:
                    continue
                seen.add(sr[j])
                sel_list.append(j)
            sel = np.array(sel_list)
            used[sel] = True
            w = ss[sel] * weights[sr[sel]]
            w = w / np.clip(w.sum(), 1e-9, None)
            fused = (bb[sel] * w[:, None]).sum(0)
            # confidence: mean of member scores, discounted when fewer models agree
            sc = float((ss[sel] * weights[sr[sel]]).sum() / weights[sr[sel]].sum())
            agree = len(sel)
            out_b.append(fused)
            out_s.append(sc * (agree / max(nm, 1)) ** 0.0)  # keep raw mean
            out_c.append(int(cls))
            # mark the other members as used too
            for j in np.where(ov >= iou_thr)[0]:
                used[j] = True
    return {"boxes": np.array(out_b), "scores": np.array(out_s), "cls": np.array(out_c, dtype=int)}


def nms(boxes, scores, cls, iou_thr=0.55):
    out_b, out_s, out_c = [], [], []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        order = idx[np.argsort(-scores[idx])]
        bb = boxes[order]
        ss = scores[order]
        used = np.zeros(len(order), dtype=bool)
        for i in range(len(order)):
            if used[i]:
                continue
            used[i] = True
            ov = iou_matrix(bb[i:i + 1], bb)[0]
            used |= ov >= iou_thr
            used[i] = True
            out_b.append(bb[i])
            out_s.append(ss[i])
            out_c.append(int(c))
    if not out_b:
        return {"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "cls": np.zeros(0, dtype=int)}
    b = np.array(out_b)
    s = np.array(out_s)
    o = np.argsort(-s)
    return {"boxes": b[o], "scores": s[o], "cls": np.array(out_c)[o]}


def to_submission(pred, image_names, conf=0.001, max_det=100, bbox_fmt="xyxy"):
    """pred: {fname: {boxes(xyxy px), scores, cls}} -> CSV-ready rows."""
    rows = []
    for name in image_names:
        d = pred.get(name)
        boxes = [] if d is None else d["boxes"]
        scores = [] if d is None else d["scores"]
        cls = [] if d is None else d["cls"]
        if len(boxes):
            order = np.argsort(-scores)[:max_det]
            for i in order:
                b = boxes[i]
                x1, y1, x2, y2 = (float(max(0, b[0])), float(max(0, b[1])),
                                  float(b[2]), float(b[3]))
                if x2 <= x1 or y2 <= y1:
                    continue
                if bbox_fmt == "xywh":
                    bw, bh = x2 - x1, y2 - y1
                    bb = [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)]
                else:
                    bb = [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                boxes.append({"class_id": int(cls[i]), "score": round(float(scores[i]), 5),
                              "bbox": bb})
        rows.append((name, boxes))
    return rows
