# -*- coding: utf-8 -*-
"""Strict submission validator (biên chính xác, không dung sai).

Checks: row count/order vs sample_submission, task types, duplicate ids, JSON shape,
class_id range, score range, exact bbox bounds, degenerate boxes, per-image cap,
empty images, intent whitelist.
Usage: python scripts/check_submission.py --path work/submission.csv
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import DATA
INTENTS = {"COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"}

ap = argparse.ArgumentParser()
ap.add_argument("--path", required=True)
ap.add_argument("--max-det", type=int, default=100)
a = ap.parse_args()
csv.field_size_limit(10 ** 9)

rows = list(csv.DictReader(open(a.path, encoding="utf-8")))
sample = list(csv.DictReader(open(DATA / "sample_submission.csv", encoding="utf-8")))
errs = []
if len(rows) != len(sample):
    errs.append(f"row count {len(rows)} != sample {len(sample)}")
if [r["sample_id"] for r in rows] != [r["sample_id"] for r in sample]:
    errs.append("sample_id order differs from sample_submission.csv")
if [r["task_type"] for r in rows] != [r["task_type"] for r in sample]:
    errs.append("task_type order differs from sample_submission.csv")
ids = collections.Counter(r["sample_id"] for r in rows)
dups = [k for k, v in ids.items() if v > 1]
if dups:
    errs.append(f"duplicate sample_id: {dups[:5]}")

nb = n50 = oob = empty = mx = degen = 0
cls = collections.Counter()
intent = collections.Counter()
for r in rows:
    if r["task_type"] == "DETECTION":
        p = json.loads(r["prediction"])
        if not isinstance(p, list):
            errs.append(f"{r['sample_id']}: prediction not a list")
            continue
        nb += len(p)
        mx = max(mx, len(p))
        if len(p) > a.max_det:
            errs.append(f"{r['sample_id']}: {len(p)} boxes > cap {a.max_det}")
        if not p:
            empty += 1
        for d in p:
            if set(d) != {"class_id", "score", "bbox"}:
                errs.append(f"{r['sample_id']}: bad keys {set(d)}")
            c, s, b = d["class_id"], d["score"], d["bbox"]
            if not (0 <= c <= 6):
                errs.append(f"{r['sample_id']}: class_id {c} out of range")
            if not (0.0 <= s <= 1.0):
                errs.append(f"{r['sample_id']}: score {s} out of [0,1]")
            if len(b) != 4:
                errs.append(f"{r['sample_id']}: bbox len {len(b)}")
                continue
            x1, y1, x2, y2 = b
            if x1 < 0 or y1 < 0 or x2 > 320 or y2 > 240:
                oob += 1
            if x2 <= x1 or y2 <= y1:
                degen += 1
            if s >= 0.5:
                n50 += 1
            cls[c] += 1
    elif r["task_type"] == "INTENT":
        if r["prediction"] not in INTENTS:
            errs.append(f"{r['sample_id']}: bad intent {r['prediction']}")
        intent[r["prediction"]] += 1
    else:
        errs.append(f"{r['sample_id']}: unknown task_type {r['task_type']}")

print("rows", len(rows), "det", sum(1 for r in rows if r["task_type"] == "DETECTION"),
      "intent", sum(1 for r in rows if r["task_type"] == "INTENT"))
print("boxes", nb, ">=0.5", n50, "oob", oob, "degenerate", degen, "empty_img", empty, "max_per_img", mx)
print("intent dist", dict(intent))
print("class dist", dict(sorted(cls.items())))
if errs:
    print("FAIL", len(errs), "errors; first 10:")
    for e in errs[:10]:
        print("  -", e)
    raise SystemExit(1)
print("PASS: all structural checks ok")