# -*- coding: utf-8 -*-
"""Đường dẫn dùng chung cho mọi script trong gói.

PKG  = thư mục gói (submission_v22/)
DATA = dữ liệu BTC (train/, test/, intent_*.csv, sample_submission.csv, classes.txt).
       Mặc định: thư mục `data/` nằm ngang hàng với gói; đổi bằng biến môi trường LAN2_DATA.
WORK = nơi sinh dataset dẫn xuất / runs / soups / cache. Mặc định PKG/work; đổi bằng LAN2_WORK.
"""
import os
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("LAN2_DATA", str(PKG.parent / "data")))
WORK = Path(os.environ.get("LAN2_WORK", str(PKG / "work")))


def default_device():
    import torch
    if torch.xpu.is_available():
        return "xpu:0"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"