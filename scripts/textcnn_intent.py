# -*- coding: utf-8 -*-
"""TextCNN intent — kiến trúc slide Training 1 (embedding → Conv1D k=2,3,4 → global max
pool → concat → dense → softmax). 41.429 params.

LƯU Ý: model này dùng cho FIRMWARE ESP32-S3 (export TFLite), KHÔNG phải cột INTENT của
submission v22 (v22 dùng TF-IDF + OOV rule, xem intent_oov.py). Giữ trong gói để đầy đủ
nhánh Intent đã khai báo.

Chạy: python scripts/textcnn_intent.py
"""
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pkg_paths import DATA, WORK, default_device

torch.manual_seed(0)
np.random.seed(0)

INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]
MAX_LEN = 32
EMB = 48
N_FILTERS = 64
DEVICE = default_device()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def tokenize(q: str):
    return norm(q).split()[:MAX_LEN]


class IntentDS(Dataset):
    def __init__(self, rows, vocab=None, labels=None):
        self.tokens = [tokenize(r["question"]) for r in rows]
        if vocab is None:
            freq = {}
            for t in self.tokens:
                for w in t:
                    freq[w] = freq.get(w, 0) + 1
            vocab = {"<pad>": 0, "<unk>": 1}
            for w, c in sorted(freq.items(), key=lambda x: -x[1]):
                if c >= 2:
                    vocab[w] = len(vocab)
        self.vocab = vocab
        self.labels = labels
        self.y = ([INTENTS.index(r["intent"]) for r in rows] if labels is None else labels)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        ids = [self.vocab.get(w, 1) for w in self.tokens[i]][:MAX_LEN]
        ids += [0] * (MAX_LEN - len(ids))
        return torch.tensor(ids), torch.tensor(self.y[i])


class TextCNN(nn.Module):
    def __init__(self, vocab_size, n_classes=5, emb=EMB, nf=N_FILTERS, ks=(2, 3, 4), drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb, nf, k, padding=k // 2) for k in ks])
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(nf * len(ks), n_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                 # B, emb, L
        h = [torch.relu(c(e)).max(dim=2).values for c in self.convs]   # global max pool
        return self.fc(self.drop(torch.cat(h, dim=1)))


def run_epoch(model, loader, opt=None):
    train = opt is not None
    model.train(train)
    crit = nn.CrossEntropyLoss()
    tot = correct = 0
    loss_sum = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        out = model(xb)
        loss = crit(out, yb)
        if train:
            opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item() * len(yb)
        correct += (out.argmax(1) == yb).sum().item()
        tot += len(yb)
    return loss_sum / tot, correct / tot


def main():
    tr = list(csv.DictReader(open(DATA / "intent_train.csv", encoding="utf-8")))
    te = list(csv.DictReader(open(DATA / "intent_test.csv", encoding="utf-8")))
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(tr))
    n_val = 600
    val_rows = [tr[i] for i in idx[:n_val]]
    trn_rows = [tr[i] for i in idx[n_val:]]

    ds_trn = IntentDS(trn_rows)
    ds_val = IntentDS(val_rows, vocab=ds_trn.vocab)
    print(f"vocab {len(ds_trn.vocab)} | train {len(ds_trn)} | val {len(ds_val)} | device {DEVICE}")

    model = TextCNN(len(ds_trn.vocab)).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TextCNN params: {n_params:,}")

    dl_trn = DataLoader(ds_trn, batch_size=128, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=256)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    best_acc, best_state, patience = 0.0, None, 0
    for ep in range(1, 26):
        l_tr, a_tr = run_epoch(model, dl_trn, opt)
        l_va, a_va = run_epoch(model, dl_val)
        if a_va > best_acc:
            best_acc, best_state, patience = a_va, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
        if ep % 5 == 0 or ep == 1:
            print(f"  ep{ep:02d} loss {l_tr:.3f} acc {a_tr:.4f} | val loss {l_va:.3f} acc {a_va:.4f}")
        if patience >= 6:
            print("  early stop")
            break
    model.load_state_dict(best_state)
    print(f"val accuracy tốt nhất: {best_acc:.4f}")

    # dự đoán test
    ds_te = IntentDS(te, vocab=ds_trn.vocab, labels=[0] * len(te))
    dl_te = DataLoader(ds_te, batch_size=256)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in dl_te:
            preds += model(xb.to(DEVICE)).argmax(1).cpu().tolist()
    labels = [INTENTS[i] for i in preds]
    from collections import Counter
    print("test dist:", dict(Counter(labels)))

    out = WORK / "results_textcnn"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "vocab": ds_trn.vocab}, out / "textcnn.pt")
    with open(out / "intent_test_predictions_textcnn.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["question_id", "intent"])
        for r, lab in zip(te, labels):
            w.writerow([r["question_id"], lab])
    json.dump({"params": n_params, "val_acc": best_acc, "vocab": len(ds_trn.vocab)},
              open(out / "metrics.json", "w"), indent=1)
    print("saved", out)

    # export ONNX để sau này convert TFLite
    try:
        dummy = torch.zeros(1, MAX_LEN, dtype=torch.long, device=DEVICE)
        torch.onnx.export(model, dummy, str(out / "textcnn.onnx"), input_names=["input_ids"],
                          output_names=["logits"], dynamic_axes={"input_ids": {0: "batch"}},
                          opset_version=13)
        print("exported ONNX:", out / "textcnn.onnx")
    except Exception as e:
        print("ONNX export lỗi:", str(e)[:150])


if __name__ == "__main__":
    main()