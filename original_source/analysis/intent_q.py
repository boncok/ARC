"""Intent classifier for the qualifier round (6000 train / 4000 test, 5 balanced classes).

Vietnamese questions, heavily templated but paraphrased. We compare linear models on
word+char TF-IDF with 5-fold CV and keep the best; a rule layer is evaluated as well.
"""
import csv, re, unicodedata
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_union, make_pipeline, Pipeline
from sklearn.base import clone

ROOT = Path(r"F:\tele\edge-ai-challenge-2026-lan-2")
Q = ROOT / "egde-ai-challenge-2026-lan-2-qualifiers"
WORK = ROOT / "qual_work"
WORK.mkdir(exist_ok=True)
INTENTS = ["COUNT", "EXIST", "LIST", "LOCATION", "OUT_OF_SCOPE"]


def norm(s: str) -> str:
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.replace("đ", "d")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- rules
RULES = [
    ("OUT_OF_SCOPE", ["dung de", "lam bang", "viet bang tieng", "phong khach", "phong ngu",
                      "do sang", "gia bao nhieu", "bao nhieu tien", "mua o dau", "ai lam",
                      "hang gi", "chat lieu", "binh hoa", "may man", "thoi tiet", "hom nay",
                      "nang hay", "mua hay", "trong cay", "nau an", "an gi"]),
    ("LIST", ["lop doi tuong", "nhom doi tuong", "loai doi tuong", "loai vat the",
              "nhom vat the", "nhom mon do", "loai do dung", "chung loai", "kieu do dung",
              "nhom do vat", "so loai", "nhung loai", "cac loai", "bao nhieu loai",
              "bao nhieu nhom", "bao nhieu lop", "bao nhieu chung", "bao nhieu kieu",
              "cac nhom", "cac lop", "cac kieu"]),
    ("LOCATION", ["ben trai", "ben phai", "trai giua", "o giua", "o dau", "vi tri",
                  "chieu ngang", "phia nao", "nam o", "ben nao", "khung hinh la"]),
    ("COUNT", ["bao nhieu", "dem so", "dem giup", "dem va", "xin dem", "so luong",
               "may cai", "may chiec", "tong so", "dem xem"]),
    ("EXIST", ["co khong", "khong vay", "co nhin thay", "co thay", "co xuat hien",
               "co ... nao", "nao khong", "co phai la", "co ton tai", "co hay khong"]),
]


def rule_intent(q: str):
    qn = norm(q)
    for intent, kws in RULES:
        for kw in kws:
            if " " in kw:
                if kw in qn:
                    return intent, kw
            else:
                if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", qn):
                    return intent, kw
    return None, None


def word_char_union():
    return make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    )


def candidates():
    return {
        "logreg_C5": make_pipeline(word_char_union(), LogisticRegression(max_iter=3000, C=5)),
        "logreg_C20": make_pipeline(word_char_union(), LogisticRegression(max_iter=3000, C=20)),
        "svc": make_pipeline(word_char_union(), LinearSVC(C=1.0)),
        "svc_cal": make_pipeline(word_char_union(),
                                 CalibratedClassifierCV(LinearSVC(C=1.0), cv=3)),
        "cnb": make_pipeline(word_char_union(), ComplementNB(alpha=0.3)),
    }


def main():
    tr = list(csv.DictReader(open(Q / "intent_train.csv", encoding="utf-8")))
    te = list(csv.DictReader(open(Q / "intent_test.csv", encoding="utf-8")))
    X = [r["question"] for r in tr]
    y = np.array([r["intent"] for r in tr])
    Xt = [r["question"] for r in te]
    Xn = [norm(x) for x in X]
    Xtn = [norm(x) for x in Xt]
    print(f"train {len(X)} | test {len(Xt)} | classes {sorted(set(y))}")

    # rules only
    rp = [rule_intent(q)[0] for q in X]
    cov = np.mean([p is not None for p in rp])
    acc_rules = np.mean([p == t for p, t in zip(rp, y) if p is not None]) if cov else 0
    print(f"rules: coverage {cov:.1%}, accuracy(when covered) {acc_rules:.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    best_name, best_acc, best_model = None, -1, None
    for name, pipe in candidates().items():
        accs = []
        for tr_i, va_i in skf.split(Xn, y):
            m = clone(pipe)
            m.fit([Xn[i] for i in tr_i], y[tr_i])
            accs.append((m.predict([Xn[i] for i in va_i]) == y[va_i]).mean())
        acc = float(np.mean(accs))
        print(f"{name:12s} 5-fold acc = {acc:.4f} (+-{np.std(accs):.4f})")
        if acc > best_acc:
            best_name, best_acc, best_model = name, acc, pipe

    print(f"\nbest: {best_name} {best_acc:.4f}")
    model = clone(best_model)
    model.fit(Xn, y)
    pred = model.predict(Xtn)

    # hybrid: rule wins when it fires on OUT_OF_SCOPE/LIST (highest precision patterns)
    hybrid = []
    n_rule = 0
    for q, pm in zip(Xtn, pred):
        ri, kw = rule_intent(q)
        if ri in ("OUT_OF_SCOPE", "LIST"):
            hybrid.append(ri); n_rule += 1
        else:
            hybrid.append(pm)
    print(f"hybrid used rules on {n_rule}/{len(Xtn)} test questions")
    from collections import Counter
    print("test intent distribution (model):", dict(Counter(pred)))
    print("test intent distribution (hybrid):", dict(Counter(hybrid)))

    out = WORK / "intent_test_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "intent", "intent_hybrid"])
        for r, p, h in zip(te, pred, hybrid):
            w.writerow([r["question_id"], p, h])
    print("saved", out)


if __name__ == "__main__":
    main()
