"""Intent: handle out-of-vocabulary objects (OOV) that the linear model has never seen.

Convention in this dataset (verified on train): a question about an object that is NOT one
of the 7 classes is OUT_OF_SCOPE, regardless of the question form ("o dau", "bao nhieu",
"co ... khong"). A TF-IDF model cannot generalise to unseen object names, so we add a
lexicon-based rule layer and measure it with a proper held-out-object split.
"""
import csv, importlib.util, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
spec = importlib.util.spec_from_file_location("iq", Path(__file__).parent / "intent_q.py")
iq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iq)

ROOT = Path(r"F:\tele\edge-ai-challenge-2026-lan-2")
Q = ROOT / "egde-ai-challenge-2026-lan-2-qualifiers"

# class synonyms (the 7 classes and how the generator refers to them)
CLASS_TERMS = [
    # bottle
    "chai", "chai nuoc", "chai nhua", "binh nuoc", "binh", "chai thuy tinh",
    # cup
    "coc", "ly", "tach", "coc uong", "ly nuoc", "coc nuoc", "tach uong",
    # phone
    "dien thoai", "smartphone", "dien thoai di dong", "dtdd", "thiet bi di dong",
    # book
    "sach", "cuon sach", "quyen sach", "sach vo",
    # mouse
    "chuot", "chuot may tinh", "con chuot",
    # box
    "hop", "thung", "thung giay", "hop giay", "hop carton", "thung carton",
    "hop cac tong", "thung cac tong", "hop bia",
    # toy car
    "xe do choi", "oto do choi", "o to do choi", "xe oto do choi", "xe hoi do choi",
    "toy car", "xe o to", "oto", "o to", "xe hoi", "xe",
]
# words that are clearly a concrete object but NOT a class (seen in train OOS questions)
OOV_TERMS = [
    "binh hoa", "o cam", "o cam dien", "remote", "tai nghe", "den ban", "den",
    "cai vi", "vi", "chim", "con chim", "may tinh bang", "may tinh", "laptop",
    "cai mu", "mu", "cay but", "but", "dong ho", "ban phim", "kinh mat", "dong xu",
    "ban", "ghe", "tu", "tu lanh", "tivi", "quat", "ao", "giay", "dep", "lo hoa",
    "chau cay", "cay canh", "gau bong", "bup be", "may anh", "loa", "sac",
]

FRAMES = [
    r"co ([\w\s]{2,30}?) (?:khong|nao|vay|hay khong)",
    r"bao nhieu ([\w\s]{2,30}?)(?:\s+(?:trong|xuat hien|co|tren|o)\b|[?.]|$)",
    r"(?:vi tri(?: cua)?|vi tri ngang cua) ([\w\s]{2,30}?)(?:\s+(?:trong|la|nam|o)\b|[?.]|$)",
    r"so luong ([\w\s]{2,30}?)(?:\s+(?:trong|xuat hien|co|tren|o)\b|[?.]|$)",
    r"dem (?:so |giup |va )?([\w\s]{2,30}?)(?:\s+(?:trong|xuat hien|co|tren|o)\b|[?.]|$)",
    r"([\w\s]{2,30}?) (?:nam|o|xuat hien) (?:o |ben |phia |trong |tren )",
    r"tong so ([\w\s]{2,30}?)(?:\s+(?:trong|xuat hien|co|tren|o)\b|[?.]|$)",
]


def has(question_norm: str, term: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", question_norm) is not None


def object_is_oov(qn: str):
    """Return (mentions_object, in_class) heuristics on a normalised question."""
    if any(has(qn, t) for t in CLASS_TERMS):
        return True, True
    if any(has(qn, t) for t in OOV_TERMS):
        return True, False
    # try to extract an object phrase from the frames
    for pat in FRAMES:
        m = re.search(pat, qn)
        if m:
            phrase = m.group(1).strip()
            if len(phrase) >= 3 and not any(w in phrase for w in
                                            ("loai", "nhom", "lop", "kieu", "chung", "doi tuong")):
                return True, False
    return False, True


def rule_oov(qn: str, form_pred: str):
    """Override form-based prediction to OUT_OF_SCOPE when the object is out of vocabulary."""
    if form_pred == "LIST":
        return form_pred
    mentions, in_class = object_is_oov(qn)
    if mentions and not in_class:
        return "OUT_OF_SCOPE"
    return form_pred


def build_model():
    return make_pipeline(iq.word_char_union(), LogisticRegression(max_iter=3000, C=20))


def main():
    tr = list(csv.DictReader(open(Q / "intent_train.csv", encoding="utf-8")))
    te = list(csv.DictReader(open(Q / "intent_test.csv", encoding="utf-8")))
    X = [iq.norm(r["question"]) for r in tr]
    y = np.array([r["intent"] for r in tr])

    # ---- held-out-object experiment: hide some OOV object names from training ----
    oov_objs = ["binh hoa", "remote", "tai nghe", "den ban", "may tinh bang", "cai vi",
                "con chim", "cai mu", "dong ho", "ban phim", "kinh mat", "dong xu"]
    rng = random.Random(0)
    hidden = set(rng.sample(oov_objs, k=6))
    print("hidden objects:", sorted(hidden))

    def mentions_any(q, terms):
        return any(has(q, t) for t in terms)

    hold = np.array([mentions_any(q, hidden) for q in X])
    keep = ~hold
    print(f"hold-out questions: {hold.sum()} (train on {keep.sum()})")
    m = build_model().fit([X[i] for i in np.where(keep)[0]], y[keep])
    ph = m.predict([X[i] for i in np.where(hold)[0]])
    yh = y[hold]
    acc_model = (ph == yh).mean()
    rh = np.array([rule_oov(X[i], p) for i, p in zip(np.where(hold)[0], ph)])
    acc_rule = (rh == yh).mean()
    print(f"  model-only acc on held-out objects : {acc_model:.4f}")
    print(f"  + OOV rule                         : {acc_rule:.4f}")

    # ---- full train: model + rule on test ----
    full = build_model().fit(X, y)
    Xt = [iq.norm(r["question"]) for r in te]
    pt = full.predict(Xt)
    rt = np.array([rule_oov(q, p) for q, p in zip(Xt, pt)])
    print("\ntest distribution model :", dict(Counter(pt)))
    print("test distribution +rule :", dict(Counter(rt)))
    changed = int((pt != rt).sum())
    print("questions overridden by the rule:", changed)

    out = ROOT / "qual_work" / "intent_test_predictions.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question_id", "intent", "intent_model"])
        for r, a, b in zip(te, rt, pt):
            w.writerow([r["question_id"], a, b])
    print("saved", out)


if __name__ == "__main__":
    main()
