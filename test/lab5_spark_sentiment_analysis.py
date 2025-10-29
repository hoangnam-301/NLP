# test/lab5_improvement_test.py
import os, sys, re, csv, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Iterable, Tuple
import numpy as np
from collections import Counter, defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from src.preprocessing.regex_tokenizer import RegexTokenizer
from src.representations.count_vectorizer import CountVectorizer

# Nếu bạn có TF-IDF trong src, sẽ tự dùng được
try:
    from src.representations.tfidf_vectorizer import TfidfVectorizer
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False

# ============================== CONFIG ==============================
DATA_PATH = "data/sentiments.csv"   # Đường dẫn dataset (cùng file Spark dùng)
USE_TFIDF = False                   # True để dùng TF-IDF (nếu bạn có lớp TfidfVectorizer)
NGRAM_MODE = "unigram"              # "unigram" hoặc "bigram_neg"
K_FOLDS = 5
MIN_DF = 1                          # Với tập nhỏ, nên để 1 để tránh lọc quá tay
# ====================================================================

NEGATORS = {"not", "no", "never", "cannot", "can't", "dont", "don't"}

# --------------------------- Tokenizers ----------------------------
class UnigramTokenizer(RegexTokenizer):
    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[A-Za-z0-9']+", text.lower())

class BigramNegTokenizer(RegexTokenizer):
    def tokenize(self, text: str) -> List[str]:
        toks = re.findall(r"[A-Za-z0-9']+", text.lower())
        if not toks:
            return []
        bigrams = [f"{toks[i]}_{toks[i+1]}" for i in range(len(toks)-1)]
        neg_terms = []
        for i, t in enumerate(toks[:-1]):
            if t in NEGATORS:
                neg_terms.append(f"{t}_{toks[i+1]}")
        return toks + bigrams + neg_terms

# ----------------------------- Data IO -----------------------------
def _label_to_int(y_raw: str) -> int:
    ys = str(y_raw).strip().lower()
    if ys in ("1", "pos", "positive", "true", "yes"):
        return 1
    if ys in ("0", "neg", "negative", "false", "no"):
        return 0
    return int(ys)  # cố gắng ép int lần cuối

def _load_csv_like(path: str) -> Tuple[list, np.ndarray]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        first_line = head.splitlines()[0] if head else ""
        looks_like_header = "text" in first_line.lower()  # đoán có header

        if looks_like_header:
            reader = csv.DictReader(f)
            flds = [c.lower() for c in (reader.fieldnames or [])]
            if "text" not in flds or not (("label" in flds) or ("sentiment" in flds)):
                raise ValueError("CSV cần header có cột 'text' và 'label' hoặc 'sentiment'.")

            # tìm tên cột đúng (giữ nguyên hoa/thường trong file)
            text_col_real = next(c for c in reader.fieldnames if c.lower() == "text")
            lbl_name = "label" if "label" in flds else "sentiment"
            label_col_real = next(c for c in reader.fieldnames if c.lower() == lbl_name)

            for row in reader:
                t = (row.get(text_col_real, "") or "").strip()
                y = row.get(label_col_real, None)
                if t == "" or y is None or str(y).strip() == "":
                    continue
                labels.append(_label_to_int(y))
                texts.append(t)
        else:
            # Không header: đọc 2 cột text,label
            reader = csv.reader(f)
            for row in reader:
                if not row or len(row) < 2:
                    continue
                t = (row[0] or "").strip()
                y = (row[1] or "").strip()
                if t == "" or y == "":
                    continue
                labels.append(_label_to_int(y))
                texts.append(t)

    if not texts:
        raise ValueError("CSV rỗng/không hợp lệ. Cần 2 cột: text + label/sentiment (có/không header).")
    print(f"Loaded {len(texts)} samples from {path}")
    return texts, np.array(labels, dtype=int)

def _load_json_like(path: str) -> Tuple[list, np.ndarray]:
    texts, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(2048)
        f.seek(0)
        # JSON Lines?
        if "\n" in head.strip():
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                t = str(obj.get("text", "")).strip()
                y = obj.get("label", obj.get("sentiment", None))
                if t != "" and y is not None:
                    labels.append(_label_to_int(y))
                    texts.append(t)
        else:
            # JSON array
            arr = json.load(f)
            for obj in arr:
                t = str(obj.get("text", "")).strip()
                y = obj.get("label", obj.get("sentiment", None))
                if t != "" and y is not None:
                    labels.append(_label_to_int(y))
                    texts.append(t)
    if not texts:
        raise ValueError("JSON/JSONL rỗng/không hợp lệ. Cần key: text + (label|sentiment).")
    print(f"Loaded {len(texts)} samples from {path}")
    return texts, np.array(labels, dtype=int)

def load_dataset(path: str) -> Tuple[list, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    ext = os.path.splitext(path.lower())[1]
    if ext in [".csv", ".tsv", ""]:
        return _load_csv_like(path)
    if ext in [".json", ".jsonl"]:
        return _load_json_like(path)
    # Mặc định coi là CSV
    return _load_csv_like(path)

# ---------------------------- Helpers ------------------------------
def build_token_filter(base_tok: RegexTokenizer, train_texts: Iterable[str], min_df: int = 1):
    df = Counter()
    for s in train_texts:
        df.update(set(base_tok.tokenize(s)))
    keep = {tok for tok, v in df.items() if v >= min_df}
    class FilteredTokenizer(RegexTokenizer):
        def tokenize(self, text: str) -> List[str]:
            toks = base_tok.tokenize(text)
            return toks if not keep else [t for t in toks if t in keep]
    return FilteredTokenizer()

def metrics(y_true, y_pred):
    return dict(
        acc = accuracy_score(y_true, y_pred),
        prec= precision_score(y_true, y_pred, zero_division=0),
        rec = recall_score(y_true, y_pred, zero_division=0),
        f1  = f1_score(y_true, y_pred, zero_division=0),
    )

def avg(ms):
    out = defaultdict(float)
    for m in ms:
        for k, v in m.items():
            out[k] += v
    n = max(len(ms), 1)
    return {k: out[k] / n for k in out}

def run_cv(name: str, make_model, base_tokenizer: RegexTokenizer, texts: list, labels: np.ndarray):
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    allm = []
    for fold, (tr, te) in enumerate(skf.split(texts, labels), 1):
        tr_txt = [texts[i] for i in tr]
        te_txt = [texts[i] for i in te]
        y_tr = labels[tr]
        y_te = labels[te]

        tok = build_token_filter(base_tokenizer, tr_txt, min_df=MIN_DF)
        vec = TfidfVectorizer(tok) if (USE_TFIDF and HAS_TFIDF) else CountVectorizer(tok)

        X_tr = np.array(vec.fit_transform(tr_txt), dtype=float)
        X_te = np.array(vec.transform(te_txt), dtype=float)

        model = make_model().fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        m = metrics(y_te, y_pred)
        allm.append(m)
        print(f"[{name}] Fold {fold}: acc={m['acc']:.3f} prec={m['prec']:.3f} rec={m['rec']:.3f} f1={m['f1']:.3f}")

    mean = avg(allm)
    vec_name = "TF-IDF" if (USE_TFIDF and HAS_TFIDF) else "Count"
    print(f"\n>>> {name} AVG ({K_FOLDS}-fold, {vec_name}, {NGRAM_MODE}): "
          f"acc={mean['acc']:.3f}, prec={mean['prec']:.3f}, rec={mean['rec']:.3f}, f1={mean['f1']:.3f}\n")

# ------------------------------ Main --------------------------------
if __name__ == "__main__":
    texts, labels = load_dataset(DATA_PATH)
    tokenizer = UnigramTokenizer() if NGRAM_MODE == "unigram" else BigramNegTokenizer()

    # Naive Bayes
    run_cv("MultinomialNB(alpha=1.0)",
           lambda: MultinomialNB(alpha=1.0),
           tokenizer, texts, labels)

    # Logistic Regression
    run_cv("LogReg(liblinear, max_iter=1000)",
           lambda: LogisticRegression(solver="liblinear", max_iter=1000),
           tokenizer, texts, labels)
