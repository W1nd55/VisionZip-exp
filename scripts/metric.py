import string
import math
from typing import List, Dict, Any, Optional
from scripts.abstract import BaseMetric, Sample
from collections import defaultdict, Counter
import re

_YESNO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)

# ================= #
# Metric Implement  #
# ================= #

def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()

def _normalize_caption(s: str) -> str:
    s = (s or "").lower()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s

def percentile(arr: List[float], p: float) -> float:
    arr = sorted(arr)
    if not arr:
        return 0.0
    k = (len(arr)-1) * (p/100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return arr[int(k)]
    d0 = arr[f]*(c-k)
    d1 = arr[c]*(k-f)
    return d0 + d1

class ExactMatch(BaseMetric):
    def __init__(self):
        self.n = 0
        self.c = 0
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers: 
            return
        self.n += 1
        p = _normalize_text(pred_text)
        golds = {_normalize_text(a) for a in sample.answers}
        if p in golds:
            self.c += 1
    def compute(self) -> Dict[str, Any]:
        return {"exact_match": self.c / self.n if self.n else 0.0, "count": self.n}

class VQASoftAcc(BaseMetric):
    """Simplified VQA accuracy: acc = min(# of humans that said pred / 3, 1)."""
    def __init__(self):
        self.n = 0
        self.sum_acc = 0.0
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers: 
            return
        self.n += 1
        p = _normalize_text(pred_text)
        answers = [_normalize_text(a) for a in sample.answers]
        cnt = sum(1 for a in answers if a == p)
        self.sum_acc += min(cnt / 3.0, 1.0)
    def compute(self) -> Dict[str, Any]:
        return {"vqa_soft_acc": self.sum_acc / self.n if self.n else 0.0, "count": self.n}

class DelayStats(BaseMetric):
    """Collects latency distribution per stage and throughput."""
    def __init__(self):
        self.logs = []  # list of timings_ms with extra fields: num_new_tokens
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        self.logs.append(timings_ms.copy())
    def compute(self) -> Dict[str, Any]:
        if not self.logs:
            return {}
        keys = sorted({k for x in self.logs for k in x.keys()})
        out = {}
        for k in keys:
            arr = [x[k] for x in self.logs if k in x]
            if not arr: 
                continue
            base = k[:-3] if k.endswith("_ms") else k
            out[f"{base}_ms_p50"] = percentile(arr, 50)
            out[f"{base}_ms_p95"] = percentile(arr, 95)
            out[f"{base}_ms_avg"] = sum(arr)/len(arr)
 
         # 1) e2e_avg(ms)
        if "end2end_ms_avg" in out:
            out["e2e_avg(ms)"] = out["end2end_ms_avg"]
        elif "e2e_ms_avg" in out:
            out["e2e_avg(ms)"] = out["e2e_ms_avg"]
 
        # tok/s
        if any('num_new_tokens' in x for x in self.logs) and any(('decode_ms' in x) or ('decode' in x) for x in self.logs):
            toks = sum(x.get('num_new_tokens', 0) for x in self.logs)
            decode_ms = out.get("e2e_avg(ms)", 0.0)
            tps = toks / (decode_ms/1000.0 + 1e-9)
            out["decode_tok_per_s"] = tps
        return out
    
def _yn(s: str):
    if not s:
        return None
    m = _YESNO_RE.search(s)
    if not m:
        return None
    return m.group(1).lower()

# --- MME Accuracy ---
class MMEAcc(BaseMetric):
    """Per-question accuracy (ACC)"""
    def __init__(self):
        self.n = 0
        self.c = 0
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        gold = _normalize_text(sample.answers[0]) if sample.answers else None
        if gold not in ("yes","no"):
            raise ValueError(f"MMEAcc expected 'yes'/'no' answers, got: {gold}")
        pred = _yn(pred_text)
        self.n += 1
        if pred == gold:
            self.c += 1
    def compute(self) -> Dict[str, Any]:
        return {"mme_acc": self.c / self.n if self.n else 0.0, "mme_count_q": self.n}

# --- MME Acc Plus ---
class MMEAccPlus(BaseMetric):
    """Per-image paired accuracy (ACC+): counts 1 only if both positive and negative questions for the same image_id are correct."""
    def __init__(self):
        self.pairs = defaultdict(lambda: {"pos": None, "neg": None})
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.meta or "image_id" not in sample.meta:
            return
        iid = sample.meta["image_id"]
        which = sample.meta.get("pair")  # 'pos' or 'neg'
        gold = _normalize_text(sample.answers[0]) if sample.answers else None
        pred = _yn(pred_text)
        self.pairs[iid][which] = (pred == gold) if gold in ("yes","no") else False
    def compute(self) -> Dict[str, Any]:
        n_img = len(self.pairs)
        both_true = sum(1 for v in self.pairs.values() if v["pos"] is True and v["neg"] is True)
        return {"mme_acc_plus": (both_true / n_img) if n_img else 0.0, "mme_count_pair": n_img}

# --- POPE Accuracy ---
class POPEAcc(BaseMetric):
    """POPE accuracy metric - simple yes/no accuracy"""
    def __init__(self):
        self.n = 0
        self.correct = 0
        
    def update(self, sample: Any, pred_text: str, timings_ms: Dict[str, float]):
        if not hasattr(sample, 'answers') or not sample.answers:
            return
        
        gold = sample.answers[0]  # "yes" or "no"
        if gold not in ("yes", "no"):
            return
            
        pred = _yn(pred_text)
        
        self.n += 1
        if pred == gold:
            self.correct += 1
    
    def compute(self) -> Dict[str, Any]:
        acc = self.correct / self.n if self.n > 0 else 0.0
        return {
            "pope_acc": acc,
            "pope_correct": self.correct,
            "pope_total": self.n
        }

# --- POPE Precision, Recall, F1 ---
class POPEPrecisionRecallF1(BaseMetric):
    """POPE Precision, Recall, F1 (treating 'yes' as positive class)"""
    def __init__(self):
        # initialize Confusion Matrix
        self.tp = 0  # True Positive: pred=yes, gold=yes
        self.fp = 0  # False Positive: pred=yes, gold=no
        self.tn = 0  # True Negative: pred=no, gold=no
        self.fn = 0  # False Negative: pred=no, gold=yes
        self.total_pred_yes = 0 #  record yes number
    
    def update(self, sample: Any, pred_text: str, timings_ms: Dict[str, float]):
        if not hasattr(sample, 'answers') or not sample.answers:
            return
        
        gold = sample.answers[0]
        if gold not in ("yes", "no"):
            return
        
        pred = _yn(pred_text)
        if pred not in ("yes", "no"):
            return
        
        if pred == "yes":
            self.total_pred_yes += 1

        # update matrix counts
        if pred == "yes" and gold == "yes":
            self.tp += 1
        elif pred == "yes" and gold == "no":
            self.fp += 1
        elif pred == "no" and gold == "no":
            self.tn += 1
        elif pred == "no" and gold == "yes":
            self.fn += 1
    
    def compute(self) -> Dict[str, Any]:
        
        # Total samples
        total = self.tp + self.fp + self.tn + self.fn
        
        # Precision, Recall, F1
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy
        acc = (self.tp + self.tn) / total if total > 0 else 0.0
        
        # Yes Ratio
        yes_ratio = self.total_pred_yes / total if total > 0 else 0.0
        
        return {
            "pope_precision": precision,
            "pope_recall": recall,
            "pope_f1": f1,
            "pope_acc_derived": acc,
            "pope_yes_ratio": yes_ratio,
            "pope_tp": self.tp,
            "pope_fp": self.fp,
            "pope_tn": self.tn,
            "pope_fn": self.fn,
            "pope_total_samples": total
        }

# =============================== #
#   Caption Helper Functions      #
# =============================== #

_PUNCT_TABLE = str.maketrans("", "", string.punctuation)

def _caption_tokenize(s: str) -> List[str]:
    s = (s or "").lower().strip()
    s = s.translate(_PUNCT_TABLE)
    # collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.split() if s else []

def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

# =============================== #
#          BLEU (1-4)             #
# =============================== #

def _corpus_bleu(refs_all, hyps_all, max_n=4, smooth=True) -> Dict[str, float]:
    """
    refs_all: List[List[List[str]]]  # [i][k][t]
    hyps_all: List[List[str]]
    返回 { 'bleu1':..., 'bleu2':..., 'bleu3':..., 'bleu4':... }
    """
    assert len(refs_all) == len(hyps_all)
    N = max_n
    total_ref_len = 0
    total_hyp_len = 0
    clipped_counts = [0] * N
    total_counts = [0] * N

    for refs, hyp in zip(refs_all, hyps_all):
        hyp_len = len(hyp)
        total_hyp_len += hyp_len
        # 选一个长度最接近 hyp 的 ref（标准 BLEU 做法）
        ref_lens = [len(r) for r in refs] or [0]
        closest_ref_len = min(ref_lens, key=lambda rl: abs(rl - hyp_len))
        total_ref_len += closest_ref_len

        for n in range(1, N+1):
            hyp_ngrams = Counter(_ngrams(hyp, n))
            if not hyp_ngrams:
                continue
            max_ref_ngrams = Counter()
            for r in refs:
                max_ref_ngrams |= Counter(_ngrams(r, n))
            overlap = 0
            for g, cnt in hyp_ngrams.items():
                overlap += min(cnt, max_ref_ngrams.get(g, 0))
            clipped_counts[n-1] += overlap
            total_counts[n-1] += sum(hyp_ngrams.values())

    # brevity penalty
    if total_hyp_len == 0:
        return {f"bleu{n}": 0.0 for n in range(1, N+1)}

    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - float(total_ref_len) / float(total_hyp_len + 1e-12))

    out = {}
    for up_to in range(1, N+1):
        log_p = 0.0
        valid = True
        for n in range(1, up_to+1):
            c = clipped_counts[n-1]
            t = total_counts[n-1]
            if t == 0:
                valid = False
                break
            if c == 0:
                if smooth:
                    c = 1
                    t = t + 1
                else:
                    valid = False
                    break
            log_p += (1.0 / up_to) * math.log(c / t)
        if not valid:
            out[f"bleu{up_to}"] = 0.0
        else:
            out[f"bleu{up_to}"] = bp * math.exp(log_p)
    return out

class CaptionBLEU(BaseMetric):
    """Corpus BLEU-1/2/3/4 for captions."""
    def __init__(self, max_n: int = 4, smooth: bool = True):
        self.max_n = max_n
        self.smooth = smooth
        self.refs_all = []  # List[List[List[str]]]
        self.hyps_all = []  # List[List[str]]

    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        refs_tok = [_caption_tokenize(a) for a in sample.answers if a]
        if not refs_tok:
            return
        hyp_tok = _caption_tokenize(pred_text)
        self.refs_all.append(refs_tok)
        self.hyps_all.append(hyp_tok)

    def compute(self) -> Dict[str, Any]:
        if not self.refs_all:
            return {f"bleu{n}": 0.0 for n in range(1, self.max_n+1)}
        return _corpus_bleu(self.refs_all, self.hyps_all, max_n=self.max_n, smooth=self.smooth)

# =============================== #
#           ROUGE-L               #
# =============================== #

def _lcs_len(a: List[str], b: List[str]) -> int:
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        ai = a[i-1]
        row = dp[i]
        prev_row = dp[i-1]
        for j in range(1, n+1):
            if ai == b[j-1]:
                row[j] = prev_row[j-1] + 1
            else:
                row[j] = max(prev_row[j], row[j-1])
    return dp[m][n]

def _rouge_l_score(refs_all, hyps_all, beta: float = 1.2) -> float:
    """Corpus-level ROUGE-L (F-measure with LCS)."""
    assert len(refs_all) == len(hyps_all)
    sum_f = 0.0
    cnt = 0
    for refs, hyp in zip(refs_all, hyps_all):
        if not hyp or not refs:
            continue
        best_f = 0.0
        for r in refs:
            lcs = _lcs_len(hyp, r)
            if lcs == 0:
                continue
            prec = lcs / (len(hyp) + 1e-12)
            rec = lcs / (len(r) + 1e-12)
            if prec == 0 and rec == 0:
                continue
            f = ((1 + beta**2) * prec * rec) / (rec + beta**2 * prec + 1e-12)
            if f > best_f:
                best_f = f
        sum_f += best_f
        cnt += 1
    return sum_f / cnt if cnt > 0 else 0.0

class CaptionROUGEL(BaseMetric):
    """Corpus ROUGE-L (F-measure)."""
    def __init__(self, beta: float = 1.2):
        self.beta = beta
        self.refs_all = []
        self.hyps_all = []

    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        refs_tok = [_caption_tokenize(a) for a in sample.answers if a]
        if not refs_tok:
            return
        hyp_tok = _caption_tokenize(pred_text)
        self.refs_all.append(refs_tok)
        self.hyps_all.append(hyp_tok)

    def compute(self) -> Dict[str, Any]:
        if not self.refs_all:
            return {"rouge_l": 0.0}
        score = _rouge_l_score(self.refs_all, self.hyps_all, beta=self.beta)
        return {"rouge_l": score}

# =============================== #
#            METEOR               #
# =============================== #

def _meteor_sentence(refs: List[List[str]], hyp: List[str]) -> float:
    if not hyp or not refs:
        return 0.0

    best_score = 0.0
    hyp_unigrams = Counter(hyp)
    for r in refs:
        if not r:
            continue
        ref_unigrams = Counter(r)
        overlap = sum(min(c, ref_unigrams[w]) for w, c in hyp_unigrams.items())
        if overlap == 0:
            continue
        precision = overlap / (len(hyp) + 1e-12)
        recall = overlap / (len(r) + 1e-12)
        if precision == 0 and recall == 0:
            continue
        # Fmean，alpha = 0.9 => weight recall more
        alpha = 0.9
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall + 1e-12)
        if fmean > best_score:
            best_score = fmean
    return best_score

class CaptionMETEOR(BaseMetric):
    """Simplified METEOR (unigram F-mean, no synonym / fragmentation)."""
    def __init__(self):
        self.scores = []

    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        refs_tok = [_caption_tokenize(a) for a in sample.answers if a]
        if not refs_tok:
            return
        hyp_tok = _caption_tokenize(pred_text)
        s = _meteor_sentence(refs_tok, hyp_tok)
        self.scores.append(s)

    def compute(self) -> Dict[str, Any]:
        if not self.scores:
            return {"meteor": 0.0}
        return {"meteor": sum(self.scores) / len(self.scores)}

# =============================== #
#             CIDEr               #
# =============================== #

def _build_cider_idf(refs_all, max_n=4):
    df = [Counter() for _ in range(max_n)]
    N = len(refs_all)
    for refs in refs_all:
        for n in range(1, max_n+1):
            seen = set()
            for r in refs:
                for g in set(_ngrams(r, n)):
                    seen.add(g)
            for g in seen:
                df[n-1][g] += 1
    idf = []
    for n in range(1, max_n+1):
        idf_n = {}
        for g, df_g in df[n-1].items():
            idf_n[g] = math.log((N + 1.0) / (df_g + 1.0))  # smoothing
        idf.append(idf_n)
    return idf  # List[Dict[ngram, idf]]

def _tfidf_vec(tokens: List[str], idf_n: Dict[tuple, float], n: int) -> Dict[tuple, float]:
    counts = Counter(_ngrams(tokens, n))
    if not counts:
        return {}
    length = sum(counts.values())
    vec = {}
    for g, c in counts.items():
        idf = idf_n.get(g, 0.0)
        vec[g] = (c / length) * idf
    return vec

def _cosine_sim(vec1: Dict[tuple, float], vec2: Dict[tuple, float]) -> float:
    if not vec1 or not vec2:
        return 0.0
    # dot
    if len(vec1) < len(vec2):
        v_small, v_big = vec1, vec2
    else:
        v_small, v_big = vec2, vec1
    dot = 0.0
    for g, w in v_small.items():
        dot += w * v_big.get(g, 0.0)
    # norms
    n1 = math.sqrt(sum(w*w for w in vec1.values()))
    n2 = math.sqrt(sum(w*w for w in vec2.values()))
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot / (n1 * n2 + 1e-12)

def _cider_score(refs_all, hyps_all, max_n=4, sigma: float = 6.0) -> float:
    assert len(refs_all) == len(hyps_all)
    idf_all = _build_cider_idf(refs_all, max_n=max_n)
    scores = []
    for refs, hyp in zip(refs_all, hyps_all):
        if not hyp or not refs:
            continue
        per_ref_scores = []
        for r in refs:
            sim_sum = 0.0
            for n in range(1, max_n+1):
                vec_h = _tfidf_vec(hyp, idf_all[n-1], n)
                vec_r = _tfidf_vec(r, idf_all[n-1], n)
                sim = _cosine_sim(vec_h, vec_r)
                sim_sum += sim
            per_ref_scores.append(sim_sum / max_n)
        if per_ref_scores:
            scores.append(sum(per_ref_scores) / len(per_ref_scores))
    return sum(scores) / len(scores) if scores else 0.0

class CaptionCIDEr(BaseMetric):
    """Simplified CIDEr (TF-IDF n-gram cosine)."""
    def __init__(self, max_n: int = 4):
        self.max_n = max_n
        self.refs_all = []
        self.hyps_all = []

    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        refs_tok = [_caption_tokenize(a) for a in sample.answers if a]
        if not refs_tok:
            return
        hyp_tok = _caption_tokenize(pred_text)
        self.refs_all.append(refs_tok)
        self.hyps_all.append(hyp_tok)

    def compute(self) -> Dict[str, Any]:
        if not self.refs_all:
            return {"cider": 0.0}
        score = _cider_score(self.refs_all, self.hyps_all, max_n=self.max_n)
        return {"cider": score}