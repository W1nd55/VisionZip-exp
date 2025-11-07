import string
import math
from typing import List, Dict, Any, Optional
from scripts.abstract import BaseMetric, Sample
from collections import defaultdict
import re

_YESNO_RE = re.compile(r"\b(yes|no)\b", flags=re.IGNORECASE)

# ================= #
# Metric Implement  #
# ================= #

# def _normalize_text(s: str) -> str:
#     s = s.lower().strip()
#     table = str.maketrans('', '', string.punctuation)
#     s = s.translate(table)
#     s = " ".join(s.split())
#     for art in [" a ", " an ", " the "]:
#         s = s.replace(art, " ")
#     return s.strip()
def _normalize_text(s: str) -> str:
    return (s or "").strip().lower()

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
            decode_ms = sum(
                 (x.get('decode_ms', 0.0) if 'decode_ms' in x else x.get('decode', 0.0))
                 for x in self.logs
            )
            tps = toks / (decode_ms/1000.0 + 1e-9)
            out["decode_tok_per_s"] = tps
            out["tok/s"] = tps
        return out
    
def _yn(s: str):
    if not s:
        return None
    m = _YESNO_RE.search(s)
    if not m:
        return None
    return m.group(1).lower()

class MMEAcc(BaseMetric):
    """Per-question accuracy (ACC)"""
    def __init__(self):
        self.n = 0
        self.c = 0
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        gold = _normalize_text(sample.answers[0]) if sample.answers else None
        if gold not in ("yes","no"):
            return
        pred = _yn(pred_text)
        self.n += 1
        if pred == gold:
            self.c += 1
    def compute(self) -> Dict[str, Any]:
        return {"mme_acc": self.c / self.n if self.n else 0.0, "mme_count_q": self.n}

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
        return {"mme_acc_plus": (both_true / n_img) if n_img else 0.0, "mme_count_img": n_img}

class POPEAcc(BaseMetric):
    """POPE accuracy metric - simple yes/no accuracy"""
    def __init__(self):
        self.n = 0
        self.correct = 0
        
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        
        gold = sample.answers[0]  # "yes" or "no"
        if gold not in ("yes", "no"):
            return
            
        # 从预测文本中提取 yes/no
        pred = _yn(pred_text)  # 使用已有的 _yn 函数
        
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


class POPEPrecisionRecallF1(BaseMetric):
    """POPE Precision, Recall, F1 (treating 'yes' as positive class)"""
    def __init__(self):
        self.tp = 0  # true positive: pred=yes, gold=yes
        self.fp = 0  # false positive: pred=yes, gold=no
        self.tn = 0  # true negative: pred=no, gold=no
        self.fn = 0  # false negative: pred=no, gold=yes
    
    def update(self, sample: Sample, pred_text: str, timings_ms: Dict[str, float]):
        if not sample.answers:
            return
        
        gold = sample.answers[0]
        if gold not in ("yes", "no"):
            return
        
        pred = _yn(pred_text)
        if pred not in ("yes", "no"):
            return
        
        if pred == "yes" and gold == "yes":
            self.tp += 1
        elif pred == "yes" and gold == "no":
            self.fp += 1
        elif pred == "no" and gold == "no":
            self.tn += 1
        elif pred == "no" and gold == "yes":
            self.fn += 1
    
    def compute(self) -> Dict[str, Any]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "pope_precision": precision,
            "pope_recall": recall,
            "pope_f1": f1,
            "pope_tp": self.tp,
            "pope_fp": self.fp,
            "pope_tn": self.tn,
            "pope_fn": self.fn
        }