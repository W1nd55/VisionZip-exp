import string
import math
from typing import List, Dict, Any
from scripts.abstract import BaseMetric, Sample

# ================= #
# Metric Implement  #
# ================= #

def _normalize_text(s: str) -> str:
    s = s.lower().strip()
    table = str.maketrans('', '', string.punctuation)
    s = s.translate(table)
    s = " ".join(s.split())
    for art in [" a ", " an ", " the "]:
        s = s.replace(art, " ")
    return s.strip()

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
        keys = sorted(self.logs[0].keys())
        out = {}
        for k in keys:
            arr = [x[k] for x in self.logs if k in x]
            if not arr: 
                continue
            out[f"{k}_ms_p50"] = percentile(arr, 50)
            out[f"{k}_ms_p95"] = percentile(arr, 95)
            out[f"{k}_ms_avg"] = sum(arr)/len(arr)
        # tok/s
        if any('num_new_tokens' in x for x in self.logs) and any('decode_ms' in x for x in self.logs):
            toks = sum(x.get('num_new_tokens', 0) for x in self.logs)
            decode_ms = sum(x.get('decode_ms', 0.0) for x in self.logs)
            out["decode_tok_per_s"] = toks / (decode_ms/1000.0 + 1e-9)
        return out
