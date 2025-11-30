from typing import List, Dict, Any, Optional
import os
import random
import json
import csv
import torch
from scripts.abstract import BaseModel, BaseDataset, BaseMetric

# ============== #
# Evaluator Core #
# ============== #

class Evaluator:
    def __init__(
        self,
        model: BaseModel,
        dataset: BaseDataset,
        metrics: List[BaseMetric],
        output_dir: str,
        seed: int = 42,
        warmup: int = 3
    ):
        self.model = model
        self.dataset = dataset
        self.metrics = metrics
        self.output_dir = output_dir
        self.seed = seed
        self.warmup = warmup
        os.makedirs(output_dir, exist_ok=True)

    def _set_seed(self):
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def run(self, limit: Optional[int]=None, save_jsonl: str="results.jsonl", save_csv: str="summary.csv") -> Dict[str, Any]:
        self._set_seed()

        # Warmup (ensure cache/JIT/cuda stability)
        print(f"[Eval] Warmup {self.warmup} samples ...")
        it = iter(self.dataset)
        for _ in range(self.warmup):
            try:
                s = next(it)
            except StopIteration:
                break
            inputs = self.model.prepare_inputs(s)
            _ = self.model.generate(inputs)

        # Formal evaluation
        print("[Eval] Running ...")
        jsonl_path = os.path.join(self.output_dir, save_jsonl)
        f_jsonl = open(jsonl_path, "w")

        count = 0
        for sample in self.dataset:
            if limit and count >= limit:
                break
            inputs = self.model.prepare_inputs(sample)
            out = self.model.generate(inputs)
            pred = out["text"]
            timings = {
                "load_ms": out.get("load_ms", 0.0),
                "preprocess_ms": out.get("preprocess_ms", 0.0),
                "end2end_ms": out.get("end2end_ms", 0.0),
                "num_new_tokens": out.get("num_new_tokens", 0),
            }

            # update metrics
            for m in self.metrics:
                m.update(sample, pred, timings)

            row = {
                "qid": sample.qid,
                "image_path": sample.image_path,
                "prompt": sample.prompt,
                "pred": pred,
                "answers": sample.answers,
                **timings
            }
            f_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            f_jsonl.flush() # Force write to disk
            count += 1
            print(f"[Eval] {count} samples done. Last latency: {timings['end2end_ms']:.2f}ms")

        f_jsonl.close()

        # Summarize results
        summary = {}
        for m in self.metrics:
            summary.update(m.compute())
        summary_path = os.path.join(self.output_dir, save_csv)
        with open(summary_path, "w", newline="") as cf:
            w = csv.writer(cf)
            for k,v in summary.items():
                w.writerow([k, v])

        print(f"[Eval] Finished {count} samples.")
        print(f"[Eval] JSONL saved to: {jsonl_path}")
        print(f"[Eval] Summary saved to: {summary_path}")
        return summary