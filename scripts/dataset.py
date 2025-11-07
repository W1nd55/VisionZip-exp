
# ================= #
# Dataset: VQAv2    #
# ================= #

from typing import List, Dict, Any, Optional, Iterable
from scripts.abstract import BaseDataset, Sample
import json


class VQAv2Dataset(BaseDataset):
    """
    Expects an annotations file with a list of dicts:
    {
      "question_id": "123",
      "image_path": "/path/to/img.jpg",
      "question": "What is ...?",
      "answers": ["cat", "a cat", "kitty", ...]  # up to 10
    }
    """
    def __init__(self, ann_path: str, limit: Optional[int]=None):
        with open(ann_path, "r") as f:
            data = json.load(f)
        if limit:
            data = data[:limit]
        self.data = data

    def __len__(self): 
        return len(self.data)

    def __iter__(self) -> Iterable[Sample]:
        for x in self.data:
            yield Sample(
                qid=str(x["question_id"]),
                image_path=x["image_path"],
                prompt=x["question"],
                answers=x.get("answers", None),
                meta=None
            )


class MMEDataset(BaseDataset):
    """
    ann_json structure (each entry represents two questions, positive and negative, for one image):
    [
      {
        "image_id": "0006adf999ccc899",
        "image_path": ".../landmark/0006adf999ccc899.jpg",
        "q_pos": "Is this a photo of Cotehele? Please answer yes or no.",
        "a_pos": "yes",
        "q_neg": "Is this a photo of Mozirje Grove? Please answer yes or no.",
        "a_neg": "no"
      },
      ...
    ]
    During evaluation, this expands into two Samples (pos/neg). MMEAccPlus will aggregate 
    results by 'image_id' to calculate ACC+.
    """
    def __init__(self, ann_path: str, limit: Optional[int] = None):
        with open(ann_path, "r") as f:
            data = json.load(f)
        if limit is not None:
            data = data[:limit]
        self._samples: List[Sample] = []
        for it in data:
            iid = str(it["image_id"])
            img = it["image_path"]
            self._samples.append(Sample(
                qid=iid + "_pos",
                image_path=img,
                prompt=it["q_pos"],
                answers=[(it.get("a_pos") or "yes").lower()],
                meta={"image_id": iid, "pair": "pos", "subtask": it.get("subtask")}
            ))
            self._samples.append(Sample(
                qid=iid + "_neg",
                image_path=img,
                prompt=it["q_neg"],
                answers=[(it.get("a_neg") or "no").lower()],
                meta={"image_id": iid, "pair": "neg", "subtask": it.get("subtask")}
            ))

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterable[Sample]:
        for s in self._samples:
            yield s

class POPEDataset(BaseDataset):
    """
    POPE (Polling-based Object Probing Evaluation) Dataset
    Expects an annotations file with a list of dicts:
    {
      "question_id": "123",
      "image": "/path/to/img.jpg",  # or "image_path"
      "text": "Is there a cat in the image?",
      "label": "yes"  # or "no"
    }
    """
    def __init__(self, ann_path: str, limit: Optional[int] = None):
        with open(ann_path, "r") as f:
            data = json.load(f)
        if limit is not None:
            data = data[:limit]
        self._samples: List[Sample] = []
        
        for idx, it in enumerate(data):
            qid = str(it.get("question_id", idx))
            img_path = it.get("image") or it.get("image_path")
            question = it.get("text") or it.get("question")
            label = (it.get("label") or it.get("answer", "")).strip().lower()
            
            if question and not question.lower().strip().endswith(("yes or no", "yes or no.")):
                question = question.rstrip() + " Please answer yes or no."
            
            self._samples.append(Sample(
                qid=qid,
                image_path=img_path,
                prompt=question,
                answers=[label] if label in ("yes", "no") else None,
                meta={"original_label": label}
            ))
    
    def __len__(self) -> int:
        return len(self._samples)
    
    def __iter__(self) -> Iterable[Sample]:
        for s in self._samples:
            yield s