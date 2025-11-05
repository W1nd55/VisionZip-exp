
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
