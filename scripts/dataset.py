#!/usr/bin/env python3
# scripts/evalkit.py
# ================= #
# Dataset: VQAv2    #
# ================= #

from typing import List, Dict, Any, Optional, Iterable
from scripts.abstract import BaseDataset, Sample
import json
from pathlib import Path

def _ensure_suffix(q: str) -> str:
    """Ensures the prompt ends with the required MME suffix."""
    if not q: return q
    suff = "Please answer yes or no."
    return q if q.lower().strip().endswith(suff.lower()) else (q.rstrip() + " " + suff)


class VQAv2Dataset(BaseDataset):
    """
    Expects an annotations file with a list of dicts:
    {
      "question_id": "123",
      "image_path": "/path/to/img.jpg",
      "question": "What is ...?",
      "answers": ["cat", "a cat", "kitty", ...]  # up to 10 possible answers
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
    Supports two types of MME annotations:
    1) Paired schema: {image_path, image_id?, q_pos, a_pos?, q_neg, a_neg?}
    2) Flat schema: {question_id, image_path, question/prompt, answers: ["yes"/"no"], meta?}
       - If meta.pair is missing, it can be inferred from the question_id suffix (*_pos/*_neg).
    """
    def __init__(self, ann_path: str, limit: int | None = None):
        self.limit = limit
        data = json.loads(Path(ann_path).read_text(encoding="utf-8"))
        self.samples = []

        def push(qid, img, prompt, answers, image_id=None, pair=None, extra_meta=None):
            meta = {"image_id": image_id, "pair": pair}
            if extra_meta and isinstance(extra_meta, dict):
                meta.update(extra_meta)
            self.samples.append(Sample(
                qid=str(qid),
                image_path=img,
                prompt=_ensure_suffix(prompt), # Ensure the prompt has the required MME suffix
                answers=list(answers) if isinstance(answers, (list, tuple)) else [answers],
                meta=meta
            ))

        for it in data:
            # ---------- Paired Schema Check ----------
            if ("q_pos" in it) or ("q_neg" in it):
                img = it["image_path"]
                base_id = it.get("image_id") or Path(img).stem

                qpos = it.get("q_pos")
                if qpos:
                    # Positive (Yes) sample
                    push(
                        qid=f"{base_id}_pos",
                        img=img,
                        prompt=qpos,
                        # Answer must be "yes" for positive question
                        answers=["yes" if str(it.get("a_pos","yes")).lower().startswith("y") else "yes"],
                        image_id=base_id,
                        pair="pos",
                        extra_meta=it.get("meta")
                    )

                qneg = it.get("q_neg")
                if qneg:
                    # Negative (No) sample
                    push(
                        qid=f"{base_id}_neg",
                        img=img,
                        prompt=qneg,
                        # Answer must be "no" for negative question
                        answers=["no"  if str(it.get("a_neg","no")).lower().startswith("n") else "no"],
                        image_id=base_id,
                        pair="neg",
                        extra_meta=it.get("meta")
                    )
                continue

            # ---------- Flat Schema Check ----------
            # Expects fields like {question_id, image_path, question|prompt, answers, meta?}
            qid   = it.get("question_id") or it.get("qid")
            img   = it.get("image_path")
            qtext = it.get("question") or it.get("prompt")
            ans   = it.get("answers") or []
            meta  = it.get("meta") or {}
            image_id = it.get("image_id") or meta.get("image_id") or (Path(img).stem if img else None)

            # Attempt to infer the pair type from QID suffix
            pair = it.get("pair") or meta.get("pair")
            if not pair and isinstance(qid, str):
                low = qid.lower()
                if low.endswith("_pos") or low.endswith("-pos"):
                    pair = "pos"
                elif low.endswith("_neg") or low.endswith("-neg"):
                    pair = "neg"

            if not qid or not img or not qtext or not ans:
                # Skip if minimal required fields are missing
                continue

            push(qid=qid, img=img, prompt=qtext, answers=ans, image_id=image_id, pair=pair, extra_meta=meta)

        # Optional: Skipping the limit truncation here; let Evaluator.run(limit=...) handle it
        # If pre-truncation is needed:
        # if self.samples and self.limit: self.samples = self.samples[:self.limit]

    def __iter__(self):
        # The Evaluator.run method will handle the limit; iterate over all collected samples here
        return iter(self.samples)

    def __len__(self):
        return len(self.samples)

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
            
            # Ensure question ends with the required yes/no suffix for POPE
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

class COCOCaptionDataset(BaseDataset):
    """
    COCO Caption Dataset

    Expects a COCO-style caption annotation JSON, e.g.:
      captions_val2014.json or captions_train2014.json

    Structure:
    {
      "images": [
        {"id": 391895, "file_name": "COCO_val2014_000000391895.jpg", ...},
        ...
      ],
      "annotations": [
        {"image_id": 391895, "caption": "A group of people ..."},
        ...
      ]
    }

    For each image_id, we gather all its reference captions into Sample.answers.

    image_path is constructed as:
        datasets/coco/train2014/
        datasets/coco/val2014/
    """

    def __init__(self, ann_path: str, limit: Optional[int] = None, image_root: str = "datasets/coco"):
        self._limit = limit
        self._image_root = Path(image_root)

        ann_path = Path(ann_path)
        with open(ann_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        # 1) image_id -> file_name
        id2fname: Dict[int, str] = {}
        for img in obj.get("images", []):
            img_id = img.get("id")
            fname = img.get("file_name")
            if img_id is None or not fname:
                continue
            id2fname[int(img_id)] = fname

        # 2) image_id -> [captions...]
        refs: Dict[int, List[str]] = {}
        for a in obj.get("annotations", []):
            img_id = a.get("image_id")
            cap = a.get("caption")
            if img_id is None or not cap:
                continue
            img_id = int(img_id)
            refs.setdefault(img_id, []).append(cap)

        # 3) Build Sample list
        self._samples: List[Sample] = []
        for idx, (img_id, caps) in enumerate(refs.items()):
            if self._limit is not None and idx >= self._limit:
                break

            fname = id2fname.get(img_id, None)
            if not fname:
                continue
            # Determine image_path based on file_name
            if "val2014" in fname:
                split = "val2014"
            elif "train2014" in fname:
                split = "train2014"
            else:
                split = None

            if split is not None:
                image_path = str(self._image_root / split / fname)
            else:
                image_path = str(self._image_root / fname)

            meta = {
                "image_id": img_id,
                "file_name": fname,
                "split": split,
            }

            self._samples.append(
                Sample(
                    qid=str(img_id),
                    image_path=image_path,
                    prompt="Describe the image in one sentence.",
                    answers=caps,  # reference captions
                    meta=meta,
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterable[Sample]:
        for s in self._samples:
            yield s

class DocVQADataset(BaseDataset):
    """
    DocVQA (Single-Page Document VQA, Task 1)

    Expected JSON format (official DocVQA train/val/test, including *_withQT.json):
    {
      "dataset_name": "docvqa",
      "dataset_split": "train" | "val" | "test",
      "dataset_version": "1.0",
      "data": [
        {
          "questionId": 52212,
          "question": "Whose signature is given?",
          "image": "documents/txpn0095_1.png",
          "docId": 1968,
          "ucsf_document_id": "txpn0095",
          "ucsf_document_page_no": "1",
          "answers": ["Edward R. Shannon", "Edward Shannon"],
          "questionType": "..."      # in *_withQT.json
        },
        ...
      ]
    }

    - ann_path: path to e.g. docvqa_train_v1.0_withQT.json
    - image_root (optional): base directory for images.
        If None, defaults to ann_path.parent, so "image" field is joined as:
            Path(ann_path).parent / it["image"]
    """

    def __init__(
        self,
        ann_path: str,
        limit: Optional[int] = None,
        image_root: Optional[str] = None,
    ):
        self._samples: List[Sample] = []

        ann_path = Path(ann_path)
        with ann_path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        # Handle both official dict format {"data": [...]} and a plain list
        if isinstance(obj, dict) and "data" in obj:
            entries = obj["data"]
        elif isinstance(obj, list):
            entries = obj
        else:
            raise ValueError(f"Unrecognized DocVQA annotation format: {ann_path}")

        if limit is not None:
            entries = entries[:limit]

        # Base dir for images
        base_dir = Path(image_root) if image_root is not None else ann_path.parent

        for idx, it in enumerate(entries):
            # ---- Basic fields ----
            qid = it.get("questionId") or it.get("question_id") or idx
            question = it.get("question")
            img_rel = it.get("image") or it.get("image_path")
            answers = it.get("answers") or []

            if question is None or img_rel is None:
                # skip malformed entries
                continue

            # Normalize answers to a list of strings
            if isinstance(answers, str):
                answers = [answers]
            elif not isinstance(answers, (list, tuple)):
                answers = [str(answers)]

            # Build full image path
            img_path = base_dir / img_rel

            # ---- Meta info (doc id, page, question type, etc.) ----
            meta_keys = [
                "docId",
                "ucsf_document_id",
                "ucsf_document_page_no",
                "questionType",
                "question_type",
                "dataset_split",
                "dataset_name",
                "dataset_version",
            ]
            meta = {k: it[k] for k in meta_keys if k in it}

            self._samples.append(
                Sample(
                    qid=str(qid),
                    image_path=str(img_path),
                    prompt=question,
                    answers=answers,
                    meta=meta or None,
                )
            )

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterable[Sample]:
        for s in self._samples:
            yield s