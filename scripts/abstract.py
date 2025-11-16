from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
import torch

# =============== #
# Abstract Layers #
# =============== #

@dataclass
class Sample:
    qid: str
    image_path: Optional[str]
    prompt: str
    answers: Optional[List[str]] = None  # for VQA-style
    meta: Optional[Dict[str, Any]] = None

class BaseDataset(ABC):
    @abstractmethod
    def __len__(self) -> int: ...
    @abstractmethod
    def __iter__(self) -> Iterable[Sample]: ...

class BaseModel(ABC):
    @abstractmethod
    def device(self) -> torch.device: ...
    @abstractmethod
    def prepare_inputs(self, sample: Sample) -> Dict[str, Any]: ...
    @abstractmethod
    def generate(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return dict with keys:
           - 'text': str
           - 'num_new_tokens': int (optional)"""
        ...

class BaseMetric(ABC):
    @abstractmethod
    def update(self, sample: Sample, pred_text: str): ...
    @abstractmethod
    def compute(self) -> Dict[str, Any]: ...
