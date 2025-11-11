import time
import torch


# ===================== #
# Utils: timing helpers #
# ===================== #

class StageTimer:
    """CUDA-aware timer for stage-wise latency."""
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda and torch.cuda.is_available()
        if self.use_cuda:
            self.events = {}
        self.wall = {}

    def _now(self):
        return time.perf_counter()

    def start(self, name: str):
        if self.use_cuda:
            self.events.setdefault(name, [torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)])
            self.events[name][0].record()
        self.wall.setdefault(name, [None, None])
        self.wall[name][0] = self._now()

    def end(self, name: str):
        if self.use_cuda:
            self.events[name][1].record()
            torch.cuda.synchronize()
        self.wall[name][1] = self._now()

    def result_ms(self, name: str) -> float:
        if self.use_cuda:
            start, end = self.events[name]
            # CUDA event returns milliseconds
            return float(start.elapsed_time(end))
        else:
            st, ed = self.wall[name]
            return float((ed - st) * 1000.0)
