import torch
from torchmetrics import Metric


class FIDMetrics(Metric):
    def __init__(self, prefix='prefix',
                 dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.ones(1, dtype=torch.int64)*999,
                       dist_reduce_fx="mean")

    @torch.no_grad()
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        pass

    def compute(self, fid):
        metric_dict = {self.prefix + "/fid": fid}
        return {k: v for k, v in metric_dict.items()}
