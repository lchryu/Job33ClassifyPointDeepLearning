import logging
import torch.nn.functional as F
import torch

from classifierPoint.modules.PVCNN import spvnas
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.models.segmentation.base import SegmentationBase
from torchsparse import SparseTensor

log = logging.getLogger(__name__)


class SPVNAS(SegmentationBase):
    __REQUIRED_DATA__ = [
        "coords",
    ]

    def __init__(self, option, model_type, dataset, modules):
        super(SPVNAS, self).__init__(option, dataset, sparse_class=True)
        self.model = spvnas.make_model(dataset)
        self.loss_names = ["loss_seg"]
    def norm(self,centered_points):
        max_distance = torch.max(torch.sqrt(torch.sum(centered_points**2, dim=1)))
        normalized_points = centered_points / max_distance
        return normalized_points
    def set_input(self, data, device):
        if self.sampler:
            data = self.sampler(data)
        if data.batch.dim() == 1:
            data.batch = data.batch.unsqueeze(-1)
        coords = torch.cat([data.coords, data.batch], -1).type(torch.int)
        self.batch_idx = data.batch.squeeze()
        self.input = SparseTensor(data.x, coords).to(self.device)
        if data.y is not None:
            self.labels = data.y.to(self.device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
