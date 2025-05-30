import logging
import torch.nn.functional as F
import torch

from classifierPoint.modules.DRINet import drinet
from classifierPoint.models.base_model import BaseModel
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.models.segmentation.base import SegmentationBase
from torchsparse import SparseTensor

log = logging.getLogger(__name__)


class DRINet(SegmentationBase):
    def __init__(self, option, model_type, dataset, modules):
        super(DRINet, self).__init__(option, dataset, sparse_class=True)
        self.model = drinet.DRINet(option, dataset)
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        if self.sampler:
            data = self.sampler(data)
        if data.batch.dim() == 1:
            data.batch = data.batch.unsqueeze(-1)
        coords = torch.cat([data.coords, data.batch], -1).int()
        self.batch_idx = data.batch.squeeze()
        #data.x = data.x[:,2].unsqueeze(1) #将xyz特征只保留z特征
        self.input = SparseTensor(data.x, coords).to(self.device)
        if data.y is not None:
            self.labels = data.y.to(self.device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
