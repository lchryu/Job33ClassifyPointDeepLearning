import logging
import torch.nn.functional as F
import torch

from classifierPoint.models.base_model import BaseModel
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.modules.torchsparse import initialize_minkowski_unet
from torchsparse import SparseTensor
from classifierPoint.models.segmentation.base import SegmentationBase

log = logging.getLogger(__name__)


class Minkowski_Baseline_Model(SegmentationBase):
    def __init__(self, option, model_type, dataset, modules):
        super(Minkowski_Baseline_Model, self).__init__(option, dataset, sparse_class=True)
        self._weight_classes = dataset.weight_classes
        self.model = initialize_minkowski_unet(option.model_name, dataset.feature_dimension, dataset.num_classes)
        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        if self.sampler:
            data = self.sampler(data)
        if data.batch.dim() == 1:
            data.batch = data.batch.unsqueeze(-1)
        coords = torch.cat([data.coords, data.batch], -1).type(torch.int)
        self.input = SparseTensor(data.x, coords).to(self.device)
        if data.y is not None:
            self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
