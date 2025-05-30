import logging
import torch.nn.functional as F
import torch

from classifierPoint.modules.PVCNN import pvcnn
from classifierPoint.models.segmentation.base import SegmentationBase
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.core.data_transform import VoxelGrid
from torchsparse import SparseTensor
#from need.torchsparse import SparseTensor
# from need.spvnas import get_model
# from need.pvcnn2 import PVCNN
#from need import pvcnn2
log = logging.getLogger(__name__)


class PVCNN(SegmentationBase):
    def __init__(self, option, model_type, dataset, modules):
        super(PVCNN, self).__init__(option, dataset, sparse_class=True)
        self.model = pvcnn.PVCNN(option, dataset)
        #self.model = pvcnn2.PVCNN(option, dataset)
        self.loss_names = ["loss_seg"]
    def set_input(self, data, device):
        if self.sampler:
            data = self.sampler(data)
        if data.batch.dim() == 1:
            data.batch = data.batch.unsqueeze(-1)
        coords = torch.cat([data.coords, data.batch], -1).type(torch.int)
        self.batch_idx = data.batch.squeeze()
        self.input = SparseTensor(data.x, coords).to(device)
        if data.y is not None:
            self.labels = data.y.to(device)

    def forward(self, *args, **kwargs):
        self.output = self.model(self.input)
