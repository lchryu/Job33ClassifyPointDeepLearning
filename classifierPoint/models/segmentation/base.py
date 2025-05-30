import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any
from classifierPoint.models.base_model import BaseModel
from classifierPoint.models.base_architectures import UnetBasedModel
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.utils.check_transform import check_VoxelGrid, VoxelGrid

log = logging.getLogger(__name__)


class Segmentation_MP(UnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        """Initialize this model class.
        Parameters:
            opt -- training/test options
        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, model names, and optimizers
        """
        UnetBasedModel.__init__(
            self, option, model_type, dataset, modules
        )  # call the initialization method of UnetBasedModel

        self._weight_classes = dataset.weight_classes

        nn = option.mlp_cls.nn
        self.dropout = option.mlp_cls.get("dropout")
        self.lin1 = torch.nn.Linear(nn[0], nn[1])
        self.lin2 = torch.nn.Linear(nn[2], nn[3])
        self.lin3 = torch.nn.Linear(nn[4], dataset.num_classes)

        self.loss_names = ["loss_seg"]

    def set_input(self, data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        data = data.to(device)
        self.input = data
        self.labels = data.y
        self.batch_idx = data.batch

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        data = self.model(self.input)
        x = F.relu(self.lin1(data.x))
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin2(x)
        x = F.dropout(x, p=self.dropout, training=bool(self.training))
        x = self.lin3(x)
        self.output = F.log_softmax(x, dim=-1)

        self.data_visual = self.input
        self.data_visual.y = self.labels
        self.data_visual.pred = torch.max(self.output, -1)[1]
        return self.output

    def compute_loss(self):
        self.loss_seg = F.nll_loss(self.output, self.labels, ignore_index=IGNORE_LABEL) + self.get_internal_loss()

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G


class SegmentationBase(BaseModel):
    def __init__(self, option, dataset=None, sparse_class=False):
        BaseModel.__init__(self, option)
        self.sampler = None
        if sparse_class:
            self.sampler = VoxelGrid(size=0.1) if not check_VoxelGrid(dataset) else None
            if self.sampler:
                log.warning(f"will use VoxelGrid(size=0.1) sample")
        self.loss_module, _ = BaseModel.get_metric_loss_and_miner(getattr(self.opt, "loss", None), None)
        self.loss_names = ["loss_seg"]
        self.CE_loss =  nn.CrossEntropyLoss(ignore_index=-1)
    def compute_loss(self):
        self.loss_seg = self.loss_module(self.output, self.labels) + self.CE_loss(self.output, self.labels)

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        #print(self.loss_seg)
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
