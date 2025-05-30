from typing import Any
import logging
from torch.nn import Sequential, Dropout, Linear
import torch.nn.functional as F
import torch
from torch import nn

from classifierPoint.core.common_modules import FastBatchNorm1d
from classifierPoint.modules.KPConv import *
from classifierPoint.core.base_conv.partial_dense import *
from classifierPoint.models.base_architectures.unet import UnwrappedUnetBasedModel
from classifierPoint.datasets.multiscale_data import MultiScaleBatch
from classifierPoint.models.base_model import BaseModel
from torch_geometric.data import Data

log = logging.getLogger(__name__)


class KPConvPaper(UnwrappedUnetBasedModel):
    def __init__(self, option, model_type, dataset, modules):
        # Extract parameters from the dataset
        self._num_classes = dataset.num_classes

        # Assemble encoder / decoder
        UnwrappedUnetBasedModel.__init__(self, option, model_type, dataset, modules)
        self.loss_module, _ = BaseModel.get_metric_loss_and_miner(getattr(self.opt, "loss", None), None)
        self.loss_names = ["loss_seg"]
        # Build final MLP
        last_mlp_opt = option.mlp_cls
        in_feat = last_mlp_opt.nn[0]
        self.FC_layer = Sequential()
        for i in range(1, len(last_mlp_opt.nn)):
            self.FC_layer.add_module(
                str(i),
                Sequential(
                    *[
                        Linear(in_feat, last_mlp_opt.nn[i], bias=False),
                        FastBatchNorm1d(last_mlp_opt.nn[i], momentum=last_mlp_opt.bn_momentum),
                        LeakyReLU(0.2),
                    ]
                ),
            )
            in_feat = last_mlp_opt.nn[i]

        if last_mlp_opt.dropout:
            self.FC_layer.add_module("Dropout", Dropout(p=last_mlp_opt.dropout))

        self.FC_layer.add_module("Class", Lin(in_feat, self._num_classes, bias=False))

    def compute_loss(self):
        self.loss_seg = self.loss_module(self.output, self.labels)

    def set_input(self, data: Data, device):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """
        # delete some useless attr
        data = data.to(device)
        # First add a column of 1 as feature for the network to be able to learn 3D shapes
        data.x = add_ones(data.pos, data.x, True)

        if isinstance(data, MultiScaleBatch):
            self.pre_computed = data.multiscale
            self.upsample = data.upsample
            del data.upsample
            del data.multiscale
        else:
            self.upsample = None
            self.pre_computed = None

        self.input = data
        if data.y is not None:
            self.labels = data.y
        self.batch_idx = data.batch

    def forward(self, *args, **kwargs) -> Any:
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        stack_down = []

        data = self.input
        for i in range(len(self.down_modules) - 1):
            data = self.down_modules[i](data, precomputed=self.pre_computed)
            stack_down.append(data)

        data = self.down_modules[-1](data, precomputed=self.pre_computed)
        innermost = False

        if not isinstance(self.inner_modules[0], Identity):
            stack_down.append(data)
            data = self.inner_modules[0](data)
            innermost = True

        for i in range(len(self.up_modules)):
            if i == 0 and innermost:
                data = self.up_modules[i]((data, stack_down.pop()))
            else:
                data = self.up_modules[i]((data, stack_down.pop()), precomputed=self.upsample)

        last_feature = data.x
        self.output = self.FC_layer(last_feature)

        if self.labels is not None:
            self.compute_loss()

        self.data_visual = self.input
        self.data_visual.pred = torch.max(self.output, -1)[1]
        return self.output

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results
        self.loss_seg.backward()  # calculate gradients of network G w.r.t. loss_G
