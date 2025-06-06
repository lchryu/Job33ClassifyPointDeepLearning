from collections import OrderedDict, deque

import torch
import torch.nn as nn

import torchsparse
import torchsparse.nn as spnn
import torchsparse.nn.functional as spf
from torchsparse import SparseTensor, PointTensor
from torchsparse.utils import *

# from torchsparse.utils.helpers import *


from classifierPoint.modules.PVCNN.utils import *
from classifierPoint.modules.PVCNN.layers import *
from classifierPoint.modules.PVCNN.networks import *


class SPVNAS(RandomNet):
    base_channels = 32
    # [base_channels, 32, 64, 128, 256, 256, 128, 96, 96]
    output_channels_lb = [base_channels, 16, 32, 64, 128, 128, 64, 48, 48]
    output_channels = [base_channels, 48, 96, 192, 384, 384, 192, 128, 128]
    max_macro_depth = 2
    max_micro_depth = 2
    num_down_stages = len(output_channels) // 2

    def __init__(self, input_dims, num_classes, macro_depth_constraint, **kwargs):
        super().__init__()
        self.cr_bounds = [0.125, 1.0] if "cr_bounds" not in kwargs else kwargs["cr_bounds"]
        self.up_cr_bounds = [0.125, 1.0] if "up_cr_bounds" not in kwargs else kwargs["up_cr_bounds"]
        self.trans_cr_bounds = [0.125, 1.0] if "trans_cr_bounds" not in kwargs else kwargs["trans_cr_bounds"]

        if "output_channels_ub" not in kwargs:
            self.output_channels_ub = self.output_channels
        else:
            self.output_channels_ub = kwargs["output_channels_ub"]

        if "output_channels_lb" in kwargs:
            self.output_channels_lb = kwargs["output_channels_lb"]

        base_channels = self.base_channels
        output_channels = self.output_channels
        output_channels_lb = self.output_channels_lb

        self.stem = nn.Sequential(
            spnn.Conv3d(input_dims, base_channels, kernel_size=3, stride=1),
            # spnn.BatchNorm(base_channels),
            spnn.GroupNorm(8, base_channels),
            spnn.ReLU(True),
            spnn.Conv3d(base_channels, base_channels, kernel_size=3, stride=1),
            # spnn.BatchNorm(base_channels),
            spnn.GroupNorm(8, base_channels),
            spnn.ReLU(True),
        )

        num_down_stages = self.num_down_stages

        stages = []
        for i in range(1, num_down_stages + 1):
            stages.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "transition",
                                DynamicConvolutionBlock(
                                    base_channels,
                                    base_channels,
                                    cr_bounds=self.trans_cr_bounds,
                                    ks=2,
                                    stride=2,
                                    dilation=1,
                                ),
                            ),
                            (
                                "feature",
                                RandomDepth(
                                    *[
                                        DynamicResidualBlock(
                                            base_channels,
                                            output_channels[i],
                                            cr_bounds=self.cr_bounds,
                                            ks=3,
                                            stride=1,
                                            dilation=1,
                                        ),
                                        DynamicResidualBlock(
                                            output_channels[i],
                                            output_channels[i],
                                            cr_bounds=self.cr_bounds,
                                            ks=3,
                                            stride=1,
                                            dilation=1,
                                        ),
                                    ],
                                    depth_min=macro_depth_constraint
                                ),
                            ),
                        ]
                    )
                )
            )
            base_channels = output_channels[i]

        self.downsample = nn.ModuleList(stages)

        # take care of weight sharing after concat!
        upstages = []
        for i in range(1, num_down_stages + 1):
            new_base_channels = output_channels[num_down_stages + i]
            upstages.append(
                nn.Sequential(
                    OrderedDict(
                        [
                            (
                                "transition",
                                DynamicDeconvolutionBlock(
                                    base_channels,
                                    new_base_channels,
                                    cr_bounds=self.up_cr_bounds,
                                    ks=2,
                                    stride=2,
                                ),
                            ),
                            (
                                "feature",
                                RandomDepth(
                                    *[
                                        DynamicResidualBlock(
                                            output_channels[num_down_stages - i] + new_base_channels,
                                            new_base_channels,
                                            cr_bounds=self.up_cr_bounds,
                                            ks=3,
                                            stride=1,
                                            dilation=1,
                                        ),
                                        DynamicResidualBlock(
                                            new_base_channels,
                                            new_base_channels,
                                            cr_bounds=self.up_cr_bounds,
                                            ks=3,
                                            stride=1,
                                            dilation=1,
                                        ),
                                    ],
                                    depth_min=macro_depth_constraint
                                ),
                            ),
                        ]
                    )
                )
            )
            base_channels = new_base_channels

        self.upsample = nn.ModuleList(upstages)

        self.point_transforms = nn.ModuleList(
            [
                DynamicLinearBlock(
                    output_channels[0],
                    output_channels[num_down_stages],
                    bias=True,
                    no_relu=False,
                    no_bn=False,
                ),
                DynamicLinearBlock(
                    output_channels[num_down_stages],
                    output_channels[num_down_stages + 2],
                    bias=True,
                    no_relu=False,
                    no_bn=False,
                ),
                DynamicLinearBlock(
                    output_channels[num_down_stages + 2],
                    output_channels[-1],
                    bias=True,
                    no_relu=False,
                    no_bn=False,
                ),
            ]
        )

        self.classifier = DynamicLinear(output_channels[-1], num_classes)
        self.classifier.set_output_channel(num_classes)

        self.dropout = nn.Dropout(0.3, True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def random_sample(self):
        sample = {}
        # sample layer configuration
        cur_outputs_channels = []
        for i in range(len(self.output_channels)):
            c = random.randint(self.output_channels_lb[i], self.output_channels_ub[i])
            c = make_divisible(c)
            cur_outputs_channels.append(c)
        self.cur_outputs_channels = cur_outputs_channels
        sample["output_channels"] = cur_outputs_channels

        # fix point branch
        self.point_transforms[0].manual_select(self.cur_outputs_channels[self.num_down_stages])
        self.point_transforms[1].manual_select(self.cur_outputs_channels[self.num_down_stages + 2])
        self.point_transforms[2].manual_select(self.cur_outputs_channels[-1])

        # sample down blocks
        # all residual blocks, except the first one, must have inc = outc
        for i in range(len(self.downsample)):
            # sample output channels for transition block
            self.downsample[i].transition.random_sample()
            # sample depth
            cur_depth = self.downsample[i].feature.random_sample()

            # random sample each residual block
            for j in range(cur_depth):
                # random sample middile layers
                self.downsample[i].feature.layers[j].random_sample()
                # determine the output channel
                self.downsample[i].feature.layers[j].constrain_output_channel(cur_outputs_channels[i + 1])

            for j in range(cur_depth, len(self.downsample[i].feature.layers)):
                self.downsample[i].feature.layers[j].clear_sample()

        # sample up blocks
        for i in range(len(self.upsample)):
            # sample output channels for transition block
            trans_output_channels = self.upsample[i].transition.random_sample()
            # sample depth
            cur_depth = self.upsample[i].feature.random_sample()
            # random sample each residual block
            for j in range(cur_depth):

                self.upsample[i].feature.layers[j].random_sample()
                self.upsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[len(self.downsample) + 1 + i]
                )
                # special case: 1st layer for 1st residual block (because of concat)
                if j == 0:
                    cons = list(range(trans_output_channels)) + list(
                        range(
                            self.output_channels[len(self.downsample) + i + 1],
                            self.output_channels[len(self.downsample) + i + 1]
                            + cur_outputs_channels[len(self.downsample) - 1 - i],
                        )
                    )
                    self.upsample[i].feature.layers[j].net.layers[0].constrain_in_channel(cons)
                    self.upsample[i].feature.layers[j].downsample.constrain_in_channel(cons)

            for j in range(cur_depth, len(self.upsample[i].feature.layers)):
                self.upsample[i].feature.layers[j].clear_sample()

        for name, module in self.named_random_modules():
            try:
                cur_val = module.status()
                sample[name] = cur_val
            except:
                # random depth, ignored layer
                pass

        return sample

    def manual_select(self, sample):
        for name, module in self.named_random_modules():
            if sample[name] is not None:
                module.manual_select(sample[name])

        cur_outputs_channels = copy.deepcopy(sample["output_channels"])

        # fix point branch
        self.point_transforms[0].manual_select(cur_outputs_channels[self.num_down_stages])
        self.point_transforms[1].manual_select(cur_outputs_channels[self.num_down_stages + 2])
        self.point_transforms[2].manual_select(cur_outputs_channels[-1])

        for i in range(len(self.downsample)):
            for j in range(self.downsample[i].feature.depth):
                self.downsample[i].feature.layers[j].constrain_output_channel(cur_outputs_channels[i + 1])

        for i in range(len(self.upsample)):
            trans_output_channels = self.upsample[i].transition.status()
            for j in range(self.upsample[i].feature.depth):
                self.upsample[i].feature.layers[j].constrain_output_channel(
                    cur_outputs_channels[len(self.downsample) + 1 + i]
                )
                # special case: 1st layer for 1st residual block (because of concat)
                if j == 0:
                    cons = list(range(trans_output_channels)) + list(
                        range(
                            self.output_channels[len(self.downsample) + i + 1],
                            self.output_channels[len(self.downsample) + i + 1]
                            + cur_outputs_channels[len(self.downsample) - 1 - i],
                        )
                    )
                    self.upsample[i].feature.layers[j].net.layers[0].constrain_in_channel(cons)
                    self.upsample[i].feature.layers[j].downsample.constrain_in_channel(cons)

        self.cur_outputs_channels = cur_outputs_channels

    def determinize(self, input_dims, local_rank=0):
        # Get the determinized SPVNAS network by running dummy inference.
        self = self.to("cuda:%d" % local_rank)
        self.eval()
    
        sample_feat = torch.randn(1000, input_dims)
        sample_coord = torch.randn(1000, 4).random_(997)
        sample_coord[:, -1] = 0
        # x = SparseTensor(sample_feat,
        #                 sample_coord.int()).to('cuda:%d' % local_rank)
        if torch.cuda.is_available():
            x = SparseTensor(sample_feat, sample_coord.int()).to("cuda:%d" % local_rank)
        else:
            x = SparseTensor(sample_feat, sample_coord.int())
        with torch.no_grad():
            x = self.forward(x)

        model = copy.deepcopy(self)

        queue = deque([model])
        while queue:
            x = queue.popleft()
            for name, module in x._modules.items():
                while isinstance(module, RandomModule):
                    module = x._modules[name] = module.determinize()
                queue.append(module)

        return model

    def forward(self, x):
        # x: SparseTensor z: PointTensor
        z = PointTensor(x.F, x.C.float())
        x0 = point_to_voxel(x, z)

        x0 = self.stem(x0)
        z0 = voxel_to_point(x0, z)
        z0.F = z0.F  # + self.point_transforms[0](z.F)

        x1 = point_to_voxel(x0, z0)
        x1 = self.downsample[0](x1)
        x2 = self.downsample[1](x1)
        x3 = self.downsample[2](x2)
        x4 = self.downsample[3](x3)

        # point transform 32 to 256
        z1 = voxel_to_point(x4, z0)
        z1.F = z1.F + self.point_transforms[0](z0.F)

        y1 = point_to_voxel(x4, z1)
        y1.F = self.dropout(y1.F)
        y1 = self.upsample[0].transition(y1)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.upsample[0].feature(y1)

        # print('y1', y1.C)
        y2 = self.upsample[1].transition(y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.upsample[1].feature(y2)
        # point transform 256 to 128
        z2 = voxel_to_point(y2, z1)
        z2.F = z2.F + self.point_transforms[1](z1.F)

        y3 = point_to_voxel(y2, z2)
        y3.F = self.dropout(y3.F)
        y3 = self.upsample[2].transition(y3)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.upsample[2].feature(y3)

        y4 = self.upsample[3].transition(y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.upsample[3].feature(y4)
        z3 = voxel_to_point(y4, z2)
        z3.F = z3.F + self.point_transforms[2](z2.F)

        self.classifier.set_in_channel(z3.F.shape[-1])
        out = self.classifier(z3.F)

        return out


def make_model(dataset):
    import json

    net_config = json.loads(
        """{
            "output_channels": [32, 28, 40, 76, 156, 132, 76, 60, 48],
            "downsample.0.transition": 20,
            "downsample.0.feature": 2,
            "downsample.0.feature.layers.0.net": 2,
            "downsample.0.feature.layers.0.net.layers.0": 28,
            "downsample.0.feature.layers.0.net.layers.1": 28,
            "downsample.0.feature.layers.0.downsample": 28,
            "downsample.0.feature.layers.1.net": 2,
            "downsample.0.feature.layers.1.net.layers.0": 12,
            "downsample.0.feature.layers.1.net.layers.1": 28,
            "downsample.1.transition": 28,
            "downsample.1.feature": 2,
            "downsample.1.feature.layers.0.net": 2,
            "downsample.1.feature.layers.0.net.layers.0": 40,
            "downsample.1.feature.layers.0.net.layers.1": 40,
            "downsample.1.feature.layers.0.downsample": 40,
            "downsample.1.feature.layers.1.net": 2,
            "downsample.1.feature.layers.1.net.layers.0": 40,
            "downsample.1.feature.layers.1.net.layers.1": 40,
            "downsample.2.transition": 56,
            "downsample.2.feature": 1,
            "downsample.2.feature.layers.0.net": 2,
            "downsample.2.feature.layers.0.net.layers.0": 96,
            "downsample.2.feature.layers.0.net.layers.1": 76,
            "downsample.2.feature.layers.0.downsample": 76,
            "downsample.2.feature.layers.1.net": null,
            "downsample.2.feature.layers.1.net.layers.0": 120,
            "downsample.2.feature.layers.1.net.layers.1": 96,
            "downsample.3.transition": 116,
            "downsample.3.feature": 2,
            "downsample.3.feature.layers.0.net": 2,
            "downsample.3.feature.layers.0.net.layers.0": 100,
            "downsample.3.feature.layers.0.net.layers.1": 156,
            "downsample.3.feature.layers.0.downsample": 156,
            "downsample.3.feature.layers.1.net": 2,
            "downsample.3.feature.layers.1.net.layers.0": 100,
            "downsample.3.feature.layers.1.net.layers.1": 156,
            "upsample.0.transition": 124,
            "upsample.0.feature": 1,
            "upsample.0.feature.layers.0.net": 2,
            "upsample.0.feature.layers.0.net.layers.0": 108,
            "upsample.0.feature.layers.0.net.layers.1": 132,
            "upsample.0.feature.layers.0.downsample": 132,
            "upsample.0.feature.layers.1.net": null,
            "upsample.0.feature.layers.1.net.layers.0": 196,
            "upsample.0.feature.layers.1.net.layers.1": 232,
            "upsample.1.transition": 96,
            "upsample.1.feature": 2,
            "upsample.1.feature.layers.0.net": 2,
            "upsample.1.feature.layers.0.net.layers.0": 64,
            "upsample.1.feature.layers.0.net.layers.1": 76,
            "upsample.1.feature.layers.0.downsample": 76,
            "upsample.1.feature.layers.1.net": 2,
            "upsample.1.feature.layers.1.net.layers.0": 48,
            "upsample.1.feature.layers.1.net.layers.1": 76,
            "upsample.2.transition": 36,
            "upsample.2.feature": 1,
            "upsample.2.feature.layers.0.net": 2,
            "upsample.2.feature.layers.0.net.layers.0": 44,
            "upsample.2.feature.layers.0.net.layers.1": 60,
            "upsample.2.feature.layers.0.downsample": 60,
            "upsample.2.feature.layers.1.net": 2,
            "upsample.2.feature.layers.1.net.layers.0": 32,
            "upsample.2.feature.layers.1.net.layers.1": 56,
            "upsample.3.transition": 32,
            "upsample.3.feature": 1,
            "upsample.3.feature.layers.0.net": 2,
            "upsample.3.feature.layers.0.net.layers.0": 40,
            "upsample.3.feature.layers.0.net.layers.1": 48,
            "upsample.3.feature.layers.0.downsample": 48,
            "upsample.3.feature.layers.1.net": 2,
            "upsample.3.feature.layers.1.net.layers.0": 64,
            "upsample.3.feature.layers.1.net.layers.1": 68,
            "point_transforms.0": 156,
            "point_transforms.1": 76,
            "point_transforms.2": 48,
            "num_classes": 19
        }"""
    )
    net_config["num_classes"] = dataset.num_classes
    model = SPVNAS(dataset.feature_dimension, net_config["num_classes"], macro_depth_constraint=1)
    model.manual_select(net_config)
    model = model.determinize(dataset.feature_dimension)
    return model
