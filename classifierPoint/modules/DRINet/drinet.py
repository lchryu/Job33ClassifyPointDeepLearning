from classifierPoint.modules.DRINet.sfe import SFE, SFEOrigin
from classifierPoint.modules.DRINet.sgfe import SGFE, Upsample
import torch.nn as nn
import torchsparse.nn as spnn
import torch


class DRINet(nn.Module):
    def __init__(self, option, dataset):
        super().__init__()
        self.block_num = option.block_num
        self.num_classes = dataset.num_classes
        self.scale_list = option.scale_list
        cs = [64, 64, 64, 64, 64]
        cs = [int(option.cr * x) for x in cs]

        self.stem = nn.Sequential(
            spnn.Conv3d(dataset.feature_dimension, cs[0], kernel_size=3, stride=1),
            #spnn.Conv3d(1, cs[0], kernel_size=3, stride=1),
            spnn.GroupNorm(8, cs[0]),
            spnn.LeakyReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.GroupNorm(8, cs[0]),
            spnn.LeakyReLU(True),
        )
        self.block_modulelist = nn.ModuleList(
            [
                nn.Sequential(
                    SFE(cs[i]),
                    SGFE(self.scale_list, cs[i]),
                )
                for i in torch.arange(self.block_num)
            ]
        )
        self.classifier = nn.Linear(cs[0] * self.block_num, self.num_classes)

    def forward(self, x):
        x0 = self.stem(x)
        out = []
        for i, block in enumerate(self.block_modulelist):
            x0 = block(x0)
            out.append(Upsample(x0, x, stride=2 ** (i + 1)).F)

        out = torch.cat(out, dim=1)
        self.output = self.classifier(out)
        return self.output
