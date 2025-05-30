import torch.nn as nn
import torchsparse.nn as spnn


class Bottleneck(nn.Module):
    def __init__(self, inc):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, inc, kernel_size=1, dilation=1, stride=1),
            spnn.GroupNorm(8, inc),
            spnn.LeakyReLU(True),
            spnn.Conv3d(inc, inc, kernel_size=3, dilation=1, stride=2),
            spnn.GroupNorm(8, inc),
            # spnn.LeakyReLU(True),# convnext fewer activation
            spnn.Conv3d(inc, inc, kernel_size=1, dilation=1, stride=1),
            spnn.GroupNorm(8, inc),
        )

        self.downsample = nn.Sequential(
            spnn.Conv3d(inc, inc, kernel_size=1, dilation=1, stride=2),
            spnn.GroupNorm(8, inc),
        )

    def forward(self, x):
        out = self.net(x) + self.downsample(x)
        return out


class BasicConvolutionBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            # spnn.BatchNorm(outc),
            spnn.GroupNorm(8, outc),
            spnn.LeakyReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            # spnn.BatchNorm(outc),
            spnn.GroupNorm(8, outc),
            # spnn.LeakyReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation, stride=1),
            # spnn.BatchNorm(outc),
            spnn.GroupNorm(8, outc),
        )

        self.downsample = (
            nn.Sequential()
            if (inc == outc and stride == 1)
            else nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1, stride=stride),
                # spnn.BatchNorm(outc),
                spnn.GroupNorm(8, outc),
            )
        )

        self.relu = spnn.LeakyReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out


class SFE(nn.Module):
    """Sparse Feature Encoder"""

    def __init__(self, inc) -> None:
        super().__init__()
        self.block = nn.Sequential(
            BasicConvolutionBlock(inc, inc, ks=2, stride=2, dilation=1),
            ResidualBlock(inc, inc, ks=3, stride=1, dilation=1),
            ResidualBlock(inc, inc, ks=3, stride=1, dilation=1),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class SFEOrigin(nn.Module):
    """Sparse Feature Encoder"""

    def __init__(self, inc) -> None:
        super().__init__()
        self.bottleneck = Bottleneck(inc)

    def forward(self, x):
        x = self.bottleneck(x)
        return x
