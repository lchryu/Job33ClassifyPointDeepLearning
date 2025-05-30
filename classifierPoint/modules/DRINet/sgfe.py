import torch.nn as nn
from typing import List
from torchsparse import SparseTensor
import torch
from torchsparse.nn import functional as F
from torchsparse.nn.utils import get_kernel_offsets


class AvgPooling(nn.Module):
    """avg pool"""

    def __init__(self, stride) -> None:
        super().__init__()
        self.stride = stride

    def forward(self, input: SparseTensor) -> SparseTensor:
        new_float_coord = torch.cat([input.C[:, :3] / self.stride, input.C[:, -1].view(-1, 1)], 1)
        pc_hash = F.sphash(torch.floor(new_float_coord).int())
        sparse_hash = torch.unique(pc_hash)
        idx_query = F.sphashquery(pc_hash, sparse_hash)

        counts = F.spcount(idx_query.int(), len(sparse_hash))
        inserted_feat = F.spvoxelize(input.F, idx_query, counts)

        inserted_coords = F.spvoxelize(input.C.float(), idx_query, counts)
        inserted_coords = torch.round(inserted_coords).int()

        output = SparseTensor(coords=inserted_coords, feats=inserted_feat, stride=self.stride)
        return output


def Upsample(input: SparseTensor, origin: SparseTensor, stride: int, nearest: bool = True) -> SparseTensor:
    """Upsample"""
    off = get_kernel_offsets(2, stride, 1, device=input.F.device)
    old_hash = F.sphash(
        torch.cat(
            [torch.floor(origin.C[:, :3] / stride).int() * stride, origin.C[:, -1].int().view(-1, 1)],
            1,
        ),
        off,
    )
    pc_hash = F.sphash(
        torch.cat(
            [torch.floor(input.C[:, :3] / stride).int() * stride, input.C[:, -1].int().view(-1, 1)],
            1,
        )
    )
    idx_query = F.sphashquery(old_hash, pc_hash)
    weights = F.calc_ti_weights(origin.C, idx_query, scale=stride).transpose(0, 1).contiguous()
    idx_query = idx_query.transpose(0, 1).contiguous()
    if nearest:
        max_idx = torch.argmax(weights, dim=1).unsqueeze(1)
        weights[:, :] = 0
        weights = weights.scatter(dim=1, index=max_idx, src=torch.ones_like(weights, device=input.F.device))
    new_feat = F.spdevoxelize(input.F, idx_query, weights)
    output = SparseTensor(coords=origin.C, feats=new_feat)
    return output


class MSP(nn.Module):
    """Multi-scale Sparse Projection layer"""

    def __init__(self, stride: int, inputc: int, outc: int) -> None:
        super().__init__()
        self.stride = stride
        self.avgpooling = AvgPooling(stride)
        self.mlp = nn.Sequential(nn.Linear(inputc, inputc), nn.ReLU(inplace=True))
        self.mlp2 = nn.Sequential(nn.Linear(inputc, outc), nn.ReLU(inplace=True))

    def forward(self, x: SparseTensor) -> SparseTensor:
        vs = self.avgpooling(x)
        os = SparseTensor(feats=x.F, coords=x.C)
        os = os - Upsample(vs, x, self.stride)
        os.F = self.mlp(os.F) * x.F
        os.F = self.mlp2(os.F)
        return os


class AMF(nn.Module):
    """Attentive Multi-scale Fusion"""

    def __init__(self, scale_num: int, inputc: int, outc: int) -> None:
        super().__init__()
        self.scale_num = scale_num
        self.attention_module = nn.ModuleList(
            [nn.Sequential(nn.Linear(inputc, 1), nn.Sigmoid()) for i in torch.arange(scale_num)]
        )

    def forward(self, x):
        """x [M,C,scale_nums]"""
        sum_features = torch.sum(x, axis=2)
        scale_features = torch.zeros_like(x[:, :, 0])
        for i, attention in enumerate(self.attention_module):
            scale_features += x[:, :, i] * attention(sum_features)
        return scale_features


class SGFE(nn.Module):
    """Sparse Geometry Feature Enhancement"""

    def __init__(self, scale_list: List, intc: int) -> None:
        super().__init__()
        self.msp_module = nn.ModuleList([MSP(scale, intc, intc) for scale in scale_list])
        self.amf = AMF(len(scale_list), intc, intc)

    def forward(self, x: SparseTensor) -> SparseTensor:
        msp_featrue = torch.stack([msp(x).F for msp in self.msp_module], dim=2)
        amf_feature = self.amf(msp_featrue)
        out = SparseTensor(feats=amf_feature, coords=x.C, stride=x.s)
        return out
