from typing import List, Optional
import numpy as np
import torch
import random
from torch_geometric.data import Data
from classifierPoint.utils.geometry import euler_angles_to_rotation_matrix

from classifierPoint.core.spatial_ops.neighbour_finder import KNNNeighbourFinder


class RandomRotation(object):
    """
    Rotate pointcloud with random angles along x, y, z axis

    The angles should be given `in degrees`.

    Parameters
    -----------

    rot_x: float
        Rotation angle in degrees on x axis
    rot_y: float
        Rotation anglei n degrees on y axis
    rot_z: float
        Rotation angle in degrees on z axis
    """

    def __init__(
        self,
        rot_x: float = None,
        rot_y: float = None,
        rot_z: float = None,
    ):
        self._apply_rotation = True
        if (rot_x is None) and (rot_y is None) and (rot_z is None):
            self._apply_rotation = False
        self._rot_x = np.abs(rot_x) if rot_x else 0
        self._rot_y = np.abs(rot_y) if rot_y else 0
        self._rot_z = np.abs(rot_z) if rot_z else 0

        self._degree_angles = [self._rot_x, self._rot_y, self._rot_z]

    def generate_random_rotation_matrix(self):
        thetas = torch.zeros(3, dtype=torch.float)
        for axis_ind, deg_angle in enumerate(self._degree_angles):
            if deg_angle > 0:
                rand_deg_angle = random.random() * 2 * deg_angle - deg_angle
                rand_radian_angle = float(rand_deg_angle * np.pi) / 180.0
                thetas[axis_ind] = rand_radian_angle
        return euler_angles_to_rotation_matrix(thetas, random_order=True)

    def __call__(self, data):
        if self._apply_rotation:
            pos = data.pos.float()
            M = self.generate_random_rotation_matrix()
            data.pos = pos @ M.T
            if getattr(data, "norm", None) is not None:
                data.norm = data.norm.float() @ M.T
        return data

    def __repr__(self):
        return "{}(apply_rotation={}, rot_x={}, rot_y={}, rot_z={})".format(
            self.__class__.__name__,
            self._apply_rotation,
            self._rot_x,
            self._rot_y,
            self._rot_z,
        )


class RandomTranslation(object):
    """
    random translation
    Parameters
    -----------
    delta_min: list
        min translation
    delta_max: list
        max translation
    """

    def __init__(self, delta_max: List = [1.0, 1.0, 1.0], delta_min: List = [-1.0, -1.0, -1.0]):
        self.delta_max = torch.tensor(delta_max)
        self.delta_min = torch.tensor(delta_min)

    def __call__(self, data):
        pos = data.pos
        trans = torch.rand(3) * (self.delta_max - self.delta_min) + self.delta_min
        data.pos = pos + trans
        return data

    def __repr__(self):
        return "{}(delta_min={}, delta_max={})".format(self.__class__.__name__, self.delta_min, self.delta_max)


class AddFeatsByKeys(object):
    """This transform takes a list of attributes names and if allowed, add them to x

    Example:

        Before calling "AddFeatsByKeys", if data.x was empty

        - transform: AddFeatsByKeys
          params:
              list_control: [False, True, True]
              feat_names: ['normal', 'rgb', "elevation"]
              input_nc_feats: [3, 3, 1]

        After calling "AddFeatsByKeys", data.x contains "rgb" and "elevation". Its shape[-1] == 4 (rgb:3 + elevation:1)
        If input_nc_feats was [4, 4, 1], it would raise an exception as rgb dimension is only 3.

    Paremeters
    ----------
    list_control: List[bool]
        For each boolean within list_control, control if the associated feature is going to be concatenated to x
    feat_names: List[str]
        The list of features within data to be added to x
    input_nc_feats: List[int], optional
        If provided, evaluate the dimension of the associated feature shape[-1] found using feat_names and this provided value. It allows to make sure feature dimension didn't change
    stricts: List[bool], optional
        Recommended to be set to list of True. If True, it will raise an Exception if feat isn't found or dimension doesn t match.
    delete_feats: List[bool], optional
        Wether we want to delete the feature from the data object. List length must match teh number of features added.
    """

    def __init__(
        self,
        list_control: List[bool],
        feat_names: List[str],
        input_nc_feats: List[Optional[int]] = None,
        stricts: List[bool] = None,
        delete_feats: List[bool] = None,
    ):

        self._feat_names = feat_names
        self._list_control = list_control
        self._delete_feats = delete_feats
        if self._delete_feats:
            assert len(self._delete_feats) == len(self._feat_names)
        else:
            self._delete_feats = [True] * len(self._feat_names)
        from torch_geometric.transforms import Compose

        num_names = len(feat_names)
        if num_names == 0:
            raise Exception("Expected to have at least one feat_names")

        assert len(self._list_control) == num_names

        if input_nc_feats:
            assert len(input_nc_feats) == num_names
        else:
            input_nc_feats = [None for _ in range(num_names)]

        if stricts:
            assert len(stricts) == num_names
        else:
            stricts = [True for _ in range(num_names)]

        transforms = [
            AddFeatByKey(add_to_x, feat_name, input_nc_feat=input_nc_feat, strict=strict)
            for add_to_x, feat_name, input_nc_feat, strict in zip(list_control, feat_names, input_nc_feats, stricts)
        ]

        self.transform = Compose(transforms)

    def __call__(self, data):
        data = self.transform(data)
        if self._delete_feats:
            for feat_name, delete_feat in zip(self._feat_names, self._delete_feats):
                if delete_feat:
                    delattr(data, feat_name)
        return data

    def __repr__(self):
        msg = ""
        for f, a in zip(self._feat_names, self._list_control):
            msg += "{}={}, ".format(f, a)
        return "{}({})".format(self.__class__.__name__, msg[:-2])


class AddFeatByKey(object):
    """This transform is responsible to get an attribute under feat_name and add it to x if add_to_x is True

    Paremeters
    ----------
    add_to_x: bool
        Control if the feature is going to be added/concatenated to x
    feat_name: str
        The feature to be found within data to be added/concatenated to x
    input_nc_feat: int, optional
        If provided, check if feature last dimension maches provided value.
    strict: bool, optional
        Recommended to be set to True. If False, it won't break if feat isn't found or dimension doesn t match. (default: ``True``)
    """

    def __init__(self, add_to_x: bool, feat_name: str, input_nc_feat=None, strict: bool = True):

        self._add_to_x = add_to_x
        self._feat_name = feat_name
        self._input_nc_feat = input_nc_feat
        self._strict = strict

    def __call__(self, data: Data):
        if not self._add_to_x:
            return data
        feat = getattr(data, self._feat_name, None)
        if feat is None:
            if self._strict:
                raise Exception("Data should contain the attribute {}".format(self._feat_name))
            else:
                return data
        else:
            if self._input_nc_feat:
                feat_dim = 1 if feat.dim() == 1 else feat.shape[-1]
                if self._input_nc_feat != feat_dim and self._strict:
                    raise Exception("The shape of feat: {} doesn t match {}".format(feat.shape, self._input_nc_feat))
            x = getattr(data, "x", None)
            if x is None:
                if self._strict and data.pos.shape[0] != feat.shape[0]:
                    raise Exception("We expected to have an attribute x")
                if feat.dim() == 1:
                    feat = feat.unsqueeze(-1)
                data.x = feat
            else:
                if x.shape[0] == feat.shape[0]:
                    if x.dim() == 1:
                        x = x.unsqueeze(-1)
                    if feat.dim() == 1:
                        feat = feat.unsqueeze(-1)
                    data.x = torch.cat([x, feat], dim=-1)
                else:
                    raise Exception(
                        "The tensor x and {} can't be concatenated, x: {}, feat: {}".format(
                            self._feat_name, x.pos.shape[0], feat.pos.shape[0]
                        )
                    )
        return data

    def __repr__(self):
        return "{}(add_to_x: {}, feat_name: {}, strict: {})".format(
            self.__class__.__name__, self._add_to_x, self._feat_name, self._strict
        )


class AddAllFeat(object):
    BASEATTRIBUTE = ["y", "pos", "coords", "x", "file_name", "grid_size", "origin_id", "splitidx", "rgb", "intensity"]

    def __call__(self, data: Data):
        from torch_geometric.transforms import Compose

        data_num = data.pos.shape[0]
        feat_names = [
            att
            for att, _ in data._store.items()
            if not att in self.BASEATTRIBUTE and getattr(data, att, None).shape[0] == data_num
        ]
        transforms = [AddFeatByKey(add_to_x=True, feat_name=feat_name) for feat_name in feat_names]
        transforms = Compose(transforms)
        data = transforms(data)
        for feat_name in feat_names:
            delattr(data, feat_name)
        return data

    def __repr__(self):
        return "{} BASEATTRIBUTE: {}".format(self.__class__.__name__, self.BASEATTRIBUTE)


def compute_planarity(eigenvalues):
    r"""
    compute the planarity with respect to the eigenvalues of the covariance matrix of the pointcloud
    let
    :math:`\lambda_1, \lambda_2, \lambda_3` be the eigenvalues st:

    .. math::
        \lambda_1 \leq \lambda_2 \leq \lambda_3

    then planarity is defined as:

    .. math::
        planarity = \frac{\lambda_2 - \lambda_1}{\lambda_3}
    """

    return (eigenvalues[1] - eigenvalues[0]) / eigenvalues[2]


class NormalFeature(object):
    """
    add normal as feature. if it doesn't exist, compute normals
    using PCA
    """

    def __call__(self, data):
        if getattr(data, "norm", None) is None:
            raise NotImplementedError("TODO: Implement normal computation")

        norm = data.norm
        if data.x is None:
            data.x = norm
        else:
            data.x = torch.cat([data.x, norm], -1)
        return data


class PCACompute(object):
    r"""
    compute `Principal Component Analysis <https://en.wikipedia.org/wiki/Principal_component_analysis>`__ of a point cloud :math:`x_1,\dots, x_n`.
    It computes the eigenvalues and the eigenvectors of the matrix :math:`C` which is the covariance matrix of the point cloud:

    .. math::
        x_{centered} &= \frac{1}{n} \sum_{i=1}^n x_i

        C &= \frac{1}{n} \sum_{i=1}^n (x_i - x_{centered})(x_i - x_{centered})^T

    store the eigen values and the eigenvectors in data.
    in eigenvalues attribute and eigenvectors attributes.
    data.eigenvalues is a tensor :math:`(\lambda_1, \lambda_2, \lambda_3)` such that :math:`\lambda_1 \leq \lambda_2 \leq \lambda_3`.

    data.eigenvectors is a 3 x 3 matrix such that the column are the eigenvectors associated to their eigenvalues
    Therefore, the first column of data.eigenvectors estimates the normal at the center of the pointcloud.
    """

    def __call__(self, data):
        pos_centered = data.pos - data.pos.mean(axis=0)
        cov_matrix = pos_centered.T.mm(pos_centered) / len(pos_centered)
        eig, v = torch.symeig(cov_matrix, eigenvectors=True)
        data.eigenvalues = eig
        data.eigenvectors = v
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class OnesFeature(object):
    """
    Add ones tensor to data
    """

    def __call__(self, data):
        num_nodes = data.pos.shape[0]
        data.ones = torch.ones((num_nodes, 1)).float()
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


class XYZFeature(object):
    """
    Add the X, Y and Z as a feature
    Parameters
    -----------
    x: bool [default: False]
        whether we add the x position or not
    y: bool [default: False]
        whether we add the y position or not
    z: bool [default: True]
        whether we add the z position or not
    """

    def __init__(self, x=True, y=True, z=True):
        self._axis = []
        # axis_names = ["x", "y", "z"]
        if x:
            self._axis.append(0)
        if y:
            self._axis.append(1)
        if z:
            self._axis.append(2)

        # self._axis_names = [axis_names[idx_axis] for idx_axis in self._axis]
    def _norm(self,data):
        #标准化xyz
        # 沿着列的方向计算最小值和最大值
        min_vals, _ = torch.min(data, dim=0, keepdim=True)
        max_vals, _ = torch.max(data, dim=0, keepdim=True)
        # 最小-最大缩放，将x的范围缩放到[0, 1]
        scaled_data = (data - min_vals) / (max_vals - min_vals)
        return scaled_data
    def _norm2(self,data):
        #按照长轴归一化
        centroid = torch.mean(data, axis=0) # 求取点云的中心
        data = data - centroid # 将点云中心置于原点 (0, 0, 0)
        m = torch.max(torch.sqrt(torch.sum(data ** 2, axis=1))) # 求取长轴的的长度
        data_normalized = data / m # 依据长轴将点云归一化到 (-1, 1)  centroid: 点云中心, m: 长轴长度, centroid和m可用于keypoints的计算
        data = (data - centroid) / m
        return data   
    def centralize_and_normalize_tensor(self,points):
        # 中心化
        centroid = points.min(0)[0]
        centered_points = points - centroid

        # 归一化
        # max_distance = torch.max(torch.sqrt(torch.sum(centered_points**2, dim=1)))
        # normalized_points = centered_points / max_distance

        return centered_points
    def __call__(self, data):
        assert data.pos is not None
        # for axis_name, id_axis in zip(self._axis_names, self._axis):
        #     f = data.pos[:, id_axis].clone()
        #     setattr(data, "pos_{}".format(axis_name), f)
        f = data.pos[:, self._axis].clone()
        ##都在这儿切片
        # f = self._norm2(f)[:,2]
        ####

        #
        f = f[:,2].unsqueeze(1)
        f = self._norm2(f)
        #
        # f = self.centralize_and_normalize_tensor(f)
        setattr(data, "xyz", f)
        return data

    def __repr__(self):
        return "{}(axis={})".format(self.__class__.__name__, self._axis_names)


class RGBFeature(object):
    """
    Add RGB as a feature
    Parameters

    """

    def __call__(self, data):
        transforms = AddFeatByKey(add_to_x=True, feat_name="rgb")
        data = transforms(data)
        # 添加到data.x中 删除重复属性
        delattr(data, "rgb")
        return data

    def __repr__(self):
        return "{}(axis={})".format(self.__class__.__name__, self._axis_names)


class IntensityFeature(object):
    """
    Add Intensity as a feature
    Parameters
    """

    def __call__(self, data):
        transforms = AddFeatByKey(add_to_x=True, feat_name="intensity")
        data = transforms(data)
        # 添加到data.x中 删除重复属性
        delattr(data, "intensity")
        return data

    def __repr__(self):
        return "{}(axis={})".format(self.__class__.__name__, self._axis_names)


class LGAFeature(object):
    """
    Add the  Local Geometric Anisotropy  as a feature
    impl for https://arxiv.org/pdf/2001.10709.pdf PAlos
    Parameters
    -----------
    K: int [default: 7]
        6 Neighbour and itself
    """

    def __init__(self, K=7):

        self.K = K
        self.knn_finder = KNNNeighbourFinder(self.K)

    def __call__(self, data):
        assert data.pos is not None

        neighbour = self.knn_finder(data.pos, data.pos, None, None)
        neighbour = data.origin_id[neighbour[1]]
        neighbour = neighbour.view(data.pos.shape[0], self.K)
        label_clone = data.y.clone()
        target_lga = label_clone[neighbour]
        target_mask = target_lga - target_lga[:, 0:1]
        lga = (target_mask != 0).sum(1)
        data.lga = lga
        return data

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)
