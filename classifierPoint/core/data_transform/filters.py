import numpy as np
import random
from scipy.spatial import KDTree
from classifierPoint.core.data_transform.features import (
    PCACompute,
    compute_planarity,
)
from classifierPoint.core.data_transform.grid_transform import group_data

import pdal
import json
import torch


def struct_data(np_data):
    tmp = np.c_[np_data, np.arange(np_data.shape[0])]
    new = np.array(
        [tuple(line) for line in tmp],
        dtype=[("X", np.float64), ("Y", np.float64), ("Z", np.float64), ("mask", np.int)],
    )
    return new


class Splitter(object):
    """The Splitter Filter breaks a point cloud into square tiles of a specified size.
    Parameters
    ----------
    length
    Length of the sides of the tiles that are created to hold points. [Default: 1000]

    origin_x
    X Origin of the tiles. [Default: none (chosen arbitrarily)]

    origin_y
    Y Origin of the tiles. [Default: none (chosen arbitrarily)]

    buffer
    Amount of overlap to include in each tile. This buffer is added onto length in both the x and the y direction. [Default: 0]
    """

    def __init__(self, length, origin_x=None, origin_y=None, buffer=0):
        self._length = length
        self._origin = [origin_x, origin_y]
        self._buffer = buffer
        self._dict = {
            "type": "filters.splitter",
            "length": length,
            "buffer": buffer,
        }
        if origin_x != None:
            self._dict["origin_x"] = origin_x
        if origin_y != None:
            self._dict["origin_y"] = origin_y
        self.json_pipeline = json.dumps([self._dict])

    def __call__(self, data):
        coords = data.pos
        tmp_data = struct_data(coords)
        pipeline = pdal.Pipeline(self.json_pipeline, arrays=[tmp_data])
        pipeline.execute()
        arrays = pipeline.arrays
        data_list = []
        data_dict = {"data": data_list}
        if hasattr(data, "origin_id"):
            data_dict.update({"origin_id": data.origin_id})
        for array in arrays:
            tmp = data.clone()
            delattr(tmp, "origin_id")
            mask = np.asarray(array.tolist())[:, -1].astype(np.int)
            split_data = group_data(data=tmp, unique_pos_indices=mask, mode="last")
            split_data.splitidx = torch.tensor(mask)
            data_list.append(split_data)
        return data_dict

    def __repr__(self):
        return "{}(length={}, origin={}, buffer={})".format(
            self.__class__.__name__, self._length, self._origin, self._buffer
        )


class Chipper(object):
    """The Chipper Filter takes a single large point cloud and converts it into a set of smaller clouds, or chips. The chips are all spatially contiguous and non-overlapping, so the result is a an irregular tiling of the input data.
    Parameters
    ----------
    capacity
    How many points to fit into each chip. The number of points in each chip will not exceed this value, and will sometimes be less than it. [Default: 5000]
    """

    def __init__(self, capacity=5000):
        self._capacity = capacity

        self.json_pipeline = json.dumps(
            [
                {
                    "type": "filters.chipper",
                    "capacity": capacity,
                }
            ]
        )

    def __call__(self, data):
        coords = data.pos
        tmp_data = struct_data(coords)
        pipeline = pdal.Pipeline(self.json_pipeline, arrays=[tmp_data])
        pipeline.execute()
        arrays = pipeline.arrays
        data_list = []
        data_dict = {"data": data_list}
        if hasattr(data, "origin_id"):
            data_dict.update({"origin_id": data.origin_id})
        for array in arrays:
            tmp = data.clone()
            delattr(tmp, "origin_id")
            mask = np.asarray(array.tolist())[:, -1].astype(np.int)
            split_data = group_data(data=tmp, unique_pos_indices=mask, mode="last")
            split_data.splitidx = torch.tensor(mask)
            data_list.append(split_data)
        return data_dict

    def __repr__(self):
        return "{}(capacity={})".format(self.__class__.__name__, self._capacity)


class FCompose(object):
    """
    allow to compose different filters using the boolean operation

    Parameters
    ----------
    list_filter: list
        list of different filter functions we want to apply
    boolean_operation: function, optional
        boolean function to compose the filter (take a pair and return a boolean)
    """

    def __init__(self, list_filter, boolean_operation=np.logical_and):
        self.list_filter = list_filter
        self.boolean_operation = boolean_operation

    def __call__(self, data):
        assert len(self.list_filter) > 0
        res = self.list_filter[0](data)
        for filter_fn in self.list_filter:
            res = self.boolean_operation(res, filter_fn(data))
        return res

    def __repr__(self):
        rep = "{}([".format(self.__class__.__name__)
        for filt in self.list_filter:
            rep = rep + filt.__repr__() + ", "
        rep = rep + "])"
        return rep


class StatisticalFilter(object):
    """
    StatisticalOutlierRemoval
    Parameters
    ----------
    k: int
        k nearest neighbors
    threshold: float
        std threshold
    """

    def __init__(self, k: int = 10, threshold: float = 5):
        self._k = k
        self._threshold = threshold

    def __call__(self, data):
        kdtree = KDTree(data.pos)
        dist, _ = kdtree.query(data.pos, k=self._k)
        mean = np.mean(dist, axis=1)
        std = np.std(mean, axis=0)
        all_mean = np.mean(mean)
        idx = np.where(mean < all_mean + self._threshold * std)
        data = group_data(data, unique_pos_indices=idx, mode="last")
        return data

    def __repr__(self):
        rep = "{}(k={},threshold={})".format(self.__class__.__name__, self._k, self._threshold)
        return rep


class PlanarityFilter(object):
    """
    compute planarity and return false if the planarity of a pointcloud is above or below a threshold

    Parameters
    ----------
    thresh: float, optional
        threshold to filter low planar pointcloud
    is_leq: bool, optional
        choose whether planarity should be lesser or equal than the threshold or greater than the threshold.
    """

    def __init__(self, thresh=0.3, is_leq=True):
        self.thresh = thresh
        self.is_leq = is_leq

    def __call__(self, data):
        if getattr(data, "eigenvalues", None) is None:
            data = PCACompute()(data)
        planarity = compute_planarity(data.eigenvalues)
        if self.is_leq:
            return planarity <= self.thresh
        else:
            return planarity > self.thresh

    def __repr__(self):
        return "{}(thresh={}, is_leq={})".format(self.__class__.__name__, self.thresh, self.is_leq)


class RandomFilter(object):
    """
    Randomly select an elem of the dataset (to have smaller dataset) with a bernouilli distribution of parameter thresh.

    Parameters
    ----------
    thresh: float, optional
        the parameter of the bernouilli function
    """

    def __init__(self, thresh=0.3):
        self.thresh = thresh

    def __call__(self, data):
        return random.random() < self.thresh

    def __repr__(self):
        return "{}(thresh={})".format(self.__class__.__name__, self.thresh)
