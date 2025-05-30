from classifierPoint.core.data_transform import VoxelGrid


def check_VoxelGrid(dataset):
    for transform in dataset.pre_transform.transforms:
        if isinstance(transform, VoxelGrid):
            return True
    for transform in dataset.helper_transform.transforms:
        if isinstance(transform, VoxelGrid):
            return True
    if dataset.train_dataset:
        for transform in dataset.train_transform.transforms:
            if isinstance(transform, VoxelGrid):
                return True
    elif dataset.val_dataset:
        for transform in dataset.val_transform.transforms:
            if isinstance(transform, VoxelGrid):
                return True
    elif dataset.test_dataset:
        for transform in dataset.inference_transform.transforms:
            if isinstance(transform, VoxelGrid):
                return True
    return False
