import torch
import os
from typing import Dict

# import re
# import logging
from classifierPoint.utils.get_json import default_param_merge
from classifierPoint.utils.func_rename import func_rename


def config_parse(package_path: str) -> Dict:
    try:
        checkpoint = torch.load(package_path)
        config = checkpoint["run_config"]["origin_params"]
        config = func_rename(config, True)
        config = default_param_merge(config)
    except:
        return None
    return config


def config_parse_origin(package_path: str) -> Dict:
    try:
        checkpoint = torch.load(package_path)
        config = checkpoint["run_config"]["origin_params"]
    except:
        return None
    return config


# def un_repr(str_json: str) -> Dict:
#     func_match = re.compile("Compose\(\[(.*)\]\)", re.S)
#     func_str = func_match.findall(str_json)[0]
#     tmp = re.sub("\n", "", func_str)
def from_model(checkpoint_file: str):
    if not os.path.exists(checkpoint_file):
        raise Exception(f"{checkpoint_file} is not exists")
    checkpoint = torch.load(checkpoint_file)
    return checkpoint


def header_from_model(checkpoint_file: str):
    if not os.path.exists(checkpoint_file):
        raise Exception(f"{checkpoint_file} is not exists")
    checkpoint = torch.load(checkpoint_file)
    header = checkpoint["run_config"]["data"]
    return header


def header_from_data(package_path: str, external_mode=True):
    package_path = os.path.join(package_path, "processed_data", "conf", "pre_transform.pt")
    if not os.path.exists(package_path):
        raise Exception(f"{package_path} is not exists")
    header = torch.load(package_path)
    if external_mode:
        header = func_rename(header, True)
        header = default_param_merge(header, True)
    return header


if __name__ == "__main__":
    pack_path = "D:\workspace\lidar360-deeplearning\PVCNN.pt"
    config = header_from_model(pack_path)
