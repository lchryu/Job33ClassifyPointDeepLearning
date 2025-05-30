from omegaconf import OmegaConf
from typing import Dict, List
from omegaconf import DictConfig
from classifierPoint.utils.config_unpack import header_from_model, from_model
import yaml
import pkgutil
from classifierPoint.utils.util import init_classmapping


def _getmodelsconfig(json: Dict, loss: Dict) -> Dict:
    model_name = json["model"]["class"]
    model_path = f"config/model/{model_name}.yaml"
    try:
        data = pkgutil.get_data("classifierPoint", model_path)
        modelsconfig = yaml.safe_load(data)
    except:
        return None
    for key, value in json["model"]["params"].items():
        if key in modelsconfig[model_name].keys():
            modelsconfig[model_name][key] = value
        elif "define_constants" in modelsconfig[model_name].keys():
            if key in modelsconfig[model_name]["define_constants"].keys():
                modelsconfig[model_name]["define_constants"][key] = value
    modelsconfig[model_name]["loss"] = loss
    return modelsconfig


def check_feature(transforms):#检查用户是否加入特征
    flag = 0
    for transform in transforms:
        if "feature" in transform["class"].lower():
            flag += 1
    return True if flag > 0 else False


def _getdataconfig(
    json: Dict,
    task: str,
    classmapping_config: Dict = {},
    reclassmapping_config: Dict = {},
    train_path: str = "",
    val_path: str = "",
) -> Dict:
    mode_list = ["train_transform", "pre_transform", "split_transform"]
    dataconfig = {
        "class": "gvai.LidarClassifyDataset",
        "dataset_name": "lidar360",
        "process_workers": 0,
        "task": task,
        "classmapping": classmapping_config,
        "reversal_classmapping": reclassmapping_config,
        "train_path": train_path,
        "val_path": val_path,
    }
    flag = 0
    for mode in mode_list:
        if not mode in json["transform"].keys():
            json["transform"][mode] = []
        transform = json["transform"][mode]
        # list[transforms]
        if not transform:
            dataconfig[mode] = []
            continue
        if isinstance(transform, dict):
            transform = [transform]
        flag += check_feature(transform)
        dataconfig[mode] = transform
    if not flag:
        dataconfig["train_transform"].append({"class": "OnesFeature"})
        dataconfig["helper_transform"] = [{"class": "OnesFeature"}]
    dataconfig["train_transform"].append({"class": "AddAllFeat"})
    dataconfig["val_transform"] = dataconfig["train_transform"]
    return dataconfig


def _gettrainingconfig(json: Dict, checkpoint_file, resume) -> Dict:
    training_config = {
        "checkpoint_file": checkpoint_file,
        "cuda": 0,
        "shuffle": True,
        "num_workers": 0,
        "resume": resume,
        **json,
    }
    return training_config


def getlogconfig():
    log_path = "config" + "/log.yaml"
    data = pkgutil.get_data("classifierPoint", log_path)
    log_config = yaml.safe_load(data)
    return log_config


def _getevalconfig(json: Dict) -> Dict:
    evalconfig = {"num_workers": 0, "cuda": 0, "enable_cudnn": True, **json}
    #evalconfig = {**json}
    return evalconfig


def trainconfig(json: Dict, task: str) -> DictConfig:
    classmapping_config, reversal_classmapping = init_classmapping(json["class_remap"])
    if json["checkpoint_dir"]:
        resume = True
        checkpoint_file = json["checkpoint_dir"]
        checkpoint = from_model(checkpoint_file)
        model_classnum = checkpoint["dataset_properties"]["num_classes"]
        
        if classmapping_config:
            data_classes = max(classmapping_config.values()) + 1
            if data_classes > model_classnum:
                raise Exception(f"data classes:{data_classes} is more than model classes {model_classnum}")
    else:
        
        checkpoint_file = json["model"]["model"]["class"] + ".pt"
        resume = False 
    
    origin_json = json.copy()
    modelsconfig = _getmodelsconfig(json["model"], json["training"]["loss"])
    dataconfig = _getdataconfig(
        json,
        task=task,
        classmapping_config=classmapping_config,
        reclassmapping_config=reversal_classmapping,
        train_path=json["train_path"],
        val_path=json["val_path"],
    )

    training_config = _gettrainingconfig(json["training"], checkpoint_file, resume=resume)
    config = OmegaConf.create(
        {
            "models": {**modelsconfig},
            "data": {**dataconfig},
            "training": {**training_config},
            "save_path": json["save_path"],
            "origin_params": {**origin_json},
            "task_id": json["task_id"],
            "usegpu": json["usegpu"]
        }
    )
    return config


def evalconfig(json: Dict, task: str = "segmentation") -> DictConfig:
    if not json["checkpoint_dir"]:
        raise Exception("don't have model file,please check")
    checkpoint_file = json["checkpoint_dir"]
    header = header_from_model(checkpoint_file)
    pre_transform = header["pre_transform"]
    if "reversal_classmapping" not in json:
        reversal_classmapping = header["reversal_classmapping"]
    else:
        reversal_classmapping = json["reversal_classmapping"]
    evalconfig = _getevalconfig(json["eval"])
    evalconfig["reversal_classmapping"] = reversal_classmapping
    evalconfig["checkpoint_file"] = checkpoint_file
    evalconfig["test_path"] = json["test_path"]
    evalconfig["pre_transform"] = pre_transform
    evalconfig["usegpu"] = json["usegpu"]
    if "helper_transform" in header.keys():
        evalconfig["pre_transform"] += header["helper_transform"]
    evalconfig["test_transform"] = [{"class": "AddAllFeat"}]
    config = OmegaConf.create({**evalconfig, "save_path": json["save_path"], "mode": "eval"})
    return config
