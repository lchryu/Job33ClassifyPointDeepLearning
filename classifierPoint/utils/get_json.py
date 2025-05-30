import json
from typing import Dict
import pathlib as P
import pkgutil
from typing import List
import os
import torch
from classifierPoint.utils.util import classmapping_compress

JOSNBASEPATH = "json"
BESTPARCTICEPATH = "json/default"


class JsonPathFinder:
    def __init__(self, json: Dict, mode="key"):
        self.data = json
        self.mode = mode

    def iter_node(self, rows, road_step, target):
        if isinstance(rows, dict):
            key_value_iter = (x for x in rows.items())
        elif isinstance(rows, list):
            key_value_iter = (x for x in enumerate(rows))
        else:
            return
        for key, value in key_value_iter:
            current_path = road_step.copy()
            current_path.append(key)
            if self.mode == "key":
                check = key
            else:
                check = value
            if check == target:
                yield current_path
            if isinstance(value, (dict, list)):
                yield from self.iter_node(value, current_path, target)

    def find_one(self, target: str) -> list:
        path_iter = self.iter_node(self.data, [], target)
        for path in path_iter:
            return path
        return []

    def find_all(self, target) -> List[list]:
        path_iter = self.iter_node(self.data, [], target)
        return list(path_iter)


def parsingGenerator(data, pre=[]):
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                for subData in parsingGenerator(v, pre + [k]):
                    yield subData
            else:
                yield (pre + [k], v)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                for subData in parsingGenerator(item, pre):
                    yield subData
            else:
                yield (pre, item)


def get_config(class_name: str) -> Dict:
    """
    get components config list from json file return json Dict
    Parameters
    -----------
    class_name: str  eval,model,spilt,training,transform
    components name
    """

    def _url_hook(dict):
        if "url" in dict:
            url_path = P.Path(JOSNBASEPATH) / dict["url"]
            data = pkgutil.get_data("classifierPoint", str(url_path))
            dict = json.loads(data)
        return dict

    json_path = (P.Path(JOSNBASEPATH) / class_name).with_suffix(".json")
    try:
        data = pkgutil.get_data("classifierPoint", str(json_path))
        json_data = json.loads(data, object_hook=_url_hook)
    except:
        return None

    return json_data


def get_best_practice(model_name: str) -> Dict:
    """
    get model best practice config from json file return json Dict
    Parameters
    -----------
    model_name: str
    model name
    """
    model_dict = get_config("model")
    training_dict = get_config("training")
    transform_fun_dict = get_config("train_transform")
    classinfo_dict = get_config("classinfo")
    data = {}
    data["model"] = model_dict
    data["training"] = training_dict
    data["transform"] = transform_fun_dict
    data["classinfo"] = classinfo_dict
    json_path = (P.Path(BESTPARCTICEPATH) / model_name).with_suffix(".json")
    try:
        json_data = pkgutil.get_data("classifierPoint", str(json_path))
        json_data = json.loads(json_data)
    except:
        return data

    finder = JsonPathFinder(data, mode="value")
    class_name = ""
    for key, value in parsingGenerator(json_data):
        if key[-1] == "class":
            class_name = value
            path_list = finder.find_one(key[-2])
            path_list[-1] = "default"
            if len(path_list) == 4:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
        else:
            path_lists = finder.find_all(key[-1])
            if len(path_lists) > 1:
                path_lists = [path_list for path_list in path_lists if class_name in path_list]
            elif not path_lists:
                continue
            path_list = path_lists[0]
            path_list[-1] = "default"
            if len(path_list) == 4:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
            elif len(path_list) == 6:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]] = value
            elif len(path_list) == 7:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]][
                    path_list[6]
                ] = value
    return data


def default_param_merge(default_param: Dict, transform_mode=False) -> Dict:
    data = {}
    if not transform_mode:
        model_dict = get_config("model")
        training_dict = get_config("training")
        classinfo_dict = get_config("classinfo")
        data["model"] = model_dict
        data["training"] = training_dict
        data["classinfo"] = classinfo_dict
    train_transform_fun_dict = get_config("train_transform")
    pre_transform_fun_dict = get_config("pre_transform")
    split_transform_fun_dict = get_config("split_transform")
    data["transform"] = train_transform_fun_dict
    data["pre_transform"] = pre_transform_fun_dict
    data["split_transform"] = split_transform_fun_dict

    finder = JsonPathFinder(data, mode="value")
    class_name = ""
    for key, value in parsingGenerator(default_param):
        if key[-1] == "class":
            class_name = value
            path_list = finder.find_one(key[-2])
            if not path_list:
                continue
            path_list[-1] = "default"
            if len(path_list) == 4:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
        else:
            path_lists = finder.find_all(key[-1])
            if len(path_lists) > 1:
                path_lists = [path_list for path_list in path_lists if class_name in path_list]
            if not path_lists:
                continue
            path_list = path_lists[0]
            path_list[-1] = "default"
            if len(path_list) == 4:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
            elif len(path_list) == 6:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]] = value
            elif len(path_list) == 7:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]][
                    path_list[6]
                ] = value
            finder = JsonPathFinder(data, mode="value")
    return data


def _header_from_data(package_path: str) -> dict:
    package_path = os.path.join(package_path, "processed_data", "conf", "pre_transform.pt")
    if not os.path.exists(package_path):
        return None
    header = torch.load(package_path)
    return header


def get_dataprepare(data_path: str) -> Dict:
    """
    get dataprepare config  from data dir conf
    Parameters
    -----------
    data_path: str
        data_path
    """
    header = _header_from_data(data_path)
    classmapping = header["classmapping"]
    if not classmapping:
        classmapping = header["classes"]
    classmapping = classmapping_compress(classmapping)
    header.pop("classmapping")
    header.pop("classes")
    pre_transform_fun_dict = get_config("pre_transform")
    split_transform_fun_dict = get_config("split_transform")
    data = {}
    data["pre_transform"] = pre_transform_fun_dict
    data["split_transform"] = split_transform_fun_dict
    if not header:
        return data

    finder = JsonPathFinder(data, mode="value")
    class_name = ""
    for key, value in parsingGenerator(header):
        if key[-1] == "class":
            class_name = value
            path_list = finder.find_one(key[-2])
            path_list[-1] = "default"
            if len(path_list) == 4:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                if value not in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]]:
                    if "multi" in data[path_list[0]][path_list[1]][path_list[2]][path_list[3]]["type"].lower():
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] += value + ","
                    else:
                        data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
        else:
            path_lists = finder.find_all(key[-1])
            if len(path_lists) > 1:
                path_lists = [path_list for path_list in path_lists if class_name in path_list]
            elif not path_lists:
                continue
            path_list = path_lists[0]
            path_list[-1] = "default"
            if len(path_list) == 4:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
            elif len(path_list) == 5:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
            elif len(path_list) == 6:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]] = value
            elif len(path_list) == 7:
                data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]][
                    path_list[6]
                ] = value

    data["classmapping"] = classmapping
    return data


if __name__ == "__main__":
    print(get_best_practice("KPConv"))
