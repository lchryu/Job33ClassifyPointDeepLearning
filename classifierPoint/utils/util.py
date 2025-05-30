import os
import logging
from typing import Dict, List, Union

log = logging.getLogger("metrics")


def check_status():
    if os.path.exists(".stop"):
        os.remove(".stop")
        return True
    return False


def processbar(stage, current, total):
    log.info(f"{stage} Processing {current+1}/{total}")


def init_classmapping(classmapping: Dict):
    if not classmapping:
        return {}, {}
    else:
        reversal_classmapping = {}
        origin = set(classmapping.values()) 
        origin.add(0)  
        origin = sorted(origin)
        target = list(range(-1, len(origin) - 1)) 
        for value in target[1:]: 
            reversal_classmapping[value] = int(
                list(classmapping.keys())[list(classmapping.values()).index(value + 1)]
            )
        res_classmapping = classmapping.copy()
        for i in range(len(origin)):
            for key, value in classmapping.items():
                if value == origin[i]:
                    res_classmapping.update({key: target[i]})
        return res_classmapping, reversal_classmapping


def classmapping_compress(classmapping: Union[Dict, List]):
    ret = {}
    if isinstance(classmapping, List):
        for source in classmapping:
            ret.update({str(source): source})
    elif isinstance(classmapping, Dict):
        source_list = list(classmapping.keys())
        target_list = list(classmapping.values())
        target_set = set(classmapping.values())

        for target in target_set:
            source = [
                source for source, tmp in zip(source_list, target_list) if tmp == target
            ]
            ret.update({",".join(source): target + 1})
    return ret
