from typing import Dict
from classifierPoint.utils.get_json import JsonPathFinder
import pkgutil
from pathlib import Path
import json


def func_rename(json_data: Dict, reversal=False):
    json_path = (Path("json") / "func_remap").with_suffix(".json")

    remap_data = pkgutil.get_data("classifierPoint", str(json_path))
    remap_data = json.loads(remap_data)
    finder = JsonPathFinder(json_data, mode="value")
    finder2 = JsonPathFinder(json_data, mode="key")
    for value, key in remap_data.items():
        if reversal:
            value, key = key, value
        path_lists = finder.find_all(key)
        if path_lists:
            for path_list in path_lists:
                if len(path_list) == 2:
                    json_data[path_list[0]][path_list[1]] = value
                if len(path_list) == 3:
                    json_data[path_list[0]][path_list[1]][path_list[2]] = value
                elif len(path_list) == 4:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]] = value
                elif len(path_list) == 5:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]] = value
                elif len(path_list) == 6:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][
                        path_list[5]
                    ] = value
                elif len(path_list) == 7:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]][
                        path_list[6]
                    ] = value
                elif len(path_list) == 8:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][path_list[4]][path_list[5]][
                        path_list[6][path_list[7]]
                    ] = value
        else:
            path_lists = finder2.find_all(key)
            for path_list in path_lists:
                if len(path_list) == 5:
                    json_data[path_list[0]][path_list[1]][path_list[2]][path_list[3]][value] = json_data[path_list[0]][
                        path_list[1]
                    ][path_list[3]].pop(key)
                if len(path_list) == 4:
                    json_data[path_list[0]][path_list[1]][path_list[2]][value] = json_data[path_list[0]][path_list[1]][
                        path_list[2]
                    ].pop(key)
                if len(path_list) == 3:
                    json_data[path_list[0]][path_list[1]][value] = json_data[path_list[0]][path_list[1]].pop(key)
                if len(path_list) == 2:
                    json_data[path_list[0]][value] = json_data[path_list[0]].pop(key)
            finder = JsonPathFinder(json_data, mode="value")
    return json_data
