import os
from classifierPoint.trainer import Trainer
from classifierPoint.utils.json2conf import trainconfig, evalconfig, getlogconfig
import json
from pathlib import Path
import gc
import torch
from classifierPoint.utils.status_code import STATUS
import logging
import logging.config
import traceback
import numpy as np
from classifierPoint.utils.env_check import gpu_check
from classifierPoint.utils.func_rename import func_rename

log = logging.getLogger(__name__)
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(3047)
def mock_hydra_init(save_path):
    log_config = getlogconfig()
    if save_path:
        Path(str(save_path)).mkdir(parents=True, exist_ok=True)
        os.chdir(save_path)
    logging.config.dictConfig(log_config)
    log.info("Task Start")


def init(config: str):
    config = json.loads(config)
    config = func_rename(config)
    mock_hydra_init(config["save_path"])
    gpu_check()
    return config
def init_test(config_path):
    with open(config_path, encoding="utf-8") as data:
        config = json.load(data)
    #config = json.loads(config)
    config = func_rename(config)
    mock_hydra_init(config["save_path"])
    gpu_check()
    return config

# def train(config: str, task: str = "segmentation", mode=""):
#     try:
#         if mode == "test":
#             config = init_test(config)
#         else:
#             config = init(config)
#         log.info("Config  Init")
#         config = trainconfig(config, task=task)
#     except Exception as ex:
#         log.error(f"""INITERROR,{ex.__class__.__name__} {ex}""")
#         return {"status": STATUS.INITERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
#     try:
#         trainer = Trainer(config)
#     except Exception as ex:
#         gc.collect()
#         torch.cuda.empty_cache()
#         log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
#         return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
#     try:
#         code = trainer.train()
#     except Exception as ex:
#         del trainer
#         gc.collect()
#         torch.cuda.empty_cache()
#         if str(ex).startswith("CUDA out of memory."):
#             log.error(f"""CUDAOUTOFMEMORY,{ex.__class__.__name__} {traceback.format_exc()}""")
#             return {"status": STATUS.MEMORYERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
#         log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
#         return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
#     del trainer
#     gc.collect()
#     torch.cuda.empty_cache()
#     log.info("SUCCESS")
#     return {"status": code.value, "msg": "success"}
def train(config: str, task: str = "segmentation"):
    try:
        config = init_test(config)
        log.info("Config  Init")
        config = trainconfig(config, task=task)
    except Exception as ex:
        log.error(f"""INITERROR,{ex.__class__.__name__} {ex}""")
        return {"status": STATUS.INITERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    
    try:
        trainer = Trainer(config)
    except Exception as ex:
        gc.collect()
        torch.cuda.empty_cache()
        log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
        return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    try:
        code = trainer.train()
    except Exception as ex:
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
        if str(ex).startswith("CUDA out of memory."):
            log.error(f"""CUDAOUTOFMEMORY,{ex.__class__.__name__} {traceback.format_exc()}""")
            return {"status": STATUS.MEMORYERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
        log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
        return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    del trainer
    gc.collect()
    torch.cuda.empty_cache()
    log.info("SUCCESS")
    return {"status": code.value, "msg": "success"}

def eval(config: str):
    try:
        config = init_test(config)
        log.info("Config  Init")
        config = evalconfig(config)
    except Exception as ex:
        log.error(f"""INITERROR,{ex.__class__.__name__} {ex}""")
        return {"status": STATUS.INITERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    try:
        trainer = Trainer(config)
    except Exception as ex:
        gc.collect()
        if config["usegpu"]:
            torch.cuda.empty_cache()
        log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
        return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    try:
        code = trainer.eval()

        metrics = trainer._tracker.get_metrics()
        log.info("Evaluation Result: ")
        for key, value in metrics.items():
            log.info(f"{key}: {value}")
        log.info("=" * 50)
        
    except Exception as ex:
        del trainer
        gc.collect()
        if config["usegpu"]:
            torch.cuda.empty_cache()
        if str(ex).startswith("CUDA out of memory."):
            log.error(f"""CUDAOUTOFMEMORY,{ex.__class__.__name__} {traceback.format_exc()}""")
            return {"status": STATUS.MEMORYERROR.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
        log.error(f"""RUNNINGFAILURE,{ex.__class__.__name__} {traceback.format_exc()}""")
        return {"status": STATUS.FAILURE.value, "msg": ex.__class__.__name__ + ":" + str(ex)}
    del trainer
    gc.collect()
    if config["usegpu"]:
        torch.cuda.empty_cache()
    log.info("SUCCESS")
    return {"status": code.value, "msg": "success"}
