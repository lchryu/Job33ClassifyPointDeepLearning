import pynvml
import yaml
from pathlib import Path
import pkgutil
import logging

log = logging.getLogger(__name__)
MB = 1024 * 1024
ENVCONFIGPATH = "config/env.yaml"


def gpu_check():
    data = pkgutil.get_data("classifierPoint", ENVCONFIGPATH)
    env_config = yaml.safe_load(data)
    gpu_config = env_config["GPU"]
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    if deviceCount < 1:
        raise Exception(f"""Hasn't GPU, must use cuda""")
    if pynvml.nvmlSystemGetDriverVersion().decode("utf-8") < str(gpu_config["DRIVERVERSION"]):
        raise Exception(f"""Driver version don't match,the minimum version is {gpu_config["DRIVERVERSION"]}""")
    # for i in range(deviceCount):
    # 只检查第一个gpu
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    log.info(f"""GPU : {pynvml.nvmlDeviceGetName(handle).decode("utf-8")}""")
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_memory = int(info.total / MB)
    log.info(f"""Memory Total:{total_memory} MB """)
    if total_memory < int(gpu_config["MEMORY"]) * 1024:
        raise Exception(f"""Momory:{total_memory} is less than the minimum require {gpu_config["MEMORY"]*1024}""")
    log.info(f"""Memory Free: {info.free / MB} MB """)
    pynvml.nvmlShutdown()


if __name__ == "__main__":
    gpu_check()
