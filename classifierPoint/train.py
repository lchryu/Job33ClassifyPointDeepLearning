from classifierPoint.run import *

# Đường dẫn đến file cấu hình
config_path = r"D:\LCH\Thi\env\python38\Lib\site-packages\classifierPoint\train_config.json"

# Thực hiện train
train(config_path, "segmentation")
