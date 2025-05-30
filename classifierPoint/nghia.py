from classifierPoint.run import *


# config_ = '{"task_id": "9138dac0-f389-4782-be58-bfb1f9674611","train_path": "F:\\NGHIA\\8.0.2.0\\nghia\\train","val_path": "F:\\NGHIA\\8.0.2.0\\nghia\\train","task_description": "deeplearning\\ test","class_remap": {},"transform": {"Split Function": [{"class": "Tile by Point Number", "params": {"capacity": 800000}}],"pre_transform": [{"class": "OnesFeature"}, {"class": "Voxel Downsample", "params": {"size": 0.1, "mean": true}}],"train_transform": [{"class": "RandomTranslation", "params": {"delta_min": 0, "delta_max": 0.3}}, {"class": "Crop cubically", "params": {"c": 10, "rot_x": 2, "rot_y": 2, "rot_z": 180}}]},"model": {"model": {"class": "DRINet", "params": {}}},"training": {"batch size": 1,"weight name": "latest","epochs": 10,"lr": 0.001,"optimizer": {"class": "AdamW", "params": {"weight decay": 0.0001}},"lr_scheduler": {"class": "Cosine Annealing WarmRestarts lr scheduler", "params": {"T_0": 20, "T_mult": 2}},"loss": {"class": "LovaszSoftmaxLoss", "params": {}}},"checkpoint_dir": "","save_path": "F:\\NGHIA\\8.0.2.0\\nghia\\checkpoints"}'
# config_ = """
# {
#     "task_id": "9138dac0-f389-4782-be58-bfb1f9674611",
#     "train_path": "F:\\NGHIA\\8.0.2.0\\nghia\\train",
#     "val_path": "F:\\NGHIA\\8.0.2.0\\nghia\\train",
#     "task_description": "test",
#     "class_remap": {},
#     "transform": {
#         "Split Function": [{"class": "Tile by Point Number", "params": {"capacity": 800000}}],
#         "pre_transform": [{"class": "OnesFeature"}, {"class": "Voxel Downsample", "params": {"size": 0.1, "mean": true}}],
#         "train_transform": [{"class": "RandomTranslation", "params": {"delta_min": 0, "delta_max": 0.3}}, {"class": "Crop cubically", "params": {"c": 10, "rot_x": 2, "rot_y": 2, "rot_z": 180}}]
#     },
#     "model": {"model": {"class": "DRINet", "params": {}}},
#     "training": {
#         "batch size": 1,
#         "weight name": "latest",
#         "epochs": 10,
#         "lr": 0.001,
#         "optimizer": {"class": "AdamW", "params": {"weight decay": 0.0001}},
#         "lr_scheduler": {"class": "Cosine Annealing WarmRestarts lr scheduler", "params": {"T_0": 20, "T_mult": 2}},
#         "loss": {"class": "LovaszSoftmaxLoss", "params": {}}
#     },
#     "checkpoint_dir": "",
#     "save_path": "F:\\NGHIA\\8.0.2.0\\nghia\\checkpoints"
# }
# """
train("G:\\THI\\env\\python38\\Lib\\site-packages\\classifierPoint\\nghia_json.json", "segmentation", "test")

