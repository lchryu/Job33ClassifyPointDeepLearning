import copy
import torch
import time
import logging
import torch
# Import building function for model and dataset
from classifierPoint.datasets.dataset_factory import instantiate_dataset
from classifierPoint.models.model_factory import instantiate_model

# Import BaseModel / BaseDataset for type checking
from classifierPoint.models.base_model import BaseModel
from classifierPoint.datasets.base_dataset import BaseDataset

# Import from metrics
from classifierPoint.metrics.base_tracker import BaseTracker
from classifierPoint.metrics.colored_tqdm import Coloredtqdm as Ctq
from classifierPoint.metrics.model_checkpoint import ModelCheckpoint

# Utils import
from classifierPoint.utils.util import check_status, processbar

# PyTorch Profiler import
import torch.profiler
import torch.autograd.profiler
from contextlib import nullcontext
from classifierPoint.utils.status_code import STATUS

log = logging.getLogger(__name__)


class Trainer:
    """
    TorchPoints3d Trainer handles the logic between
        - BaseModel,
        - Dataset and its Tracker
        - A custom ModelCheckpoint
    It supports MC dropout - multiple voting_runs for val / test datasets
    """

    def __init__(self, cfg):
        self._cfg = cfg
        self._initialize_trainer()

    def _initialize_trainer(self):

        if not self.has_training:
            self._cfg.training = self._cfg
        if not getattr(self._cfg, "debugging", None):
            setattr(self._cfg, "debugging", None)

        # for eval mode dataset check
        mode = getattr(self._cfg.training, "mode", "train")
        resume = getattr(self._cfg.training, "resume", False)

        if mode == "eval":
            resume = False

        # Enable CUDNN BACKEND
        torch.backends.cudnn.enabled = self.enable_cudnn
        # Get device
        if self._cfg.usegpu and self._cfg.training.cuda > -1 and torch.cuda.is_available():
            device = "cuda"
            torch.cuda.set_device(self._cfg.training.cuda)
        else:
            device = "cpu"
            log.warning("cloud not init cuda, please check your GPU info,Note we only support Nvidia  GPU")

        self._device = torch.device(device)
        # self._device = torch.device('cpu')
        log.info("DEVICE : {}".format(self._device))

        # Profiling
        if self.profiling:
            # Set the num_workers as torch.utils.bottleneck doesn't work well with it
            self._cfg.training.num_workers = 0
        if not getattr(self._cfg.training, "weight_name", None):
            setattr(self._cfg.training, "weight_name", "latest")
        # Checkpoint
        self._checkpoint: ModelCheckpoint = ModelCheckpoint(
            self._cfg.training.checkpoint_file,
            self._cfg.training.weight_name,
            run_config=self._cfg,
            resume=resume,
            mode=mode,
        )

        # Create model and datasets
        if not self._checkpoint.is_empty:
            if mode == "eval":
                setattr(self._checkpoint.data_config, "test_path", self._cfg.test_path)
                setattr(self._checkpoint.data_config, "reversal_classmapping", self._cfg.reversal_classmapping)
                # pre_transform可能包含helper transform
                setattr(self._checkpoint.data_config, "pre_transform", self._cfg.pre_transform)
                setattr(self._checkpoint.data_config, "test_transform", self._cfg.test_transform)
                setattr(self._checkpoint.data_config, "num_features",self._checkpoint.dataset_properties["feature_dimension"],)
                [
                    delattr(self._checkpoint.data_config, key)
                    for key in ["train_path", "val_path", "train_transform", "val_transform"]
                    if hasattr(self._checkpoint.data_config, key)
                ]
                self._dataset: BaseDataset = instantiate_dataset(self._checkpoint.data_config)

            else:
                self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)

            log.info("Dataset generate success")
            self._model: BaseModel = self._checkpoint.create_model(
                self._dataset, weight_name=self._cfg.training.weight_name
            )
            
            # for resume train amp 
            if mode == "train":
                self._model.instantiate_optimizers(self._cfg, "cuda" in device)
                self._model._grad_scale = torch.cuda.amp.GradScaler(enabled=self._model.is_mixed_precision())
            log.info("Model generate success")

        else:
            self._dataset: BaseDataset = instantiate_dataset(self._cfg.data)
            self._checkpoint._checkpoint.run_config["data"][
                "reversal_classmapping"
            ] = self._dataset.reversal_classmapping
            log.info("Dataset generate success")
            self._model: BaseModel = instantiate_model(copy.deepcopy(self._cfg), self._dataset)
            self._model.instantiate_optimizers(self._cfg, "cuda" in device)
            self._model.set_pretrained_weights()
            log.info("Model generate success")
            if not self._checkpoint.validate(self._dataset.used_properties):
                log.warning(
                    "The model will not be able to be used from pretrained weights without the corresponding dataset. Current properties are {}".format(
                        self._dataset.used_properties
                    )
                )
        self._checkpoint.dataset_properties = self._dataset.used_properties

        log.debug(self._model)

        self._model.log_optimizers()
        log.debug(
            "Model size = %i",
            sum(param.numel() for param in self._model.parameters() if param.requires_grad),
        )

        # Set dataloaders
        # for eval yaml
        shuffle = getattr(self._cfg.training, "shuffle", False)
        self._dataset.create_dataloaders(
            self._model,
            self._cfg.training.batch_size,
            shuffle,
            self._cfg.training.num_workers,
            self.precompute_multi_scale,
        )
        log.debug(self._dataset)

        # Verify attributes in dataset 将会在model端自动处理
        # if mode == "eval":
        #     [self._model.verify_data(test_dataset[0]) for test_dataset in self._dataset.test_dataset]
        # else:
        #     self._model.verify_data(self._dataset.train_dataset[0])

        # Choose selection stage
        selection_stage = getattr(self._cfg, "selection_stage", "")
        self._checkpoint.selection_stage = self._dataset.resolve_saving_stage(selection_stage)
        self._tracker: BaseTracker = self._dataset.get_tracker(self.tensorboard_log)

        # Run training / evaluation
        self._model = self._model.to(self._device)

    def train(self):
        return_code = STATUS.SUCCESS
        self._is_training = True
        
        for epoch in range(self._checkpoint.start_epoch, self._cfg.training.epochs):
            log.info("EPOCH %i / %i", epoch, self._cfg.training.epochs)

            return_code = self._train_epoch(epoch)
            if return_code != STATUS.SUCCESS:
                return return_code
            if self.profiling:
                return False

            if epoch % self.eval_frequency != 0:
                continue

            # if self._dataset.has_val_loader:
            #     return_code = self._val_epoch(epoch)
            # if self._dataset.has_test_loaders:
            #     return_code = self._test_epoch(epoch)

            if return_code != STATUS.SUCCESS:
                return return_code
            processbar("Train", epoch, self._cfg.training.epochs)
            f = open(".schedule", 'w', encoding='utf8')
            f.write(str(int(((epoch + 1) / self._cfg.training.epochs)*100)))
            f.close()

        # Thêm thông tin độ đo vào log file
        metrics = self._tracker.get_metrics()
        log.info("Training completed. Final metrics:")
        for metric_name, metric_value in metrics.items():
            log.info(f"{metric_name}: {metric_value:.4f}")
        
        return return_code

    def test(self):
        return_code = STATUS.SUCCESS
        self._is_training = False
        
        if self._dataset.has_test_loaders:
            return_code = self._test_epoch(0)  # Chỉ chạy test một lần với epoch = 0
            
        return return_code

    def eval(self):
        self._is_training = False
        epoch = self._checkpoint.start_epoch
        if self._dataset.has_test_loaders:
            return self._test_epoch(epoch)
        else:
            return STATUS.DATAERROR

    def _finalize_epoch(self, epoch):
        self._tracker.finalise(**self.tracker_options)
        if self._is_training:
            metrics = self._tracker.publish(epoch)
            self._checkpoint.save_best_models_under_current_metrics(self._model, metrics, self._tracker.metric_func)
            if self._tracker._stage == "train":
                log.info("Learning rate = %f" % self._model.learning_rate)

    def _train_epoch(self, epoch: int):

        self._model.train()
        self._tracker.reset("train")
        train_loader = self._dataset.train_dataloader
        with self.profiler_profile(epoch) as prof:
            iter_data_time = time.time()
            with Ctq(train_loader) as tq_train_loader:
                for i, data in enumerate(tq_train_loader):
                    if check_status():
                        return STATUS.PAUSE
                    if data.pos.shape[0] < 100 or int(torch.max(data.y)) == -1 :
                        continue
                    t_data = time.time() - iter_data_time
                    iter_start_time = time.time()
                    with self.profiler_record_function("train_step"):
                        self._model.set_input(data, self._device)
                        self._model.optimize_parameters(epoch, self._dataset.batch_size)

                    with self.profiler_record_function("track/log/visualize"):
                        if i % 10 == 0:
                            with torch.no_grad():
                                self._tracker.track(self._model, data=data, **self.tracker_options)

                        tq_train_loader.set_postfix(
                            **self._tracker.get_metrics(),
                            data_loading=float(t_data),
                            iteration=float(time.time() - iter_start_time),
                        )

                    iter_data_time = time.time()

                    if self.pytorch_profiler_log:
                        prof.step()

                    if self.profiling:
                        if i > self.num_batches:
                            return STATUS.FAILURE

        self._finalize_epoch(epoch)
        self._tracker.print_summary(epoch)
        return STATUS.SUCCESS

    def _val_epoch(self, epoch: int):
        stage_name = "val"
        voting_runs = self._cfg.get("voting_runs", 1)
        loader = self._dataset.val_dataloader

        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        self._tracker.reset(stage_name)
        if not self._dataset.has_labels(stage_name):
            log.warning("No forward will be run on dataset %s." % stage_name)
            return STATUS.DATAERROR

        with self.profiler_profile(epoch) as prof:
            for i in range(voting_runs):
                with Ctq(loader) as tq_loader:
                    for data in tq_loader:


                        if check_status():

                            return STATUS.PAUSE
                        with torch.no_grad():
                            with self.profiler_record_function("test_step"):
                                self._model.set_input(data, self._device)
                                with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                    self._model.forward()
                                    # debug
                                    # output = torch.max(self._model.output.cpu(), -1).indices.type(torch.int)
                                    # coords = self._model.input.C.cpu()[:, :3]
                                    # import numpy as np
                                    # np.savetxt("1.txt",np.c_[coords,output.unsqueeze(1)])

                            with self.profiler_record_function("track/log/visualize"):
                                self._tracker.track(self._model, data=data, **self.tracker_options)
                                tq_loader.set_postfix(**self._tracker.get_metrics())
                        if self.pytorch_profiler_log:
                            prof.step()

                        if self.profiling:
                            if i > self.num_batches:
                                return STATUS.FAILURE

        self._finalize_epoch(epoch)
        self._tracker.print_summary()
        return STATUS.SUCCESS
    def cmd_log(self, log_type, info):
        if log_type == "standard":
            output = f"standardlog: {info}"
            print(output)
        elif log_type == "warning":
            output = f"warninglog: {info}"
            print(output)
        elif log_type == "error":
            output = f"errorlog: {info}"
            print( output)
        elif log_type == "progress":
            output = f"progress:{info}"
            print(output)
    def _test_epoch(self, epoch: int):
        stage_name = "test"
        voting_runs = self._cfg.get("voting_runs", 1)
        loaders = self._dataset.test_dataloaders
        self._model.eval()
        if self.enable_dropout:
            self._model.enable_dropout_in_eval()

        # Sửa lại thành getLogger
        metrics_logger = logging.getLogger("metrics")

        self.cmd_log("standard", "Data processing is finished")
        self.cmd_log("progress", 11)

        # Log metrics
        metrics = self._tracker.get_metrics()
        metrics_logger.info("Test Results:")
        for key, value in metrics.items():
            metrics_logger.info(f"    {key}: {value}")
        metrics_logger.info("=" * 50)

        for loader in loaders:
            self._tracker.reset(stage_name)
            with self.profiler_profile(epoch) as prof:
                for i in range(voting_runs):
                    #with Ctq(loader) as tq_loader:
                        #for j, data in enumerate(tq_loader):
                        for j, data in enumerate(loader):
                            self.cmd_log("standard", f"Infer block {j} ...")
                            if check_status():
                                return STATUS.PAUSE
                            with torch.no_grad():
                                with self.profiler_record_function("test_step"):
                                    if not data:
                                        continue
                                    # for split data
                                    if hasattr(data, "split"): # data đã được split
                                        self.cmd_log("standard", f"Split data")
                                        for i in range(len(data.split)):
                                            output = [data_seg.pos.shape[0] for data_seg in data.split[i]]
                                            output = torch.zeros([sum(output)], dtype=torch.int)
                                            for data_seg in data.split[i]:
                                                data_seg.batch = torch.zeros([data_seg.pos.shape[0]], dtype=torch.long)
                                                self._model.set_input(data_seg, self._device)
                                                with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                                    self._model.forward()
                                                output[data_seg.splitidx.long()] = torch.max(
                                                    self._model.output.cpu(), -1
                                                ).indices.type(torch.int)
                                            self._dataset.test_dataset[0].write_res_split(data[i], output)
                                    else: # data bình thường
                                        self.cmd_log("standard", f"Normal data")
                                        self._model.set_input(data, self._device)
                                        with torch.cuda.amp.autocast(enabled=self._model.is_mixed_precision()):
                                            self._model.forward()
                                        self._dataset.test_dataset[0].write_res(
                                            data, torch.max(self._model.output.cpu(), -1)[1]
                                        )
                                #processbar("inference", j, len(tq_loader))
                            
                            self.cmd_log("progress", 10 + int(((j+1)/len(loader))*90)) #Finish infer image {img_name}
                            self.cmd_log("standard", f"Finish infer block {j}")
                            if self.pytorch_profiler_log:
                                prof.step()

                            if self.profiling:
                                if i > self.num_batches:
                                    return STATUS.FAILURE
            self._finalize_epoch(epoch)
        return STATUS.SUCCESS

    @property
    def profiling(self):
        return getattr(self._cfg.debugging, "profiling", False)

    @property
    def num_batches(self):
        return getattr(self._cfg.debugging, "num_batches", 50)

    @property
    def enable_cudnn(self):
        return getattr(self._cfg.training, "enable_cudnn", True)

    @property
    def enable_dropout(self):
        return getattr(self._cfg, "enable_dropout", False)

    @property
    def has_tensorboard(self):
        return getattr(self._cfg.training, "tensorboard", False)

    @property
    def has_training(self):
        return getattr(self._cfg, "training", None)

    @property
    def precompute_multi_scale(self):
        return self._model.conv_type == "PARTIAL_DENSE" and getattr(self._cfg.training, "precompute_multi_scale", False)

    @property
    def tensorboard_log(self):
        if self.has_tensorboard:
            return getattr(self._cfg.training.tensorboard, "log", False)
        else:
            return False

    @property
    def pytorch_profiler_log(self):
        if self.tensorboard_log:
            if getattr(self._cfg.training.tensorboard, "pytorch_profiler", False):
                return getattr(self._cfg.training.tensorboard.pytorch_profiler, "log", False)
        return False

    # pyTorch Profiler
    def profiler_profile(self, epoch):
        if self.pytorch_profiler_log and (
            getattr(self._cfg.training.tensorboard.pytorch_profiler, "nb_epoch", 3) == 0
            or epoch <= getattr(self._cfg.training.tensorboard.pytorch_profiler, "nb_epoch", 3)
        ):
            return torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
                if self._cfg.training.cuda > -1
                else [torch.profiler.ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(
                    skip_first=getattr(
                        self._cfg.training.tensorboard.pytorch_profiler,
                        "skip_first",
                        10,
                    ),
                    wait=getattr(self._cfg.training.tensorboard.pytorch_profiler, "wait", 5),
                    warmup=getattr(self._cfg.training.tensorboard.pytorch_profiler, "warmup", 3),
                    active=getattr(self._cfg.training.tensorboard.pytorch_profiler, "active", 5),
                    repeat=getattr(self._cfg.training.tensorboard.pytorch_profiler, "repeat", 0),
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self._tracker._tensorboard_dir),
                record_shapes=getattr(
                    self._cfg.training.tensorboard.pytorch_profiler,
                    "record_shapes",
                    True,
                ),
                profile_memory=getattr(
                    self._cfg.training.tensorboard.pytorch_profiler,
                    "profile_memory",
                    True,
                ),
                with_stack=getattr(self._cfg.training.tensorboard.pytorch_profiler, "with_stack", True),
                with_flops=getattr(self._cfg.training.tensorboard.pytorch_profiler, "with_flops", True),
            )
        else:
            return nullcontext(type("", (), {"step": lambda self: None})())

    def profiler_record_function(self, name: str):
        if self.pytorch_profiler_log:
            return torch.autograd.profiler.record_function(name)
        else:
            return nullcontext()

    @property
    def tracker_options(self):
        return self._cfg.get("tracker_options", {})

    @property
    def eval_frequency(self):
        return self._cfg.get("eval_frequency", 1)
