from typing import Dict, Any
import torch
import numpy as np

from classifierPoint.metrics.confusion_matrix import ConfusionMatrix
from classifierPoint.metrics.base_tracker import BaseTracker, meter_value
from classifierPoint.metrics.meters import APMeter
from classifierPoint.config.IGNORE_LABEL import IGNORE_LABEL
from classifierPoint.models import model_interface


class SegmentationTracker(BaseTracker):
    def __init__(
        self,
        dataset,
        stage="train",
        use_tensorboard: bool = False,
        ignore_label: int = IGNORE_LABEL,
    ):
        """This is a generic tracker for segmentation tasks.
        It uses a confusion matrix in the back-end to track results.
        Use the tracker to track an epoch.
        You can use the reset function before you start a new epoch

        Arguments:
            dataset  -- dataset to track (used for the number of classes)

        Keyword Arguments:
            stage {str} -- current stage. (train, validation, test, etc...) (default: {"train"})
        """
        super(SegmentationTracker, self).__init__(stage, use_tensorboard)
        self._num_classes = dataset.num_classes
        self._ignore_label = ignore_label
        self._dataset = dataset
        self.reset(stage)
        self._metric_func = {
            "miou": max,
            "macc": max,
            "acc": max,
            "loss": min,
            "map": max,
            "eval_accuracy": max,
            "eval_precision": max,
            "eval_recall": max,
            "eval_f1": max
        }  # Those map subsentences to their optimization functions

    def reset(self, stage="train"):
        super().reset(stage=stage)
        self._confusion_matrix = ConfusionMatrix(self._num_classes)
        self._acc = 0
        self._macc = 0
        self._miou = 0
        self._iou_per_class = {}

    @staticmethod
    def detach_tensor(tensor):
        if torch.torch.is_tensor(tensor):
            tensor = tensor.detach()
        return tensor

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.confusion_matrix

    def track(self, model: model_interface.TrackerInterface, **kwargs):
        """Add current model predictions (usually the result of a batch) to the tracking"""
        if not self._dataset.has_labels(self._stage):
            return

        super().track(model)

        outputs = model.get_output()
        targets = model.get_labels()

        pred = torch.max(outputs, 1)[1]
        correct = (pred == targets).sum().item()
        total = targets.size(0)
        
        self._eval_metrics = {
            "eval_accuracy": correct / total,
            "eval_precision": self._compute_precision(pred, targets),
            "eval_recall": self._compute_recall(pred, targets),
            "eval_f1": self._compute_f1(pred, targets)
        }    

        self._compute_metrics(outputs, targets)

    def _compute_metrics_per_class(self, pred, targets):
        """Tính toán TP, FP, FN cho mỗi class"""
        true_positive = torch.zeros(self._num_classes)
        false_positive = torch.zeros(self._num_classes)
        false_negative = torch.zeros(self._num_classes)
        
        for c in range(self._num_classes):
            pred_c = pred == c
            target_c = targets == c
            true_positive[c] = (pred_c & target_c).sum().item()
            false_positive[c] = (pred_c & ~target_c).sum().item()
            false_negative[c] = (~pred_c & target_c).sum().item()
            
        return true_positive, false_positive, false_negative
    
    def _compute_precision(self, pred, targets):
        """Tính precision trung bình cho tất cả các class"""
        true_positive, false_positive, _ = self._compute_metrics_per_class(pred, targets)
        precision = torch.zeros(self._num_classes)
        
        for c in range(self._num_classes):
            if true_positive[c] + false_positive[c] > 0:
                precision[c] = true_positive[c] / (true_positive[c] + false_positive[c])
                
        # Loại bỏ các class không xuất hiện (để tránh ảnh hưởng đến giá trị trung bình)
        valid_classes = precision > 0
        if valid_classes.sum() > 0:
            return precision[valid_classes].mean().item()
        return 0.0

    def _compute_recall(self, pred, targets):
        """Tính recall trung bình cho tất cả các class"""
        true_positive, _, false_negative = self._compute_metrics_per_class(pred, targets)
        recall = torch.zeros(self._num_classes)
        
        for c in range(self._num_classes):
            if true_positive[c] + false_negative[c] > 0:
                recall[c] = true_positive[c] / (true_positive[c] + false_negative[c])
                
        # Loại bỏ các class không xuất hiện
        valid_classes = recall > 0
        if valid_classes.sum() > 0:
            return recall[valid_classes].mean().item()
        return 0.0

    def _compute_f1(self, pred, targets):
        """Tính F1 score từ precision và recall"""
        precision = self._compute_precision(pred, targets)
        recall = self._compute_recall(pred, targets)
        
        if precision + recall > 0:
            return 2 * (precision * recall) / (precision + recall)
        return 0.0
    def _compute_metrics(self, outputs, labels):
        mask = labels != self._ignore_label
        outputs = outputs[mask]
        labels = labels[mask]

        outputs = self._convert(outputs)
        labels = self._convert(labels)

        if len(labels) == 0:
            return

        assert outputs.shape[0] == len(labels)
        self._confusion_matrix.count_predicted_batch(labels, np.argmax(outputs, 1))

        self._acc = 100 * self._confusion_matrix.get_overall_accuracy()
        self._macc = 100 * self._confusion_matrix.get_mean_class_accuracy()
        self._miou = 100 * self._confusion_matrix.get_average_intersection_union()
        self._iou_per_class = {
            i: "{:.2f}".format(100 * v)
            for i, v in enumerate(self._confusion_matrix.get_intersection_union_per_class()[0])
        }

    # def get_metrics(self, verbose=False) -> Dict[str, Any]:
    #     """Returns a dictionnary of all metrics and losses being tracked"""
    #     metrics = super().get_metrics(verbose)

    #     metrics["{}_acc".format(self._stage)] = self._acc
    #     metrics["{}_macc".format(self._stage)] = self._macc
    #     metrics["{}_miou".format(self._stage)] = self._miou

    #     if verbose:
    #         metrics["{}_iou_per_class".format(self._stage)] = self._iou_per_class
    #     return metrics
    def get_metrics(self, verbose=False) -> Dict[str, Any]:
        """Trả về tất cả các metrics"""
        metrics = super().get_metrics(verbose)
        if hasattr(self, '_eval_metrics') and self._stage in ["test", "val"]:
            metrics.update(self._eval_metrics)
        return metrics

    @property
    def metric_func(self):
        return self._metric_func
