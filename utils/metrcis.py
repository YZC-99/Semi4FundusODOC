import math
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
from torch import Tensor
from torchmetrics.functional.classification.confusion_matrix import _confusion_matrix_update

class IoU(torchmetrics.JaccardIndex):
    """ Wrapper because native IoU does not support ignore_index.
    https://github.com/PyTorchLightning/metrics/issues/304
    """

    def __init__(self, over_present_classes: bool = False, **kwargs):
        self.over_present_classes = over_present_classes
        super().__init__(**kwargs)

    def update(self, preds: Tensor, target: Tensor) -> None:  # type: ignore
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        target = target.view(-1)
        N = len(target)
        valid_mask = target != self.ignore_index
        target = target[valid_mask]
        if len(preds.shape) == 4:
            C = preds.shape[1]
            preds = preds.permute(0, 2, 3, 1).view(N, C)
            preds = preds[valid_mask, :]
        elif len(preds.shape) == 3:
            preds = preds.view(N)
            preds = preds[valid_mask]
        confmat = _confusion_matrix_update(
            preds, target, self.num_classes, self.threshold, self.multilabel)
        self.confmat += confmat

    def compute(self) -> Tensor:
        """Computes intersection over union (IoU)"""
        return self._jaccard_from_confmat(
            self.confmat,
            self.num_classes,
            self.average,
            None,
            self.absent_score,
            self.over_present_classes,
        )

    def _jaccard_from_confmat(
        self,
        confmat: Tensor,
        num_classes: int,
        average: Optional[str] = "macro",
        ignore_index: Optional[int] = None,
        absent_score: float = 0.0,
        over_present_classes: bool = False,
    ) -> Tensor:
        """Computes the intersection over union from confusion matrix.
        Args:
            confmat: Confusion matrix without normalization
            num_classes: Number of classes for a given prediction and target tensor
            average:
                Defines the reduction that is applied. Should be one of the following:
                - ``'macro'`` [default]: Calculate the metric for each class separately, and average the
                metrics across classes (with equal weights for each class).
                - ``'micro'``: Calculate the metric globally, across all samples and classes.
                - ``'weighted'``: Calculate the metric for each class separately, and average the
                metrics across classes, weighting each class by its support (``tp + fn``).
                - ``'none'`` or ``None``: Calculate the metric for each class separately, and return
                the metric for every class. Note that if a given class doesn't occur in the
                `preds` or `target`, the value for the class will be ``nan``.
            ignore_index: optional int specifying a target class to ignore. If given, this class index does not contribute
                to the returned score, regardless of reduction method.
            absent_score: score to use for an individual class, if no instances of the class index were present in `pred`
                AND no instances of the class index were present in `target`.
        """
        allowed_average = ["macro", "weighted", "none", None]
        if average not in allowed_average:
            raise ValueError(
                f"The `average` has to be one of {allowed_average}, got {average}.")

        # Remove the ignored class index from the scores.
        if ignore_index is not None and 0 <= ignore_index < num_classes:
            confmat[ignore_index] = 0.0

        if average == "none" or average is None:
            intersection = torch.diag(confmat)
            union = confmat.sum(0) + confmat.sum(1) - intersection

            present_classes = confmat.sum(dim=1) != 0

            # If this class is absent in both target AND pred (union == 0), then use the absent_score for this class.
            scores = intersection.float() / union.float()
            scores[union == 0] = absent_score

            if ignore_index is not None and 0 <= ignore_index < num_classes:
                scores = torch.cat(
                    [
                        scores[:ignore_index],
                        scores[ignore_index + 1:],
                    ]
                )
                present_classes = torch.cat(
                    [
                        present_classes[:ignore_index],
                        present_classes[ignore_index + 1:],
                    ]
                )

            if over_present_classes:
                scores = scores[present_classes]

            return scores

        if average == "macro":
            scores = self._jaccard_from_confmat(
                confmat, num_classes, average="none", ignore_index=ignore_index,
                absent_score=absent_score, over_present_classes=over_present_classes
            )
            return torch.mean(scores)

        if average == "micro":
            raise NotImplementedError()

        weights = torch.sum(confmat, dim=1).float() / \
            torch.sum(confmat).float()
        scores = self._jaccard_from_confmat(
            confmat, num_classes, average="none", ignore_index=ignore_index,
            absent_score=absent_score, over_present_classes=over_present_classes
        )
        return torch.sum(weights * scores)