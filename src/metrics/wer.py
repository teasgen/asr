from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class WERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = self.text_encoder.get_prediction(log_probs, log_probs_length)
        for pred_texts, target_text in zip(predictions, text):
            target_text = self.text_encoder.normalize_text(target_text)
            best_text = pred_texts[0]
            wers.append(calc_wer(target_text, best_text))
        return sum(wers) / len(wers)
