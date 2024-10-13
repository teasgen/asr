import torch
from torch import nn
import torch.nn.functional as F


class DeepSpeech2(nn.Module):
    """
    Based on https://nvidia.github.io/OpenSeq2Seq/html/speech-recognition/deepspeech2.html
    """
    def __init__(self, n_feats, n_tokens, channels=32, gru_layers=5, fc_hidden=512):
        super().__init__()
        self.extractor = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=(11, 41), stride=(2, 2), padding=(5, 20)),
            nn.BatchNorm2d(channels),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            nn.BatchNorm2d(channels),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=(11, 21), stride=(1, 2), padding=(5, 10)),
            nn.BatchNorm2d(channels),
            nn.Hardtanh(min_val=0, max_val=20, inplace=True),
        )

        self.gru = nn.GRU(
            input_size=n_feats * channels // (2 ** 3),
            hidden_size=fc_hidden,
            num_layers=gru_layers,
            bidirectional=True,
            batch_first=True,
        )

        self.projection = nn.Linear(fc_hidden * 2, n_tokens)

    def forward(self, spectrogram, spectrogram_length, **batch):
        log_probs_length = self.transform_input_lengths(spectrogram_length)

        spectrogram = spectrogram.transpose(1, 2).unsqueeze(1)
        feats = self.extractor(spectrogram)

        feats = torch.transpose(feats, 1, 2).reshape((feats.size(0), feats.size(2), -1))

        feats = self.gru(feats)[0]
        log_probs = F.log_softmax(self.projection(feats), dim=-1)

        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        return (input_lengths + 1) // 2