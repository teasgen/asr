import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


class Conv2dSubsampling(nn.Module):
    def __init__(self, d_hidden=144) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=d_hidden, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=d_hidden, out_channels=d_hidden, kernel_size=3, stride=2
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.model(x.unsqueeze(1))  # bs, 1, T, d_in
        x = rearrange(x, "b d t f -> b t (d f)")
        return x


class FFNBlock(nn.Module):
    def __init__(self, dim=144, dropout=0) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.model(x)


class ConvBlock(nn.Module):
    def __init__(self, d_hidden=144, kernel_size=31, dropout=0) -> None:
        super().__init__()
        self.ln = nn.LayerNorm(d_hidden)
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden * 2, kernel_size=1),
            nn.GLU(dim=1),  # half channels dim
            nn.Conv1d(
                in_channels=d_hidden,
                out_channels=d_hidden,
                kernel_size=kernel_size,
                groups=d_hidden,
                padding="same",
            ),
            nn.BatchNorm1d(d_hidden),
            nn.SiLU(),
            nn.Conv1d(in_channels=d_hidden, out_channels=d_hidden, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.ln(x).transpose(1, 2)  # BS T D -> BS D T
        return self.model(x).transpose(1, 2)


class AbsolutePosEncoding(nn.Module):
    def __init__(self, d_hidden=144, max_len=10**4):
        super().__init__()
        self.d_hidden = d_hidden
        encodings = torch.zeros(max_len, d_hidden)
        pos = torch.arange(0, max_len, dtype=torch.float)
        inv_freq = 1 / (10000 ** (torch.arange(0.0, d_hidden, 2.0) / d_hidden))
        encodings[:, 0::2] = torch.sin(pos[:, None] * inv_freq)
        encodings[:, 1::2] = torch.cos(pos[:, None] * inv_freq)
        self.register_buffer("encodings", encodings)  # dont optimtize

    def forward(self, idx):
        return self.encodings[:idx, :]


class RMHA(nn.Module):
    """
    Relative MHA
    https://arxiv.org/pdf/1901.02860
    This class is based on https://github.com/chathasphere/pno-ai/blob/master/model/attention.py
    """

    def __init__(
        self, d_hidden=144, n_heads=4, dropout=0, pos_enc=AbsolutePosEncoding(144)
    ) -> None:
        super().__init__()
        # check dims for MHA
        assert d_hidden % n_heads == 0
        self.d_head = d_hidden // n_heads

        self.d_hidden = d_hidden
        self.n_heads = n_heads
        self.dropout = dropout
        self.pos_enc = pos_enc

        self.qkv = nn.Linear(d_hidden, 3 * d_hidden)
        self.pos = nn.Linear(d_hidden, d_hidden, bias=False)
        self.ffn = nn.Linear(d_hidden, d_hidden)
        self.attn_dropout = nn.Dropout(p=dropout)
        self.ffn_dropout = nn.Dropout(p=dropout)
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x):
        B, T, _ = x.shape

        x = self.ln(x)
        pos_emb = self.pos_enc(T)
        pos_emb = pos_emb.repeat(B, 1, 1)

        q, k, v = self.qkv(x).split(self.d_hidden, dim=2)
        q = q.view(B, T, self.n_heads, self.d_hidden // self.n_heads)
        k = rearrange(
            k.view(B, T, self.n_heads, self.d_hidden // self.n_heads),
            "b t h d -> b h d t",  # BS num_heads dim_head time
        )
        v = rearrange(
            v.view(B, T, self.n_heads, self.d_hidden // self.n_heads),
            "b t h d -> b h d t",
        )
        pos_emb = rearrange(
            self.pos(pos_emb).view(B, -1, self.n_heads, self.d_head),
            "b t h d -> b h d t",
        )

        qk = torch.matmul(q.transpose(1, 2), k)
        q_pos = torch.matmul(q.transpose(1, 2), pos_emb)
        q_pos = self._skew(q_pos)
        numenator = F.softmax((qk + q_pos) / math.sqrt(self.d_hidden), -1)

        output = torch.matmul(numenator, v.transpose(2, 3)).transpose(1, 2)
        output = output.contiguous().view(B, -1, self.d_hidden)

        output = self.ffn_dropout(self.ffn(output))
        return output

    def _skew(self, emb):
        B, N, seq_len1, seq_len2 = emb.shape
        zeros = emb.new_zeros(B, N, seq_len1, 1)
        pad_emb = torch.cat([zeros, emb], dim=-1)
        pad_emb = pad_emb.view(B, N, seq_len2 + 1, seq_len1)
        shifted_emb = pad_emb[:, :, 1:].view_as(emb)
        return shifted_emb


class ConformerBlock(nn.Module):
    def __init__(
        self,
        d_hidden=144,
        kernel_size=31,
        n_heads=4,
        dropout=0,
        pos_enc=AbsolutePosEncoding(144),
    ) -> None:
        super().__init__()
        self.ffn1 = FFNBlock(d_hidden, dropout)
        self.mha = RMHA(d_hidden, n_heads, dropout, pos_enc)
        self.conv = ConvBlock(d_hidden, kernel_size, dropout)
        self.ffn2 = FFNBlock(d_hidden, dropout)
        self.ln = nn.LayerNorm(d_hidden)

    def forward(self, x):
        residual = x
        x = self.ffn1(x) * 0.5 + residual
        residual = x
        x = self.mha(x) + residual
        residual = x
        x = self.conv(x) + residual
        residual = x
        x = self.ffn2(x) * 0.5 + residual
        return self.ln(x)


class ConformerEncoder(nn.Module):
    """
    Conformer
    """

    def __init__(
        self,
        d_in=128,
        d_hidden=144,
        kernel_size=31,
        n_layers=16,
        n_heads=4,
        dropout=0.0,
    ):
        super().__init__()
        self.conv_subsampling = Conv2dSubsampling(d_hidden)
        self.ffn_after_subsampling = nn.Linear(
            d_hidden * (((d_in - 1) // 2 - 1) // 2), d_hidden
        )
        self.dropout = nn.Dropout(p=dropout)
        self.conformer_blocks = nn.Sequential()
        pos_enc = AbsolutePosEncoding(d_hidden)
        for _ in range(n_layers):
            self.conformer_blocks.append(
                ConformerBlock(
                    d_hidden=d_hidden,
                    kernel_size=kernel_size,
                    n_heads=n_heads,
                    dropout=dropout,
                    pos_enc=pos_enc,
                )
            )

    def forward(self, spectrogram, spectrogram_length, **batch):
        """
        Model forward method.

        Args:
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        spectrogram = spectrogram.transpose(1, 2)
        x = self.conv_subsampling(spectrogram)
        x = self.ffn_after_subsampling(x)
        x = self.dropout(x)
        output = self.conformer_blocks(x)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"out": output, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        input_lengths = input_lengths >> 2
        input_lengths -= 1
        return input_lengths

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info


class LSTMDecoder(nn.Module):
    def __init__(self, d_encoder=144, d_decoder=320, n_layers=1, n_tokens=28) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=d_encoder,
            hidden_size=d_decoder,
            num_layers=n_layers,
            batch_first=True,
        )
        self.ffn = nn.Linear(d_decoder, n_tokens)

    def forward(self, x):
        x, _ = self.lstm(x)
        logits = self.ffn(x)
        return logits


class ConformerEncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, spectrogram, spectrogram_length, **batch):
        encoder_output = self.encoder(spectrogram, spectrogram_length, **batch)

        decoder_output = self.decoder(encoder_output["out"])
        log_probs = F.log_softmax(decoder_output, dim=-1)
        return {
            "log_probs": log_probs,
            "log_probs_length": encoder_output["log_probs_length"],
        }
