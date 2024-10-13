import re
from collections import defaultdict
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder

from .install_lm import install_lm


class CTCTextEncoder:
    EMPTY_TOK = ""
    SILENCE_TOKEN = " "

    def __init__(self, alphabet=None, decoder_type="argmax", **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet

        assert decoder_type in ["argmax", "beam_search", "beam_search_lm", "beam_search_torch"], "Choose one of them pls"
        if decoder_type == "beam_search_lm":
            lm_files = install_lm()
            unigrams = [
                i.split("\t")[0].lower().strip()
                for i in open(lm_files["lexicon"]).readlines()
            ]
            self.lm_decoder = build_ctcdecoder(
                [self.EMPTY_TOK] + self.alphabet,
                kenlm_model_path=lm_files["lm_path"],
                alpha=0.6,
                beta=0.5,
                unigrams=unigrams
            )
        self.decoder_type = decoder_type
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.empty_token_idx = 0

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        decoded = []
        last_char_idx = self.empty_token_idx
        for char_idx in inds:
            if char_idx == self.empty_token_idx or char_idx == last_char_idx:
                last_char_idx = char_idx
                continue
            last_char_idx = char_idx
            decoded.append(self.ind2char[char_idx])
        return "".join(decoded)

    def get_prediction(
        self,
        log_probs: torch.Tensor,
        log_probs_length: torch.Tensor,
    ) -> list[list[str]]:
        log_probs_length = log_probs_length.detach().numpy()
        log_probs = log_probs.detach().cpu()
        pred_texts = []
        if self.decoder_type == "argmax":
            predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
            for log_prob_vec, length in zip(predictions, log_probs_length):
                pred_texts.append([self.ctc_decode(log_prob_vec[:length])])
            return pred_texts
        elif self.decoder_type == "beam_search":
            for log_probs_line, length in zip(log_probs, log_probs_length):
                preds = self.ctc_beam_search(log_probs_line[:].exp().numpy(), length, 50)
                preds = [hypo for (hypo, _) in preds]
                pred_texts.append(preds)
            return pred_texts
        elif self.decoder_type == "beam_search_lm":
            log_probs = log_probs.cpu().numpy()
            for log_probs_line, length in zip(log_probs, log_probs_length):
                preds = self.lm_decoder.decode_beams(log_probs_line, 50)
                preds = [x[0] for x in preds]
                pred_texts.append(preds)
            return pred_texts
        elif self.decoder_type == "beam_search_torch":
            from torchaudio.models.decoder import ctc_decoder
            decoder = ctc_decoder(
                lexicon=None,
                tokens=self.vocab,
                blank_token=self.EMPTY_TOK,
                sil_token=self.SILENCE_TOKEN,
                nbest=1,
                beam_size=10,
            )
            pred_hypos = decoder(log_probs.to(torch.float32), torch.from_numpy(log_probs_length))
            # pred_texts = [["".join(decoder.idxs_to_tokens(hypo.tokens)).strip() for hypo in preds] for preds in pred_hypos]
            pred_texts = [[self.decode(hypo.tokens).strip() for hypo in preds] for preds in pred_hypos]
            return pred_texts
        else:
            raise NotImplementedError

    def expand_merge(self, dp: dict, next_probs: torch.Tensor):
        new_dp = defaultdict(float)
        for idx, next_prob in enumerate(next_probs):
            cur_char = self.ind2char[idx]
            for (prefix, last_char), proba in dp.items():
                if last_char == cur_char or (cur_char == self.EMPTY_TOK):
                    new_prefix = prefix
                else:
                    new_prefix = prefix + cur_char
                new_dp[(new_prefix, cur_char)] += proba * next_prob
        return new_dp

    def truncate(self, dp: dict, beam_size: int):
        return dict(sorted(dp.items(), key=lambda x: -x[1])[:beam_size])

    def ctc_beam_search(self, probs: torch.Tensor, length: torch.Tensor, beam_size: int) -> list:
        dp = {("", self.EMPTY_TOK): 1.0}
        for proba in probs[:length]:
            dp = self.expand_merge(dp, proba)
            dp = self.truncate(dp, beam_size)
        return [
            (prefix, proba)
            for (prefix, _), proba in sorted(dp.items(), key=lambda x: -x[1])
        ]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
