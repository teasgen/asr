import re
from collections import defaultdict
from string import ascii_lowercase

import torch
try:
    import youtokentome as yttm
except e:
    print(f"Incorrently installed youtokentome: {e}")

from src.text_encoder import CTCTextEncoder


class BPECTCTextEncoder(CTCTextEncoder):
    def __init__(self, bpe_path, decoder_type="argmax", **kwargs):
        bpe = yttm.BPE(model=bpe_path)
        self.EMPTY_TOK = "<PAD>"
        self.SILENCE_TOKEN = "▁"
        assert decoder_type != "beam_search_lm", "Doesnt have pretrained LM for BPE"
        super().__init__(alphabet=bpe.vocab(), decoder_type=decoder_type, **kwargs)

        self.bpe = bpe
        self.vocab = self.bpe.vocab()
        self.ind2char.pop(0)

    def encode(self, text) -> torch.Tensor:
        return torch.Tensor(self.bpe.encode(text, output_type=yttm.OutputType.ID)).unsqueeze(0)

    def decode(self, inds) -> str:
        if not isinstance(inds, list):
            inds = inds.tolist()
        res = self.bpe.decode(inds)
        if len(res) == 0:
            return ""
        return res[0]

    def ctc_decode(self, inds) -> str:
        decoded = []
        for i in range(len(inds)):
            if len(decoded) == 0 or inds[i] != decoded[-1]:
                decoded.append(inds[i])
        decoded_final = []
        for i in range(len(decoded)):
            if decoded[i] != 0:
                decoded_final.append(decoded[i])
        return self.decode(decoded_final)
