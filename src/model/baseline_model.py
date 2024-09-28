from torch import nn
from torch.nn import Sequential


class BaselineModel(nn.Module):
    """
    Simple MLP
    """

<<<<<<< HEAD
    def __init__(self, n_feats, n_class, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_class (int): number of classes.
=======
    def __init__(self, n_feats, n_tokens, fc_hidden=512):
        """
        Args:
            n_feats (int): number of input features.
            n_tokens (int): number of tokens in the vocabulary.
>>>>>>> asr
            fc_hidden (int): number of hidden features.
        """
        super().__init__()

        self.net = Sequential(
            # people say it can approximate any function...
            nn.Linear(in_features=n_feats, out_features=fc_hidden),
            nn.ReLU(),
            nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
            nn.ReLU(),
<<<<<<< HEAD
            nn.Linear(in_features=fc_hidden, out_features=n_class),
        )

    def forward(self, data_object, **batch):
=======
            nn.Linear(in_features=fc_hidden, out_features=n_tokens),
        )

    def forward(self, spectrogram, spectrogram_length, **batch):
>>>>>>> asr
        """
        Model forward method.

        Args:
<<<<<<< HEAD
            data_object (Tensor): input vector.
        Returns:
            output (dict): output dict containing logits.
        """
        return {"logits": self.net(data_object)}
=======
            spectrogram (Tensor): input spectrogram.
            spectrogram_length (Tensor): spectrogram original lengths.
        Returns:
            output (dict): output dict containing log_probs and
                transformed lengths.
        """
        output = self.net(spectrogram.transpose(1, 2))
        log_probs = nn.functional.log_softmax(output, dim=-1)
        log_probs_length = self.transform_input_lengths(spectrogram_length)
        return {"log_probs": log_probs, "log_probs_length": log_probs_length}

    def transform_input_lengths(self, input_lengths):
        """
        As the network may compress the Time dimension, we need to know
        what are the new temporal lengths after compression.

        Args:
            input_lengths (Tensor): old input lengths
        Returns:
            output_lengths (Tensor): new temporal lengths
        """
        return input_lengths  # we don't reduce time dimension here
>>>>>>> asr

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
