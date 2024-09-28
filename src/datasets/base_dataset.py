import logging
import random
<<<<<<< HEAD
from typing import List

import torch
from torch.utils.data import Dataset

=======

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from src.text_encoder import CTCTextEncoder

>>>>>>> asr
logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    Base class for the datasets.

    Given a proper index (list[dict]), allows to process different datasets
    for the same task in the identical manner. Therefore, to work with
    several datasets, the user only have to define index in a nested class.
    """

    def __init__(
<<<<<<< HEAD
        self, index, limit=None, shuffle_index=False, instance_transforms=None
=======
        self,
        index,
        text_encoder=None,
        target_sr=16000,
        limit=None,
        max_audio_length=None,
        max_text_length=None,
        shuffle_index=False,
        instance_transforms=None,
>>>>>>> asr
    ):
        """
        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
<<<<<<< HEAD
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
=======
            text_encoder (CTCTextEncoder): text encoder.
            target_sr (int): supported sample rate.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
>>>>>>> asr
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
            instance_transforms (dict[Callable] | None): transforms that
                should be applied on the instance. Depend on the
                tensor name.
        """
        self._assert_index_is_valid(index)

<<<<<<< HEAD
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

=======
        index = self._filter_records_from_dataset(
            index, max_audio_length, max_text_length
        )
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        if not shuffle_index:
            index = self._sort_index(index)

        self._index: list[dict] = index

        self.text_encoder = text_encoder
        self.target_sr = target_sr
>>>>>>> asr
        self.instance_transforms = instance_transforms

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
<<<<<<< HEAD
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["label"]

        instance_data = {"data_object": data_object, "labels": data_label}
=======
        audio_path = data_dict["path"]
        audio = self.load_audio(audio_path)
        text = data_dict["text"]
        text_encoded = self.text_encoder.encode(text)

        spectrogram = self.get_spectrogram(audio)

        instance_data = {
            "audio": audio,
            "spectrogram": spectrogram,
            "text": text,
            "text_encoded": text_encoded,
            "audio_path": audio_path,
        }

        # TODO think of how to apply wave augs before calculating spectrogram
        # Note: you may want to preserve both audio in time domain and
        # in time-frequency domain for logging
>>>>>>> asr
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def __len__(self):
        """
        Get length of the dataset (length of the index).
        """
        return len(self._index)

<<<<<<< HEAD
    def load_object(self, path):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        data_object = torch.load(path)
        return data_object
=======
    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.target_sr
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def get_spectrogram(self, audio):
        """
        Special instance transform with a special key to
        get spectrogram from audio.

        Args:
            audio (Tensor): original audio.
        Returns:
            spectrogram (Tensor): spectrogram for the audio.
        """
        return self.instance_transforms["get_spectrogram"](audio)
>>>>>>> asr

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
<<<<<<< HEAD
=======
                if transform_name == "get_spectrogram":
                    continue  # skip special key
>>>>>>> asr
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data

    @staticmethod
    def _filter_records_from_dataset(
        index: list,
<<<<<<< HEAD
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        some condition.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting.
=======
        max_audio_length,
        max_text_length,
    ) -> list:
        """
        Filter some of the elements from the dataset depending on
        the desired max_test_length or max_audio_length.
>>>>>>> asr

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
<<<<<<< HEAD
=======
            max_audio_length (int): maximum allowed audio length.
            max_test_length (int): maximum allowed text length.
>>>>>>> asr
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset that satisfied the condition. The dict has
                required metadata information, such as label and object path.
        """
<<<<<<< HEAD
        # Filter logic
        pass
=======
        initial_size = len(index)
        if max_audio_length is not None:
            exceeds_audio_length = (
                np.array([el["audio_len"] for el in index]) >= max_audio_length
            )
            _total = exceeds_audio_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_audio_length} seconds. Excluding them."
            )
        else:
            exceeds_audio_length = False

        initial_size = len(index)
        if max_text_length is not None:
            exceeds_text_length = (
                np.array(
                    [len(CTCTextEncoder.normalize_text(el["text"])) for el in index]
                )
                >= max_text_length
            )
            _total = exceeds_text_length.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) records are longer then "
                f"{max_text_length} characters. Excluding them."
            )
        else:
            exceeds_text_length = False

        records_to_filter = exceeds_text_length | exceeds_audio_length

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total} ({_total / initial_size:.1%}) records  from dataset"
            )

        return index
>>>>>>> asr

    @staticmethod
    def _assert_index_is_valid(index):
        """
        Check the structure of the index and ensure it satisfies the desired
        conditions.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """
        for entry in index:
            assert "path" in entry, (
                "Each dataset item should include field 'path'" " - path to audio file."
            )
<<<<<<< HEAD
            assert "label" in entry, (
                "Each dataset item should include field 'label'"
                " - object ground-truth label."
=======
            assert "text" in entry, (
                "Each dataset item should include field 'text'"
                " - object ground-truth transcription."
            )
            assert "audio_len" in entry, (
                "Each dataset item should include field 'audio_len'"
                " - length of the audio."
>>>>>>> asr
            )

    @staticmethod
    def _sort_index(index):
        """
<<<<<<< HEAD
        Sort index via some rules.

        This is not used in the example. The method should be called in
        the __init__ before shuffling and limiting and after filtering.
=======
        Sort index by audio length.
>>>>>>> asr

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        Returns:
            index (list[dict]): sorted list, containing dict for each element
                of the dataset. The dict has required metadata information,
                such as label and object path.
        """
<<<<<<< HEAD
        return sorted(index, key=lambda x: x["KEY_FOR_SORTING"])
=======
        return sorted(index, key=lambda x: x["audio_len"])
>>>>>>> asr

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        """
        Shuffle elements in index and limit the total number of elements.

        Args:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
            limit (int | None): if not None, limit the total number of elements
                in the dataset to 'limit' elements.
            shuffle_index (bool): if True, shuffle the index. Uses python
                random package with seed 42.
        """
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index
