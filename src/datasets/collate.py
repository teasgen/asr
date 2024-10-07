from collections import defaultdict

import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Union[Tensor, List]]): dict, containing batch-version
            of the tensors or lists.
    """
    pack_to_tensors_batch_keys = ["spectrogram", "text_encoded"]
    pack_to_list_batch_keys = ["text", "audio_path", "audio"]
    result_batch = {}
    for key in pack_to_tensors_batch_keys + pack_to_list_batch_keys:
        lengths = []
        list_of_batch_values = []
        want_tensor = key in pack_to_tensors_batch_keys
        for item in dataset_items:
            list_of_batch_values.append(item[key])
            if want_tensor:
                lengths.append(item[key].shape[-1])
        if want_tensor:
            max_len = max(lengths)
            # pad all tensors in samples to the max length in batch and save orinal lengths
            list_of_batch_values = [
                torch.nn.functional.pad(x, (0, max_len - x.shape[-1]))
                for x in list_of_batch_values
            ]
            result_batch[key] = torch.cat(list_of_batch_values, dim=0).to(torch.bfloat16)
            result_batch[f"{key}_length"] = torch.tensor(lengths)
        else:
            result_batch[key] = list_of_batch_values
    return result_batch
