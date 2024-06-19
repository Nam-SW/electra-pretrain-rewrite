import os
from typing import Optional

from datasets import disable_progress_bar, load_dataset
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer
from utils import text_processing


def load(
    tokenizer: PreTrainedTokenizer,
    load_args: dict,
    after_split_train_size: Optional[float] = None,
    shuffle_seed: Optional[int] = None,
    progressbar: Optional[bool] = True,
    num_worker: Optional[int] = None,
    train_key: str = "train",
    test_key: str = "test",
):
    num_worker = (
        min(max(1, num_worker), os.cpu_count()) if isinstance(num_worker, int) else os.cpu_count()
    )
    if not progressbar:
        disable_progress_bar()

    data = load_dataset(**OmegaConf.to_object(load_args))  # 안됨

    # =========== user define ===========

    def _process(batch):
        texts = sum([batch[key] for key in data.keys() if isinstance(batch[key], str)], [])
        return tokenizer(list(map(text_processing, texts)))

    # Write your own mapping, shuffling, and splitting sequences.

    if after_split_train_size is not None:
        assert (
            0.0 < after_split_train_size < 1.0
        ), "after_split_train_size must be a value between 0 and 1"
        data = data[train_key].train_test_split(train_size=after_split_train_size, shuffle=False)
        train_key = "train"
        test_key = "test"

    data = data.map(
        _process,
        batched=True,
        num_proc=num_worker,
        remove_columns=data["train"].column_names,
    )

    if shuffle_seed is not None:
        data = data.shuffle(seed=shuffle_seed)

    # =========== end user define ===========

    return data[train_key], data.get(test_key, None)
