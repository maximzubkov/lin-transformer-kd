import logging
import random

from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    PreTrainedTokenizer,
    default_data_collator,
)

logger = logging.getLogger(__name__)


def get_dataset(config, tokenizer: PreTrainedTokenizer, accelerator: Accelerator):
    raw_datasets = load_dataset(config.dataset.name, config.dataset.config_name)
    if "validation" not in raw_datasets.keys():
        raw_datasets["validation"] = load_dataset(
            config.dataset.name,
            config.dataset.config_name,
            split=f"train[:{config.dataset.validation_split_percentage}%]",
        )
        raw_datasets["train"] = load_dataset(
            config.dataset.name,
            config.dataset.config_name,
            split=f"train[{config.dataset.validation_split_percentage}%:]",
        )

    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=config.num_workers,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

    block_size = tokenizer.model_max_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=config.num_workers,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=config.training.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        collate_fn=default_data_collator,
        batch_size=config.training.per_device_eval_batch_size
    )

    return train_dataset, eval_dataset, train_dataloader, eval_dataloader
