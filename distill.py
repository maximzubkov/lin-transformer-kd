import argparse
import logging
import math
import os
import random

import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AdamW,
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_scheduler,
    set_seed,
)

from models import make_attention_linear

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/gpt2.yaml",
        help="Path to the config",
    )
    args_ = parser.parse_args()
    config = OmegaConf.load(args_.config_path)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(config.cache_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(
        config.dataset.tokenizer,
        use_fast=not config.dataset.use_fast_tokenizer
    )

    # Prepare Teacher model
    teacher_model = AutoModelForCausalLM.from_pretrained(config.model, **{"output_attentions": True})
    teacher_model.resize_token_embeddings(len(tokenizer))

    for param in teacher_model.parameters():
        param.requires_grad = False
    teacher_model.eval()

    # Prepare Student model
    student_model = AutoModelForCausalLM.from_pretrained(config.model, **{"output_attentions": True})
    student_model.resize_token_embeddings(len(tokenizer))
    student_model = make_attention_linear(student_model)

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

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

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

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in student_model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.training.weight_decay,
        },
        {
            "params": [p for n, p in student_model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.training.learning_rate)

    # Prepare everything with our `accelerator`.
    student_model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        student_model.tie_weights()

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / config.training.gradient_accumulation_steps)
    max_train_steps = num_update_steps_per_epoch * config.training.num_train_epochs

    lr_scheduler = get_scheduler(
        name=config.training.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.training.num_warmup_steps,
        num_training_steps=max_train_steps,
    )

    # Train!
    total_batch_size = \
        config.training.per_device_train_batch_size * \
        accelerator.num_processes * \
        config.training.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config.training.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {config.training.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0

    for epoch in range(config.training.num_train_epochs):
        student_model.train()
        for step, batch in enumerate(train_dataloader):
            with torch.no_grad():
                teacher_outputs = teacher_model(**batch)
            student_outputs = student_model(**batch)
            loss = student_outputs.loss
            kd_losses = torch.cat([
                torch.nn.functional.mse_loss(a, b).reshape(1)
                for a, b in zip(student_outputs.attentions, teacher_outputs.attentions)
            ])
            loss = loss + kd_losses.mean()
            loss = loss / config.training.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % config.training.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

        student_model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs = student_model(**batch)

            loss = outputs.loss
            losses.append(accelerator.gather(loss.repeat(config.training.per_device_eval_batch_size)))

        losses = torch.cat(losses)
        losses = losses[: len(eval_dataset)]
        try:
            perplexity = math.exp(torch.mean(losses))
        except OverflowError:
            perplexity = float("inf")

        logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if config.push_to_hub and epoch < config.training.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(student_model)
            unwrapped_model.save_pretrained(config.cache_di, save_function=accelerator.save)
            if accelerator.is_main_process:
                tokenizer.save_pretrained(config.cache_dir)

    if config.cache_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(student_model)
        unwrapped_model.save_pretrained(config.cache_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(config.cache_dir)


if __name__ == "__main__":
    main()