import argparse
import logging
import math
import os

import wandb
import datasets
import torch
import transformers
from accelerate import Accelerator, DistributedType
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import (
    MODEL_MAPPING,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)

from models import make_attention_linear
from training import get_dataset
from training.optimizer import get_optimizer

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

        wandb.init(project="distillation")
        wandb.config.update(config)  # Log args
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    set_seed(config.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(config.cache_dir, exist_ok=True)
    accelerator.wait_for_everyone()

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
    teacher_model = teacher_model.to(accelerator.device)

    # Prepare Student model
    student_model = AutoModelForCausalLM.from_pretrained(config.model, **{"output_attentions": True})
    student_model.resize_token_embeddings(len(tokenizer))
    student_model = make_attention_linear(student_model)

    train_dataset, eval_dataset, train_dataloader, eval_dataloader = get_dataset(
        config=config, tokenizer=tokenizer, accelerator=accelerator
    )
    optimizer = get_optimizer(config, model=student_model)

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

            if accelerator.is_local_main_process:
                wandb.log(
                    {
                        'LM_train_loss': student_outputs.loss.item(),
                        'KD_loss': kd_losses.mean().item(),
                        'totol_train_loss': loss.item()
                    }
                )

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
        if accelerator.is_local_main_process:
            wandb.log({'val_perplexity': perplexity})

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
