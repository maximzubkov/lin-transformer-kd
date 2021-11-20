import math
import logging

import torch
import wandb
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    step: int,
    student_model: PreTrainedModel,
    teacher_model: PreTrainedModel,
    eval_dataset: Dataset,
    eval_dataloader: DataLoader,
    accelerator: Accelerator,
    per_device_eval_batch_size: int
):
    student_model.eval()
    student_losses, teacher_losses = [], []
    for _, batch in enumerate(eval_dataloader):
        student_outputs = student_model(**batch)
        teacher_outputs = teacher_model(**batch)

        student_loss = student_outputs.loss
        teacher_loss = teacher_outputs.loss

        student_losses.append(accelerator.gather(student_loss.repeat(per_device_eval_batch_size)))
        teacher_losses.append(accelerator.gather(teacher_loss.repeat(per_device_eval_batch_size)))

    student_losses = torch.cat(student_losses)
    teacher_losses = torch.cat(teacher_losses)
    student_losses = student_losses[: len(eval_dataset)]
    teacher_losses = teacher_losses[: len(eval_dataset)]
    try:
        student_perplexity = math.exp(torch.mean(student_losses))
    except OverflowError:
        student_perplexity = float("inf")
    try:
        teacher_perplexity = math.exp(torch.mean(teacher_losses))
    except OverflowError:
        teacher_perplexity = float("inf")

    logger.info(
        f"epoch {step}: student perplexity: {student_perplexity}, teacher perplexity: {teacher_perplexity}"
    )
    if accelerator.is_local_main_process:
        wandb.log({
            'val_student_perplexity': student_perplexity,
            'val_teacher_perplexity': teacher_perplexity
        })
    student_model.train()
