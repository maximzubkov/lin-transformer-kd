import torch
import math
import wandb

def evaluate_lm(
    student_model: ,
    teacher_model:, 
    eval_dataloader:, 
    accelerator: ,
    per_device_eval_batch_size
):
    student_model.eval()
    student_losses, teacher_losses = [], []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
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
        f"epoch {epoch}: student perplexity: {student_perplexity}, teacher perplexity: {teacher_perplexity}"
    )
    if accelerator.is_local_main_process:
        wandb.log({
            'val_student_perplexity': student_perplexity,
            'val_teacher_perplexity': teacher_perplexity
        })
