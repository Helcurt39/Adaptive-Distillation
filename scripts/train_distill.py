import torch
from tqdm import tqdm
from distillation import combined_loss


def train_distillation(
    student,
    teacher,
    loader,
    optimizer,
    device,
    epoch,
    total_epochs,
    cfg,
    current_round
):
    student.train()
    teacher.eval()

    total_loss = 0
    total_lambda = 0
    total_gate = 0

    for batch in tqdm(loader):
        optimizer.zero_grad()

        if len(batch) == 4:
            images, input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        else:
            images, labels = batch
            input_ids = attention_mask = None

        images = images.to(device)
        labels = labels.to(device)

        if input_ids is not None:
            student_logits = student(images, input_ids, attention_mask)
            with torch.no_grad():
                teacher_logits = teacher(images, input_ids, attention_mask)
        else:
            student_logits = student(images)
            with torch.no_grad():
                teacher_logits = teacher(images)

        loss, lam, l_cls, l_distill, gate_rate = combined_loss(
            student_logits,
            teacher_logits,
            labels,
            epoch,
            total_epochs,
            cfg,
            current_round
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_lambda += lam
        total_gate += gate_rate

    avg_loss = total_loss / len(loader)
    avg_lambda = total_lambda / len(loader)
    avg_gate = total_gate / len(loader)

    return avg_loss, avg_lambda, avg_gate