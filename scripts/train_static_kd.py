import torch
import torch.nn.functional as F
from tqdm import tqdm


def distill_loss_bce(student_logits, teacher_logits, T):
    student_prob = torch.sigmoid(student_logits / T)
    teacher_prob = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy(student_prob, teacher_prob)


def train_static_kd(
    student,
    teacher,
    loader,
    optimizer,
    device,
    cfg,
    lambda_const=0.5
):
    student.train()
    teacher.eval()

    total_loss = 0

    for batch in tqdm(loader):
        if len(batch) == 4:
            images, input_ids, attention_mask, labels = batch
            input_ids      = input_ids.to(device)
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

        # classification loss
        l_cls = F.binary_cross_entropy_with_logits(student_logits, labels)

        # distillation loss
        l_distill = distill_loss_bce(student_logits, teacher_logits, cfg.temperature)

        # static lambda
        loss = l_cls + lambda_const * l_distill

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)