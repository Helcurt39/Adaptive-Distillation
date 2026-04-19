import torch
import torch.nn.functional as F
from tqdm import tqdm


def train_confidence(
    student,
    teacher,
    loader,
    optimizer,
    device,
    cfg
):
    student.train()
    teacher.eval()

    total_loss = 0
    total_lambda = 0

    for batch in tqdm(loader):
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

        # Classification loss
        l_cls = F.binary_cross_entropy_with_logits(student_logits, labels)

        # Distillation loss
        student_prob = torch.sigmoid(student_logits / cfg.temperature)
        teacher_prob = torch.sigmoid(teacher_logits / cfg.temperature)
        l_distill = F.binary_cross_entropy(student_prob, teacher_prob)

        # Confidence-based lambda (max confidence)
        teacher_probs = torch.sigmoid(teacher_logits)
        confidence = teacher_probs.max(dim=1).values
        lam = confidence.mean().item()
        lam = max(0.05, min(0.95, lam))  # clamp

        # Total loss
        loss = (1 - lam) * l_cls + lam * l_distill

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_lambda += lam

    avg_loss = total_loss / len(loader)
    avg_lambda = total_lambda / len(loader)

    return avg_loss, avg_lambda