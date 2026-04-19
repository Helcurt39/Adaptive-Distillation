import torch
import torch.nn.functional as F
from tqdm import tqdm

def distill_loss_kl(student_logits, teacher_logits, T):
    student_log_prob = F.log_softmax(student_logits / T, dim=0)
    teacher_prob     = F.softmax(teacher_logits / T, dim=0)
    return F.kl_div(student_log_prob, teacher_prob, reduction='batchmean') * (T ** 2)

def train_confidence_kl(student, teacher, loader, optimizer, device, cfg):
    student.train()
    teacher.eval()
    total_loss   = 0
    total_lambda = 0

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

        # KL distillation loss
        l_distill = distill_loss_kl(student_logits, teacher_logits, cfg.temperature)

        # confidence-based lambda — max teacher prob per sample, averaged
        teacher_probs = torch.sigmoid(teacher_logits)
        confidence    = teacher_probs.max(dim=1).values
        lam           = float(confidence.mean().item())
        lam           = max(0.05, min(0.95, lam))

        loss = l_cls + lam * l_distill
        loss.backward()
        optimizer.step()

        total_loss   += loss.item()
        total_lambda += lam

    return total_loss / len(loader), total_lambda / len(loader)