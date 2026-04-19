import torch
import torch.nn.functional as F
import math


# Classification loss
def classification_loss(logits, labels, cfg, device):
    pos_weight = torch.ones(cfg.num_classes).to(device) * cfg.pos_weight
    return F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


# Distillation loss
def distillation_loss(student_logits, teacher_logits, temperature):
    T = temperature
    student_prob = torch.sigmoid(student_logits / T)
    teacher_prob = torch.sigmoid(teacher_logits / T)
    return F.binary_cross_entropy(student_prob, teacher_prob)


# Curriculum lambda schedule
def curriculum_lambda(current_round, total_rounds, lambda_max, k=10):
    t = current_round / max(total_rounds, 1)
    return lambda_max * (1 / (1 + math.exp(-k * (t - 0.5))))


# Combined loss
def combined_loss(
    student_logits,
    teacher_logits,
    labels,
    epoch,
    total_epochs,
    cfg,
    current_round
):
    # No distillation before start round
    if current_round < cfg.distill_start_round:
        l_cls = classification_loss(student_logits, labels, cfg, student_logits.device)
        return l_cls, 0.0, l_cls.item(), 0.0, 0.0

    # Temperature warm-up
    warmup_rounds = 3
    ramp = min(1.0, (current_round - cfg.distill_start_round) / warmup_rounds)
    T = 1.0 + ramp * (cfg.temperature - 1.0)

    l_cls = classification_loss(student_logits, labels, cfg, student_logits.device)

    # Confidence gating (max per sample)
    teacher_probs = torch.sigmoid(teacher_logits)
    confidence = teacher_probs.max(dim=1).values
    mask = confidence > cfg.confidence_tau

    # Ensure minimum samples kept
    min_keep = int(0.5 * len(confidence))
    if mask.sum() < min_keep:
        topk = torch.topk(confidence, min_keep).indices
        mask = torch.zeros_like(confidence, dtype=torch.bool)
        mask[topk] = True

    gate_rate = mask.float().mean().item()

    # No distillation if all filtered
    if mask.sum() == 0:
        return l_cls, 0.0, l_cls.item(), 0.0, gate_rate

    # Distillation loss on gated samples
    student_prob = torch.sigmoid(student_logits / T)[mask]
    teacher_prob = torch.sigmoid(teacher_logits / T)[mask]
    l_distill = F.binary_cross_entropy(student_prob, teacher_prob)

    # Lambda (curriculum + adaptive)
    lam_curr = curriculum_lambda(
        current_round,
        cfg.num_rounds,
        cfg.lambda_max,
        cfg.curriculum_k
    )

    lam_adapt = float(
        l_cls.detach() / (l_cls.detach() + l_distill.detach() + cfg.epsilon)
    )
    lam_adapt = min(lam_adapt, cfg.lambda_max)

    lam = 0.3 * lam_curr + 0.7 * lam_adapt
    lam = max(0.0, min(cfg.lambda_max, lam))

    # Total loss
    total_loss = (1 - lam) * l_cls + lam * l_distill

    return total_loss, lam, l_cls.item(), l_distill.item(), gate_rate
