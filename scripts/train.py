import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import average_precision_score, roc_auc_score


def train_one_epoch(model, loader, optimizer, device, cfg):
    model.train()
    total_loss = 0

    pos_weight = torch.ones(cfg.num_classes).to(device) * cfg.pos_weight
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for batch in tqdm(loader):
        if len(batch) == 4:
            images, input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        elif len(batch) == 3:
            images, _, labels = batch
            input_ids = attention_mask = None
        else:
            images, labels = batch
            input_ids = attention_mask = None

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if input_ids is not None:
            outputs = model(images, input_ids, attention_mask)
        else:
            outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device, cfg=None):
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                images, input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
            elif len(batch) == 3:
                images, _, labels = batch
                input_ids = attention_mask = None
            else:
                images, labels = batch
                input_ids = attention_mask = None

            images = images.to(device)
            labels = labels.to(device)

            if input_ids is not None:
                outputs = model(images, input_ids, attention_mask)
            else:
                outputs = model(images)

            probs = torch.sigmoid(outputs)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_labels = torch.cat(all_labels)

    all_labels = (all_labels > 0.5).float()
    preds = (all_probs > 0.5).float()

    f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    precision = precision_score(all_labels, preds, average="micro", zero_division=0)
    recall = recall_score(all_labels, preds, average="micro", zero_division=0)

    try:
        auprc = average_precision_score(all_labels, all_probs, average="macro")
    except Exception:
        auprc = 0.0

    try:
        auroc = roc_auc_score(all_labels, all_probs, average="macro")
    except Exception:
        auroc = 0.0

    return f1, precision, recall, auprc, auroc