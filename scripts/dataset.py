import re
import io
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Clinical classes
CLASSES = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "effusion", "emphysema", "fibrosis", "hernia",
    "infiltration", "mass", "nodule", "opacity",
    "pleural_thickening", "pneumonia", "pneumothorax",
]

KEYWORDS = {
    "atelectasis": ["atelectasis", "volume loss", "subsegmental atelectasis",
                    "basilar atelectasis", "bibasilar atelectasis",
                    "discoid atelectasis", "linear atelectasis"],
    "cardiomegaly": ["cardiomegaly", "enlarged heart", "heart is enlarged",
                     "cardiac silhouette is enlarged", "mildly enlarged cardiac",
                     "borderline enlarged", "mild cardiomegaly"],
    "consolidation": ["consolidation", "lobar consolidation", "airspace disease",
                      "airspace opacity", "airspace opacities"],
    "edema": ["edema", "pulmonary edema", "interstitial edema",
              "vascular congestion", "pulmonary vascular congestion",
              "interstitial abnormality", "vascular prominence"],
    "effusion": ["effusion", "pleural effusion", "pleural effusions",
                 "layering fluid", "loculated fluid"],
    "emphysema": ["emphysema", "hyperinflation", "hyperexpansion",
                  "air trapping", "bullous disease"],
    "fibrosis": ["fibrosis", "fibrotic", "scarring", "scar",
                 "interstitial fibrosis", "pulmonary fibrosis"],
    "hernia": ["hernia", "hiatal hernia", "diaphragmatic hernia"],
    "infiltration": ["infiltrate", "infiltrates", "infiltration"],
    "mass": ["mass", "masses", "soft tissue mass", "lung mass"],
    "nodule": ["nodule", "nodules", "pulmonary nodule", "lung nodule",
               "nodular opacity", "granuloma"],
    "opacity": ["opacity", "opacification", "haziness", "density",
                "ground glass", "ground-glass"],
    "pleural_thickening": ["pleural thickening", "pleural scarring",
                          "pleural calcification", "blunting"],
    "pneumonia": ["pneumonia", "pneumonic", "aspiration",
                  "infectious", "lobar pneumonia"],
    "pneumothorax": ["pneumothorax"],
}

# Negation / uncertainty handling
NEGATIONS = [
    "no evidence of", "no signs of", "no definite", "no focal",
    "no acute", "no large", "without", "not seen",
    "resolved", "clear of", "absent", "free of", "negative for",
]

UNCERTAINTIES = [
    "possible", "probable", "may represent", "may reflect", "may be",
    "cannot exclude", "cannot rule out", "suggest", "suggests",
    "concerning for", "suspicious for", "likely", "consistent with",
    "could represent", "worrisome", "question of",
]


def _sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', text) if s.strip()]


def _score(sentence: str, keywords: list) -> int:
    s = sentence.lower()
    for kw in keywords:
        if kw not in s:
            continue
        idx = s.index(kw)
        before = s[max(0, idx - 80): idx]
        window = s[max(0, idx - 80): idx + len(kw) + 80]

        if any(neg in before for neg in NEGATIONS):
            continue
        if any(unc in window for unc in UNCERTAINTIES):
            return -1
        return 1
    return 0


def extract_labels(report: str) -> torch.Tensor:
    labels = {cls: 0 for cls in CLASSES}

    for sentence in _sentences(report):
        for cls in CLASSES:
            s = _score(sentence, KEYWORDS[cls])
            if s == 1:
                labels[cls] = 1
            elif s == -1 and labels[cls] == 0:
                labels[cls] = -1

    # uncertain → 0.5
    vec = [0.5 if labels[cls] == -1 else float(labels[cls]) for cls in CLASSES]
    return torch.tensor(vec, dtype=torch.float32)


class ChestXrayDataset(Dataset):
    def __init__(self, parquet_file, image_size=224, augment=False):
        if isinstance(parquet_file, list):
            dfs = [pd.read_parquet(p) for p in parquet_file]
            self.df = pd.concat(dfs, ignore_index=True)
        else:
            self.df = pd.read_parquet(parquet_file)

        # Precompute labels once
        self.labels = torch.stack([extract_labels(r) for r in self.df["reports"]])
        N = self.labels.shape[0]

        pos_counts = (self.labels > 0).float().sum(dim=0)
        neg_counts = N - pos_counts
        pos_weight = neg_counts / (pos_counts + 1e-6)

        print("\nClass Imbalance (pos_weight):")
        for i, cls in enumerate(CLASSES):
            print(f"{cls:<20} {pos_weight[i].item():.2f}")

        self.pos_weight = pos_weight
        self.transform = self._build_transform(image_size, augment)

    def _build_transform(self, image_size, augment):
        if augment:
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def decode_image(self, image_dict):
        return Image.open(io.BytesIO(image_dict["bytes"])).convert("RGB")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = self.transform(self.decode_image(row["image"]))
        report = str(row["reports"])  # raw text
        label = self.labels[idx]
        return image, report, label