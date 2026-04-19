from dataclasses import dataclass


@dataclass
class Config:
    # Dataset paths
    train_path = [
        "../../Datasets/MIMIC-CXR/train-00000-of-00002.parquet",
        "../../Datasets/MIMIC-CXR/train-00001-of-00002.parquet"
    ]
    val_path = "../../Datasets/MIMIC-CXR/validation-00000-of-00001.parquet"
    test_path = "../../Datasets/MIMIC-CXR/test-00000-of-00001.parquet"

    # Labels
    classes = [
        "atelectasis", "cardiomegaly", "consolidation", "edema",
        "effusion", "emphysema", "fibrosis", "hernia",
        "infiltration", "mass", "nodule", "opacity",
        "pleural_thickening", "pneumonia", "pneumothorax",
    ]
    num_classes = len(classes)

    # Data
    image_size = 224
    batch_size = 32
    num_workers = 0
    pin_memory = True

    # Training
    epochs = 2
    lr = 1e-4

    # Distillation defaults
    temperature = 1.0
    lambda_max = 0.5
    curriculum_k = 0.0
    confidence_tau = 0.0
    epsilon = 2e-5
    confidence_temperature = 2.0

    # Federated learning
    num_clients = 4
    num_rounds = 20
    distill_start_round = 1

    # Model
    backbone = "resnet18"
    freeze_backbone = False
    pos_weight = 3.0

    # Text modality
    use_text = True
    text_model = "emilyalsentzer/Bio_ClinicalBERT"
    text_max_len = 256
    text_embed_dim = 768
    image_embed_dim = 512
    fusion_hidden = 1024

    # Logging fields
    HEADER = [
        "experiment", "round_or_epoch", "loss", "f1",
        "precision", "recall", "auprc", "auroc",
        "lambda", "gate_rate"
    ]

    # Mode
    distill_mode = "baseline"
    noniid_alpha = 0.5


# Dynamic config generator
def get_config(mode: str):
    cfg = Config()
    cfg.distill_mode = mode

    if mode == "baseline":
        cfg.lambda_max = 0.0

    elif mode in ["static", "mse"]:
        cfg.lambda_max = 0.5

    elif mode == "kl":
        cfg.temperature = 1.0
        cfg.lambda_max = 0.5

    elif mode == "loss_adaptive":
        pass  # computed dynamically

    elif mode == "loss_adaptive_kl":
        cfg.temperature = 1.0

    elif mode == "confidence":
        cfg.confidence_tau = 0.0

    elif mode == "confidence_kl":
        cfg.temperature = 1.0
        cfg.confidence_tau = 0.0

    elif mode == "hybrid":
        cfg.distill_start_round = 1
        cfg.lambda_max = 0.5
        cfg.confidence_tau = 0.0
        cfg.curriculum_k = 0.0

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cfg