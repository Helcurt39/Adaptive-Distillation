import torch.nn as nn
import torch
import torchvision.models as models
from transformers import AutoModel


class ChestXrayModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Backbone selection
        if cfg.backbone == "resnet50":
            base = models.resnet50(pretrained=True)
        else:
            base = models.resnet18(pretrained=True)

        # Freeze control
        for param in base.parameters():
            param.requires_grad = not cfg.freeze_backbone

        # Replace final FC
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, cfg.num_classes)
        )

        # Ensure final layer is trainable
        for param in base.fc.parameters():
            param.requires_grad = True

        self.model = base

    def forward(self, x):
        return self.model(x)


class ChestXrayModelMultimodal(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Image encoder
        base = models.resnet18(pretrained=True)

        for param in base.parameters():
            param.requires_grad = not cfg.freeze_backbone

        # Unfreeze last blocks
        for param in base.layer3.parameters():
            param.requires_grad = True
        for param in base.layer4.parameters():
            param.requires_grad = True

        base.fc = nn.Identity()
        self.image_encoder = base

        # Text encoder (frozen except last layers)
        self.text_encoder = AutoModel.from_pretrained(cfg.text_model)

        for param in self.text_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(cfg.image_embed_dim + cfg.text_embed_dim, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(cfg.fusion_hidden, cfg.num_classes)
        )

    def forward(self, image, input_ids, attention_mask):
        img_feat = self.image_encoder(image)
        text_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = text_out.last_hidden_state[:, 0, :]

        fused = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(fused)