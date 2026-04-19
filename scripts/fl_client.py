import flwr as fl
import torch
import torch.optim as optim

from train_kl_kd import train_kl_kd
from train_loss_adaptive_kl import train_loss_adaptive_kl
from train_confidence_kl import train_confidence_kl

from train import train_one_epoch, evaluate
from train_static_kd import train_static_kd
from train_mse_kd import train_mse_kd
from train_loss_adaptive import train_loss_adaptive
from train_confidence import train_confidence
from train_distill import train_distillation

from collate import get_collate_fn
from model import ChestXrayModel, ChestXrayModelMultimodal
from logger import log_results

from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    average_precision_score, roc_auc_score
)

HEADER = [
    "experiment","round","loss","f1","precision","recall",
    "auprc","auroc","lambda","gate_rate"
]


class FLClient(fl.client.NumPyClient):
    def __init__(self, train_loader, val_loader, cfg, device, cid):
        self.model = (
            ChestXrayModelMultimodal(cfg).to(device)
            if cfg.use_text else
            ChestXrayModel(cfg).to(device)
        )

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.cid = cid

        self.current_round = 0
        self.last_lambda = 0
        self.last_gate = 0

    # Parameter exchange
    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict(
            {k: torch.tensor(v) for k, v in state_dict.items()},
            strict=True
        )

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        current_round = config.get("server_round", 1)
        self.current_round = config.get("server_round", 0)

        # Global model as teacher
        teacher = (
            ChestXrayModelMultimodal(self.cfg).to(self.device)
            if self.cfg.use_text else
            ChestXrayModel(self.cfg).to(self.device)
        )
        teacher.load_state_dict(self.model.state_dict())
        teacher.eval()

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=1e-4
        )

        mode = self.cfg.distill_mode

        if mode == "baseline":
            for _ in range(self.cfg.epochs):
                train_loss = train_one_epoch(
                    self.model, self.train_loader, optimizer, self.device, self.cfg
                )
            avg_lambda, gate_rate = 0, 0

        elif mode == "static":
            for _ in range(self.cfg.epochs):
                train_loss = train_static_kd(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            avg_lambda, gate_rate = 0.5, 0

        elif mode == "mse":
            for _ in range(self.cfg.epochs):
                train_loss = train_mse_kd(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            avg_lambda, gate_rate = 0.5, 0

        elif mode == "loss_adaptive":
            for _ in range(self.cfg.epochs):
                train_loss, avg_lambda = train_loss_adaptive(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            gate_rate = 0.0

        elif mode == "confidence":
            for _ in range(self.cfg.epochs):
                train_loss, avg_lambda = train_confidence(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            gate_rate = 0.0

        elif mode == "kl":
            for _ in range(self.cfg.epochs):
                train_loss = train_kl_kd(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            avg_lambda, gate_rate = 0.5, 0

        elif mode == "loss_adaptive_kl":
            for _ in range(self.cfg.epochs):
                train_loss, avg_lambda = train_loss_adaptive_kl(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            gate_rate = 0

        elif mode == "confidence_kl":
            for _ in range(self.cfg.epochs):
                train_loss, avg_lambda = train_confidence_kl(
                    self.model, teacher, self.train_loader, optimizer, self.device, self.cfg
                )
            gate_rate = 0

        elif mode == "hybrid":
            for epoch in range(self.cfg.epochs):
                global_epoch = (self.current_round - 1) * self.cfg.epochs + epoch
                total_epochs = self.cfg.num_rounds * self.cfg.epochs

                train_loss, avg_lambda, gate_rate = train_distillation(
                    self.model, teacher, self.train_loader,
                    optimizer, self.device,
                    epoch=global_epoch,
                    total_epochs=total_epochs,
                    cfg=self.cfg,
                    current_round=current_round
                )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if 'avg_lambda' not in dir():
            avg_lambda = 0.0
        if 'gate_rate' not in dir():
            gate_rate = 0.0

        self.last_lambda = float(avg_lambda)
        self.last_gate = float(gate_rate)

        # Log training
        log_results(
            f"logs/{self.cfg.distill_mode}_train.csv",
            [
                self.cfg.distill_mode,
                self.current_round,
                float(train_loss),
                None, None, None, None, None,
                float(avg_lambda),
                float(gate_rate)
            ],
            HEADER
        )

        return self.get_parameters(config), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "lambda": float(avg_lambda),
            "gate_rate": float(gate_rate)
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        f1, precision, recall, auprc, auroc = evaluate(
            self.model, self.val_loader, self.device, self.cfg
        )

        loss = 1.0 - f1
        round_num = config.get("server_round", self.current_round)

        # Log evaluation
        log_results(
            f"logs/{self.cfg.distill_mode}_eval.csv",
            [
                self.cfg.distill_mode,
                round_num,
                loss,
                f1,
                precision,
                recall,
                auprc,
                auroc,
                None,
                None
            ],
            HEADER
        )

        return loss, len(self.val_loader.dataset), {
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "auprc": float(auprc),
            "auroc": float(auroc),
        }