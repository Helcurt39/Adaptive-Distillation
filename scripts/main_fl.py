import subprocess
import os
import flwr as fl
import torch
import sys

from collate import get_collate_fn
from config import get_config
from dataset import ChestXrayDataset
from fl_utils import split_dataset
from fl_client import FLClient
from torch.utils.data import DataLoader


# Experiment modes
MODES = [
    "baseline", "static", "mse", "kl",
    "loss_adaptive", "loss_adaptive_kl",
    "confidence", "confidence_kl", "hybrid"
]


# Aggregate metrics
def weighted_average(metrics):
    total = sum([num for num, _ in metrics])

    f1 = sum([num * m["f1"] for num, m in metrics]) / total
    precision = sum([num * m["precision"] for num, m in metrics]) / total
    recall = sum([num * m["recall"] for num, m in metrics]) / total

    avg_lambda = sum([num * m.get("lambda", 0) for num, m in metrics]) / total
    avg_gate = sum([num * m.get("gate_rate", 0) for num, m in metrics]) / total

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "lambda": avg_lambda,
        "gate_rate": avg_gate,
    }


# Pass round info
def fit_config(server_round: int):
    return {"server_round": server_round}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


# Select GPU with least usage
def get_free_gpu():
    result = subprocess.check_output(
        "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader",
        shell=True
    )
    gpu_memory = [int(x) for x in result.decode().strip().split("\n")]
    best_gpu = gpu_memory.index(min(gpu_memory))

    print(f"Using GPU {best_gpu}")
    return best_gpu


def run_experiment(cfg):
    gpu_id = get_free_gpu()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = ChestXrayDataset(cfg.train_path)
    val_dataset = ChestXrayDataset(cfg.val_path)
    client_datasets = split_dataset(dataset, cfg.num_clients, cfg)

    def client_fn(cid: str):
        cid = int(cid)
        collate_fn = get_collate_fn(cfg) if cfg.use_text else None

        train_loader = DataLoader(
            client_datasets[cid],
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory
        )

        return FLClient(train_loader, val_loader, cfg, device, cid)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.25},
        ray_init_args={"include_dashboard": False, "num_cpus": 4, "num_gpus": 1}
    )


def main():
    modes = [sys.argv[1]] if len(sys.argv) > 1 else MODES

    for mode in modes:
        print("\n" + "=" * 60)
        print(f"RUNNING EXPERIMENT: {mode}")
        print("=" * 60 + "\n")

        cfg = get_config(mode)
        run_experiment(cfg)

    print("\nALL EXPERIMENTS COMPLETED!\n")


if __name__ == "__main__":
    main()