import flwr as fl
from config import Config

cfg = Config()


# Aggregate metrics across clients
def weighted_average(metrics):
    total = sum([num for num, _ in metrics])

    f1        = sum([num * m["f1"]        for num, m in metrics]) / total
    precision = sum([num * m["precision"] for num, m in metrics]) / total
    recall    = sum([num * m["recall"]    for num, m in metrics]) / total
    auprc     = sum([num * m["auprc"]     for num, m in metrics]) / total
    auroc     = sum([num * m["auroc"]     for num, m in metrics]) / total

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "auprc": auprc,
        "auroc": auroc
    }


# Pass round info to clients
def fit_config(server_round: int):
    return {"server_round": server_round}


def evaluate_config(server_round: int):
    return {"server_round": server_round}


def start_server():
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=cfg.num_clients,
        min_available_clients=cfg.num_clients,
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()