import numpy as np
from torch.utils.data import Subset

def split_dataset(dataset, num_clients, cfg):
    alpha = cfg.noniid_alpha

    labels = dataset.labels.numpy()
    num_classes = labels.shape[1]

    dominant_labels = (labels > 0).astype(int).argmax(axis=1)

    class_indices = {i: np.where(dominant_labels == i)[0] for i in range(num_classes)}

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idxs = class_indices[c]
        np.random.shuffle(idxs)

        proportions = np.random.dirichlet(np.ones(num_clients) * alpha)
        proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]

        split = np.split(idxs, proportions)

        for i in range(num_clients):
            client_indices[i].extend(split[i])

    return [Subset(dataset, idxs) for idxs in client_indices]