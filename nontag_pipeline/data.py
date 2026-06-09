from __future__ import annotations
import torch
import numpy as np
import networkx as nx
from torch_geometric.datasets import Planetoid

PUBMED_CLASSES: list[str] = [
    "Diabetes_Mellitus_Experimental",
    "Diabetes_Mellitus_Type_1",
    "Diabetes_Mellitus_Type_2",
]

# NOTE: GraphGPT uses a 70-class Cora variant; swap load_dataset("cora") source later.
CORA_CLASSES: list[str] = [
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
]

_CLASS_NAMES: dict[str, list[str]] = {
    "pubmed": PUBMED_CLASSES,
    "cora": CORA_CLASSES,
}


def load_dataset(
    name: str, seed: int = 42, root: str = "/tmp/planetoid"
) -> tuple[nx.Graph, np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Load Planetoid dataset, apply 60/20/20 seeded split, return undirected graph."""
    dataset = Planetoid(root=root, name=name.capitalize())
    data = dataset[0]
    n: int = data.num_nodes

    torch.manual_seed(seed)
    perm = torch.randperm(n)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[perm[:train_end]] = True
    val_mask[perm[train_end:val_end]] = True
    test_mask[perm[val_end:]] = True

    edge_index = data.edge_index.numpy()
    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    y = data.y.numpy()
    class_names = _CLASS_NAMES[name.lower()]
    return G, y, train_mask, val_mask, test_mask, class_names
