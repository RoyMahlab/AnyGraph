import torch
from itertools import cycle
import numpy as np
import pickle
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as PygDataLoader
from torch_geometric.utils import subgraph
import random


class SubgraphDataset(torch.utils.data.Dataset):
    def __init__(self, graph: Data, subgraph_size: int):
        self.graph = graph
        self._assert_graph()
        self.subgraph_size = subgraph_size
        self.subgraphs = self._sample_non_overlapping_subgraphs()

    def _assert_graph(self):
        assert (
            self.graph.x is not None
        ), "Node features are required for subgraph sampling"
        assert (
            self.graph.edge_index is not None
        ), "Edge index is required for subgraph sampling"
        assert (
            self.graph.edge_index.size(0) == 2
        ), "Edge index must be of shape (2, num_edges)"
        print("Graph assertiong passed")

    def _sample_non_overlapping_subgraphs(self):
        nodes = self.graph.num_nodes
        all_nodes = torch.arange(nodes)
        random.shuffle(all_nodes.tolist())  # Shuffle node indices for randomness
        subgraphs = []

        # Split into chunks of subgraph_size
        for i in range(0, len(all_nodes), self.subgraph_size):
            sub_nodes = all_nodes[i : i + self.subgraph_size]
            if len(sub_nodes) < self.subgraph_size:
                continue  # Skip if not enough nodes for a full subgraph
            edge_index, _ = subgraph(
                sub_nodes, self.graph.edge_index, relabel_nodes=True
            )
            if edge_index.size(1) == 0:
                continue
            subgraph_data = Data(
                x=self.graph.x[sub_nodes] if self.graph.x is not None else None,
                edge_index=edge_index,
            )
            subgraphs.append(subgraph_data)

        return subgraphs

    def __len__(self):
        return len(self.subgraphs)

    def __getitem__(self, idx):
        return self.subgraphs[idx]


def get_subgraph_dataloader(
    graph: Data, subgraph_size: int, batch_size: int, shuffle: bool
) -> PygDataLoader:
    subgraph_dataset = SubgraphDataset(graph, subgraph_size)
    subgraph_loader = cycle(
        PygDataLoader(
            subgraph_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
        )
    )
    return subgraph_loader


def convert_sparse_adj_matrix_to_edge_index(adj_matrix: torch.Tensor) -> torch.Tensor:
    row, col = adj_matrix.nonzero()  # Get the nonzero indices
    edge_index = torch.tensor(np.array([row, col]), dtype=torch.long)
    return edge_index


if __name__ == "__main__":
    with open("/home/roymahlab/projects/AnyGraph/datasets/arxiv/feats.pkl", "rb") as f:
        features = pickle.load(f)
    with open(
        "/home/roymahlab/projects/AnyGraph/datasets/arxiv/trn_mat.pkl", "rb"
    ) as f:
        adj_index = pickle.load(f)

    graph = Data(
        x=features, edge_index=convert_sparse_adj_matrix_to_edge_index(adj_index)
    )
    subgraph_size = graph.num_nodes // 100  # 1% of the nodes
    subgraph_dataset = SubgraphDataset(graph, subgraph_size)
    subgraph_loader = PygDataLoader(subgraph_dataset, batch_size=1, shuffle=True)

    # Iterate over the DataLoader
    for batch in subgraph_loader:
        print(batch)
