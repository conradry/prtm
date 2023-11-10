from dataclasses import dataclass


@dataclass
class PiFoldConfig:
    hidden_dim: int = 128
    node_features: int = 128
    edge_features: int = 128
    k_neighbors: int = 30
    dropout: float = 0.1
    num_encoder_layers: int = 10
    updating_edges: int = 4
    node_dist: int = 1
    node_angle: int = 1
    node_direct: int = 1
    edge_dist: int = 1
    edge_angle: int = 1
    edge_direct: int = 1
    virtual_num: int = 3
