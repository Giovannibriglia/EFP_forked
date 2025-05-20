from __future__ import annotations

import argparse
import os

import networkx as nx
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import global_mean_pool, RGCNConv
from torch_geometric.data import Data, Batch
from utils import build_sample, predict_single, load_nx_graph


def nx_to_data(
    G,
    node_vocab: dict[str, int],
    edge_vocab: dict[str, int],
    node_attr: str = "label",
    edge_attr: str = "label",
) -> Data:
    """
    • Works even when nodes have *no* 'label' attribute (like your DOT files).
    • Relabels nodes to 0…N-1 so PyG doesn’t choke on 64-bit IDs.
    """
    # 1 ▷ make node ids contiguous
    G = nx.convert_node_labels_to_integers(
        G, label_attribute="orig_id"
    )  # keeps a copy of original id

    # 2 ▷ tokenise nodes
    for _, d in G.nodes(data=True):
        raw = d.get(node_attr, str(d["orig_id"]))  # fallback to original id
        d["token_id"] = node_vocab.setdefault(raw, len(node_vocab))

    # 3 ▷ tokenise edges
    edge_index, edge_type = [], []
    for u, v, d in G.edges(data=True):
        rel = d.get(edge_attr, "UNK_EDGE")
        rel_id = edge_vocab.setdefault(rel, len(edge_vocab))
        edge_index.append([u, v])
        edge_type.append(rel_id)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    x = torch.tensor([d["token_id"] for _, d in G.nodes(data=True)], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_type=edge_type)


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_edges: int,
        emb_dim: int = 64,
        hidden: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        self.rel_emb = nn.Embedding(num_edges, emb_dim)  # for basis-decomposition trick

        self.convs = nn.ModuleList()
        in_dim = emb_dim
        for _ in range(num_layers):
            self.convs.append(
                RGCNConv(
                    in_dim,
                    hidden,
                    num_relations=num_edges,
                    num_bases=min(30, num_edges),
                )
            )
            in_dim = hidden

    def forward(self, data: Data | Batch):
        x = self.node_emb(data.x)
        for conv in self.convs:
            x = F.relu(conv(x, data.edge_index, data.edge_type))
        return global_mean_pool(x, data.batch)  # [B, hidden]


class HeuristicModel(nn.Module):
    def __init__(
        self,
        vocab_state_nodes: int,
        vocab_state_edges: int,
        vocab_goal_nodes: int,
        vocab_goal_edges: int,
        emb_dim=64,
        hidden=128,
        cross_dim=32,
        depth_mode="scalar+emb",
        vocabulary_size: int = 512,
        embedding_dim: int = 8,
    ):  # "scalar", "emb", "scalar+emb"
        super().__init__()

        self.enc_state = GraphEncoder(
            vocab_state_nodes, vocab_state_edges, emb_dim, hidden
        )
        self.enc_goal = GraphEncoder(
            vocab_goal_nodes, vocab_goal_edges, emb_dim, hidden
        )

        self.cross = nn.Bilinear(hidden, hidden, cross_dim)

        self.depth_mode = depth_mode
        if "emb" in depth_mode:
            self.vocabulary_size = vocabulary_size
            self.embedding_dim = embedding_dim
            self.depth_emb = nn.Embedding(
                vocabulary_size, embedding_dim
            )  # expand if deeper trees

        in_dim = hidden + hidden + cross_dim
        if depth_mode == "scalar":
            in_dim += 1
        elif depth_mode == "emb":
            in_dim += self.embedding_dim
        elif depth_mode == "scalar+emb":
            in_dim += 1 + self.embedding_dim

        self.head = nn.Sequential(nn.Linear(in_dim, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, data_s, data_g, depth):
        hs = self.enc_state(data_s)  # [B, hidden]
        hg = self.enc_goal(data_g)  # [B, hidden]
        cross = self.cross(hs, hg)  # [B, cross_dim]

        depth_feats = []
        if "scalar" in self.depth_mode:
            depth_feats.append(depth.float().unsqueeze(-1))  # [B,1]
        if "emb" in self.depth_mode:
            depth_feats.append(
                self.depth_emb(depth.clamp_max(self.vocabulary_size - 1))
            )  # [B,8]

        z = torch.cat([hs, hg, cross, *depth_feats], dim=-1)
        return self.head(z).squeeze(-1)  # [B]


def _make_model(vocabs):
    return HeuristicModel(
        vocab_state_nodes=len(vocabs["state_node_vocab"]),
        vocab_state_edges=len(vocabs["state_edge_vocab"]),
        vocab_goal_nodes=len(vocabs["goal_node_vocab"]),
        vocab_goal_edges=len(vocabs["goal_edge_vocab"]),
        depth_mode="scalar+emb",
    )


def main_prediction() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(
        description="Run GNN heuristic prediction on a DOT graph."
    )
    parser.add_argument("path", type=str, help="Path (long) to graph file")
    parser.add_argument("depth", type=int, help="Depth parameter")
    parser.add_argument("n_agents", type=int, help="Number of agents")
    parser.add_argument(
        "goal_file",
        type=str,
        help="The file containing the goal description in graph format",
    )
    parser.add_argument("state_repr", type=str, help="State representation: M/H")
    args = parser.parse_args()

    USE_GOAL = True

    state_repr = args.state_repr
    USE_HASH = True if state_repr == "H" else False

    s_dot_path = args.path
    G_s = load_nx_graph(s_dot_path, False)

    subdir = "new_" + s_dot_path.split("out/state/")[1].split("/")[0]
    model_dir = os.path.join("lib", "RL", "results", subdir)

    if USE_HASH:
        model_dir += "_hashing"
    else:
        model_dir += "_mapping"

    if USE_GOAL:
        model_dir += "_goal"
    else:
        model_dir += "_nogoal"

    if USE_GOAL:
        g_dot_path = model_dir + args.goal_file.split("out/ML_HEUR_datasets/DFS")[1]
        G_g = load_nx_graph(g_dot_path, True)
    else:
        G_g = None

    path_vocabs = os.path.join(model_dir, "vocabularies.pt")
    vocabs = torch.load(path_vocabs)
    state_dict_path = os.path.join(model_dir, "heuristic_model_state.pt")

    model = _make_model(vocabs)
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.to(device)
    model.eval()

    # build your Data objects
    data_s = nx_to_data(G_s, vocabs["state_node_vocab"], vocabs["state_edge_vocab"])
    data_g = nx_to_data(G_g, vocabs["goal_node_vocab"], vocabs["goal_edge_vocab"])
    depth = torch.tensor([args.depth], dtype=torch.long)

    # batch and move to device
    batch_s = Batch.from_data_list([data_s]).to(device)
    batch_g = Batch.from_data_list([data_g]).to(device)
    depth = depth.to(device)

    # now everything lines up: all tensors + model on cuda
    pred = int(model(batch_s, batch_g, depth).item())

    # 5) Output
    with open("prediction.tmp", "w", encoding="utf-8") as fp:
        fp.write(f"VALUE:{pred}\n")


if __name__ == "__main__":
    main_prediction()
