"""Graph Neural Network recommender for SpotiGraph.

Item-item link prediction with an inductive GraphSAGE encoder.

Pipeline (matches the user's mental model)
-------------------------------------------
1. A new song is a node, linked to Artist / Genre / Energy / Era nodes
   (done by the enrichment pipeline → graph_client.apply_enrichment).
2. GraphSAGE computes the node's embedding by aggregating its neighbours
   (inductive — a brand-new node gets an embedding without retraining).
3. Link prediction: probability of an edge between the currently-playing
   track and a candidate = sigmoid(emb_seed · emb_candidate). High → recommend.

Training is self-supervised: predict the graph's OWN edges (positive) vs.
random non-edges (negative). No user-interaction labels are required, which
is why this works with a single user.

If torch / torch_geometric are not installed, GNN_AVAILABLE is False and the
caller falls back to the heuristic spreading-activation scorer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import SAGEConv
    from torch_geometric.utils import negative_sampling
    GNN_AVAILABLE = True
except Exception as exc:  # pragma: no cover - depends on optional deps
    GNN_AVAILABLE = False
    logger.warning("GNN unavailable (torch/PyG import failed): %s", exc)

MODEL_PATH = Path(__file__).parent / "gnn_model.pt"
NODE_TYPES = ["Track", "Artist", "Genre", "Energy", "Era"]
_IN_DIM = len(NODE_TYPES) + 3   # one-hot(type) + [loudness, bpm, popularity]
_HIDDEN = 64
_EMB_DIM = 64
_TYPE_IDX = {t: i for i, t in enumerate(NODE_TYPES)}


# ---------------------------------------------------------------------------
# Feature construction (shared by train + score so dims always match)
# ---------------------------------------------------------------------------

def _build_features(nodes: list[dict]) -> tuple[np.ndarray, dict[str, int]]:
    """Return (feature matrix [N, _IN_DIM], id→row index)."""
    id_to_idx: dict[str, int] = {}
    feats = np.zeros((len(nodes), _IN_DIM), dtype=np.float32)
    for i, n in enumerate(nodes):
        id_to_idx[n["id"]] = i
        t = n.get("type")
        if t in _TYPE_IDX:
            feats[i, _TYPE_IDX[t]] = 1.0
        if t == "Track":
            # normalize: loudness dB (~ -20..0), bpm (~0..200), popularity (0..100)
            feats[i, len(NODE_TYPES) + 0] = (float(n.get("loudness") or 0.0)) / 20.0
            feats[i, len(NODE_TYPES) + 1] = (float(n.get("bpm") or 0.0)) / 200.0
            feats[i, len(NODE_TYPES) + 2] = (float(n.get("popularity") or 0)) / 100.0
    return feats, id_to_idx


def _build_edge_index(edges: list[dict], id_to_idx: dict[str, int]):
    """Undirected edge_index tensor [2, 2E] (both directions)."""
    src, dst = [], []
    for e in edges:
        a, b = id_to_idx.get(e["source"]), id_to_idx.get(e["target"])
        if a is None or b is None:
            continue
        src += [a, b]
        dst += [b, a]
    if not src:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

if GNN_AVAILABLE:

    class GraphSAGEModel(torch.nn.Module):
        def __init__(self, in_dim: int = _IN_DIM, hidden: int = _HIDDEN, out: int = _EMB_DIM):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, out)

        def forward(self, x, edge_index):
            h = F.relu(self.conv1(x, edge_index))
            return self.conv2(h, edge_index)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank-based ROC-AUC: P(score(pos) > score(neg)). No sklearn needed."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


class GnnScorer:
    """Thin wrapper used by the Recommender. Safe to construct even without torch."""

    def __init__(self) -> None:
        self.available = GNN_AVAILABLE

    def is_ready(self) -> bool:
        return self.available and MODEL_PATH.exists()

    # -- training -------------------------------------------------------

    def train(self, export: dict, epochs: int = 300, lr: float = 0.01) -> dict:
        if not self.available:
            return {"error": "torch/torch_geometric not installed"}

        nodes, edges = export["nodes"], export["edges"]
        if len(nodes) < 5 or len(edges) < 5:
            return {"error": f"graph too small (nodes={len(nodes)}, edges={len(edges)})"}

        feats, id_to_idx = _build_features(nodes)
        x = torch.tensor(feats)
        edge_index = _build_edge_index(edges, id_to_idx)
        num_nodes = x.size(0)

        # Hold out 10% of (undirected) edges for AUC evaluation.
        E = edge_index.size(1)
        perm = torch.randperm(E)
        n_val = max(1, E // 10)
        val_e = edge_index[:, perm[:n_val]]
        train_e = edge_index[:, perm[n_val:]]

        model = GraphSAGEModel()
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        last_loss = 0.0
        for _ in range(epochs):
            model.train()
            opt.zero_grad()
            z = model(x, train_e)
            pos = train_e
            neg = negative_sampling(train_e, num_nodes=num_nodes, num_neg_samples=pos.size(1))
            pos_s = (z[pos[0]] * z[pos[1]]).sum(-1)
            neg_s = (z[neg[0]] * z[neg[1]]).sum(-1)
            logits = torch.cat([pos_s, neg_s])
            labels = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s)])
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            opt.step()
            last_loss = float(loss.detach())

        # Evaluate AUC on held-out edges
        model.eval()
        with torch.no_grad():
            z = model(x, train_e)
            val_neg = negative_sampling(edge_index, num_nodes=num_nodes, num_neg_samples=val_e.size(1))
            pos_s = (z[val_e[0]] * z[val_e[1]]).sum(-1).numpy()
            neg_s = (z[val_neg[0]] * z[val_neg[1]]).sum(-1).numpy()
        auc = _auc(pos_s, neg_s)

        torch.save(model.state_dict(), MODEL_PATH)
        logger.info("GNN trained: nodes=%d edges=%d loss=%.4f auc=%.3f",
                    num_nodes, E, last_loss, auc)
        return {"nodes": num_nodes, "edges": E, "epochs": epochs,
                "loss": round(last_loss, 4), "heldout_auc": round(auc, 3)}

    # -- scoring --------------------------------------------------------

    def score(self, export: dict, seed_id: str, candidate_ids: list[str]) -> dict[str, float]:
        """Return {spotify_id: link probability} for candidates vs the seed."""
        if not self.is_ready():
            return {}
        nodes, edges = export["nodes"], export["edges"]
        feats, id_to_idx = _build_features(nodes)
        seed_key = f"t:{seed_id}"
        if seed_key not in id_to_idx:
            logger.info("GNN: seed not in graph yet — cannot score")
            return {}

        x = torch.tensor(feats)
        edge_index = _build_edge_index(edges, id_to_idx)

        model = GraphSAGEModel()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        with torch.no_grad():
            z = model(x, edge_index)              # inductive forward pass
            seed_vec = z[id_to_idx[seed_key]]
            scores = torch.sigmoid(z @ seed_vec)  # link prob seed↔every node

        out: dict[str, float] = {}
        for cid in candidate_ids:
            idx = id_to_idx.get(f"t:{cid}")
            if idx is not None:
                out[cid] = float(scores[idx])
        return out
