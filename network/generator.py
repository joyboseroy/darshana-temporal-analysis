"""
network/generator.py — Graph generation and centrality-based seed targeting.

Supports:
  - Erdos-Renyi G(n,p) — Krishnan's baseline
  - Barabasi-Albert (scale-free) — more realistic social networks
  - Small-world Watts-Strogatz

Centrality targeting strategies:
  - betweenness  (Krishnan Ch5 finding: best for diffusion)
  - percolation  (Krishnan Ch5 finding: also strong)
  - degree       (common baseline)
  - random       (null baseline)
"""

import networkx as nx
import random
from typing import Dict, List


def build_graph(
    num_nodes: int = 60,
    topology: str = "erdos_renyi",
    edge_prob: float = 0.12,
    ba_edges: int = 3,
    sw_neighbors: int = 4,
    sw_rewire: float = 0.1,
    seed: int = 42,
) -> nx.Graph:
    """Return an undirected NetworkX graph."""
    if topology == "erdos_renyi":
        G = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=seed)
    elif topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(num_nodes, ba_edges, seed=seed)
    elif topology == "small_world":
        G = nx.watts_strogatz_graph(num_nodes, sw_neighbors, sw_rewire, seed=seed)
    else:
        raise ValueError(f"Unknown topology: {topology}")

    # Ensure connectivity — attach isolates to a random node
    for node in list(nx.isolates(G)):
        target = random.choice([n for n in G.nodes() if n != node])
        G.add_edge(node, target)

    return G


def compute_centralities(G: nx.Graph) -> Dict[str, Dict[int, float]]:
    """Compute all centrality measures used in Krishnan Ch5."""
    return {
        "betweenness": nx.betweenness_centrality(G),
        "degree": nx.degree_centrality(G),
        "percolation": _percolation_centrality(G),
        "random": {n: random.random() for n in G.nodes()},
    }


def _percolation_centrality(G: nx.Graph) -> Dict[int, float]:
    """
    Percolation centrality approximation.
    Krishnan uses this as an alternative to betweenness for diffusion targeting.
    Full version weights betweenness paths by node percolation state.
    Here we approximate using a normalised combination of
    betweenness and clustering coefficient (light version for demo).
    """
    bc = nx.betweenness_centrality(G)
    cc = nx.clustering(G)
    result = {}
    for n in G.nodes():
        # Nodes with high betweenness but low clustering = good bridge nodes
        result[n] = bc[n] * (1 - cc.get(n, 0))
    # Normalise to [0,1]
    max_val = max(result.values()) if result else 1.0
    if max_val > 0:
        result = {k: v / max_val for k, v in result.items()}
    return result


def get_seed_nodes(
    centralities: Dict[str, Dict[int, float]],
    strategy: str,
    seed_fraction: float = 0.10,
    num_nodes: int = 60,
) -> List[int]:
    """Return top seed_fraction of nodes by chosen centrality strategy."""
    k = max(1, int(num_nodes * seed_fraction))
    scores = centralities.get(strategy, centralities["random"])
    return sorted(scores, key=scores.get, reverse=True)[:k]
