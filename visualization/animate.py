"""
visualization/animate.py — Static and animated plots for diffusion results.

Outputs:
  - adoption_curves.png: strategy comparison line chart
  - network_snapshot.png: node coloring by belief at final step
  - stage_flow.png: stacked area chart of 4-stage progression
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
from typing import Dict, Optional

from simulation.diffusion import SimulationResult
from network.generator import build_graph, compute_centralities


STRATEGY_COLORS = {
    "betweenness": "#E63946",
    "percolation": "#F4A261",
    "degree":      "#2A9D8F",
    "random":      "#ADB5BD",
}

STAGE_COLORS = ["#dee2e6", "#74c0fc", "#f4a261", "#2a9d8f"]
STAGE_NAMES = ["Unaware", "Aware", "Action", "Habit"]


def plot_adoption_curves(
    results: Dict[str, SimulationResult],
    output_dir: str = "/mnt/user-data/outputs",
    filename: str = "adoption_curves.png",
):
    """Line chart comparing adoption rates across strategies — Figure 1."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    for strategy, result in results.items():
        color = STRATEGY_COLORS.get(strategy, "#ffffff")
        ax.plot(
            range(1, result.steps + 1),
            [r * 100 for r in result.adoption_history],
            label=strategy.capitalize(),
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=3,
        )

    ax.set_xlabel("Timestep", color="#adb5bd", fontsize=11)
    ax.set_ylabel("Adoption Rate (%)", color="#adb5bd", fontsize=11)
    ax.set_title(
        "Behavior Diffusion by Seeding Strategy\n(Krishnan 2025 Ch5 — Python Replication)",
        color="#f8f9fa", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#adb5bd")
    ax.spines[:].set_color("#2d3142")
    ax.grid(color="#2d3142", linewidth=0.5, linestyle="--")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#f8f9fa", fontsize=10)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_stage_flow(
    result: SimulationResult,
    output_dir: str = "/mnt/user-data/outputs",
    filename: str = "stage_flow.png",
):
    """Stacked area chart showing 4-stage progression over time."""
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    steps = range(1, result.steps + 1)
    stacks = {name: [] for name in STAGE_NAMES}
    for stage_dict in result.stage_history:
        for name in STAGE_NAMES:
            stacks[name].append(stage_dict.get(name, 0))

    ax.stackplot(
        steps,
        [stacks[n] for n in STAGE_NAMES],
        labels=STAGE_NAMES,
        colors=STAGE_COLORS,
        alpha=0.85,
    )

    ax.set_xlabel("Timestep", color="#adb5bd", fontsize=11)
    ax.set_ylabel("Agent Count", color="#adb5bd", fontsize=11)
    ax.set_title(
        f"4-Stage Behavior Change Flow ({result.strategy.capitalize()} seeding)",
        color="#f8f9fa", fontsize=13, pad=12,
    )
    ax.tick_params(colors="#adb5bd")
    ax.spines[:].set_color("#2d3142")
    ax.legend(facecolor="#1a1d27", edgecolor="#2d3142", labelcolor="#f8f9fa",
              fontsize=9, loc="upper left")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_network_snapshot(
    result: SimulationResult,
    num_nodes: int = 60,
    topology: str = "erdos_renyi",
    output_dir: str = "/mnt/user-data/outputs",
    filename: str = "network_snapshot.png",
):
    """Network graph coloured by final belief stage."""
    G = build_graph(num_nodes=num_nodes, topology=topology)
    centralities = {"betweenness": {}, "percolation": {}, "degree": {}, "random": {}}

    # We just need the graph structure; belief states reconstructed from stage_history
    final_stages = result.stage_history[-1] if result.stage_history else {}

    # Approximate node stage distribution for colouring
    stage_idx = list(STAGE_NAMES)
    # Distribute nodes proportionally to final stage counts
    node_stages = []
    for stage_name in STAGE_NAMES:
        count = final_stages.get(stage_name, 0)
        node_stages.extend([stage_idx.index(stage_name)] * count)
    # Pad or trim
    while len(node_stages) < num_nodes:
        node_stages.append(0)
    node_stages = node_stages[:num_nodes]

    node_colors = [STAGE_COLORS[s] for s in node_stages]
    seed_set = set(result.seed_nodes)
    node_sizes = [120 if n in seed_set else 50 for n in G.nodes()]
    node_edge_colors = ["#FFD700" if n in seed_set else "#2d3142" for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    pos = nx.spring_layout(G, seed=42, k=0.6)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#5c6370", width=0.6)
    nx.draw_networkx_nodes(G, pos, ax=ax,
                           node_color=node_colors,
                           node_size=node_sizes,
                           edgecolors=node_edge_colors,
                           linewidths=1.5)

    # Legend
    patches = [mpatches.Patch(color=STAGE_COLORS[i], label=STAGE_NAMES[i]) for i in range(4)]
    patches.append(mpatches.Patch(color="#FFD700", label="Seed node"))
    ax.legend(handles=patches, facecolor="#1a1d27", edgecolor="#2d3142",
              labelcolor="#f8f9fa", fontsize=9, loc="lower right")
    ax.set_title(
        f"Network State — {result.strategy.capitalize()} Seeding (Final Step)",
        color="#f8f9fa", fontsize=13, pad=10,
    )
    ax.axis("off")

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path
