# darshana-temporal-analysis

**Three computational studies of Indian philosophical traditions using the darshana-graph.**

This repository contains the code, data, and experiments supporting the paper:

> **Attribution Bias in Philosophical Knowledge Graphs: Corpus Frequency versus Temporal Sourcing**

The paper argues that corpus-frequency attribution systematically misattributes philosophical concepts to later, textually dominant schools, demonstrates this using the [darshana-graph](https://huggingface.co/datasets/joyboseroy/darshana-graph) (28,322 documented relationships across Hindu, Buddhist, and Jain traditions), and shows that temporal attribution enables a new class of cross-tradition structural analysis.

---

## Studies in this repository

### Study 1: Attribution Bias and Temporal Sourcing (`temporal/`)

Corpus-frequency attribution assigns each concept to the school that mentions it most in the available text. We show this conflates textual power, historical priority, and philosophical significance — three different things.

**Key findings:**
- Seven of the top 25 concepts by betweenness centrality predate their attributed school by 288 to 2,288 years
- Moksha, attributed to Advaita Vedanta, appears first in Jain sources over 1,200 years earlier
- The network at 300 BCE (17 Tier-1 concepts with explicit citations) is 59% Vedic, 24% Jain, 18% Buddhist — not the Advaita-dominated picture corpus frequency produces
- Between 300 CE and 800 CE the network grows from 18 to 1,028 nodes, with 97.4% carrying Advaita proxy dates — a corpus composition artifact, not historical reality

**Scripts:**
```bash
cd temporal/

# Temporal mismatch analysis: which concepts predate their attributed school?
python3 temporal_analysis.py
python3 temporal_analysis.py --concept moksha
python3 temporal_analysis.py --era -300   # network at 300 BCE

# Era snapshots with betweenness centrality
python3 integrate_temporal_v2.py
python3 integrate_temporal_v2.py --era_betweenness
python3 integrate_temporal_v2.py --era 800
```

**Data:** `temporal/temporal_source_layer.json` — 38 scholarly-cited sources, ~120 concept-datings with full citations covering Vedic through modern periods.

---

### Study 2: Structural Homology Across Traditions (`temporal/`)

Once concepts are temporally attributed, cross-tradition structural comparison becomes possible. We compute ego-network feature vectors for 48 temporally labelled concepts across eight traditions and identify concept pairs with high structural similarity.

**Key findings:**
- Recovery rate 8/14 known scholarly correspondences (57%), including purusha-jiva (Samkhya/Jain, sim 0.990) and prakriti-maya (Samkhya/Vedic, sim 0.972)
- Novel homologies: nibbana and samsara score 0.954 despite doctrinal opposition — both are the ultimate reference concept in their tradition's soteriology
- Cetana (Buddhist intention) and ajiva (Jain non-living matter) score 0.923 — a pairing absent from the scholarly literature, identified computationally
- Moha (Buddhist delusion) and tapas (Jain austerity) score 0.959 — mirror-image positions in the soteriological network

**Scripts:**
```bash
cd temporal/

# Install dependency
pip install scikit-learn --break-system-packages

# Run structural homology experiment
python3 structural_homology.py --top_n 50
python3 structural_homology.py --top_n 50 --min_degree 5  # stricter filter
```

**Data:** `temporal/homologues_v7.json` — top-200 cross-tradition structural homologue pairs with similarity scores and tradition labels.

---

### Study 3: Diffusion Simulation and Path-Finding (`notebooks/`)

How far does a new idea travel if seeded at a structural hub? Which historical figures were the real bridges between traditions?

**Key findings:**
- Centrality-based seeding reaches 62.6% adoption vs 58.4% for random seeding after 25 steps
- Brahman seeds produce a classic S-curve: tipping point at step 9, plateau at 62%
- Ramakrishna is the most central bridge figure: one step from nirvana, fana, and brahman simultaneously
- Shortest paths: sunyata to krsna consciousness (2 steps via maya), anatta to atman (2 steps via Ramana Maharshi), fana to krsna consciousness (3 steps via Ramakrishna)

**Scripts:**
```bash
# Install dependencies
pip install networkx matplotlib datasets --break-system-packages

# Betweenness centrality and seeding strategies
python3 notebooks/darshana_diffusion_v2.py --top_concepts
python3 notebooks/darshana_diffusion_v2.py --seed_concept brahman --steps 30
python3 notebooks/darshana_diffusion_v2.py --compare --steps 25

# Path-finding through historical figures
python3 notebooks/darshana_transmission_viz.py --from sunyata --to "krsna consciousness" --animate
python3 notebooks/darshana_transmission_viz.py --from anatta --to atman --animate
python3 notebooks/darshana_transmission_viz.py --show_figure Ramakrishna
python3 notebooks/darshana_transmission_viz.py --all_paths
```

**Data:** `transmission_layer.json` — 24 historical figures (Nagarjuna to Chogyam Trungpa), 155 hand-curated transmission edges.

---

## Repository structure

```
agentic-diffusion-sim/
├── temporal/                          # Studies 1 and 2
│   ├── temporal_source_layer.json     # 38 sources, ~120 concept-datings
│   ├── temporal_analysis.py           # Mismatch analysis and era snapshots
│   ├── integrate_temporal_v2.py       # Three-tier era betweenness analysis
│   ├── structural_homology.py         # Cross-tradition structural homologue experiment
│   └── homologues_v7.json             # Top-200 cross-tradition homologue pairs
├── notebooks/                         # Study 3
│   ├── darshana_diffusion_v2.py       # Darshana diffusion simulation
│   ├── darshana_path_viz.py           # Path finder on classical graph
│   ├── darshana_transmission_viz.py   # Two-layer path finder and animator
│   └── darshana_diffusion_standalone.py
├── agents/
│   └── belief_agent.py                # Four-stage belief update agent
├── network/
│   └── generator.py                   # Graph generation and centrality targeting
├── simulation/
│   └── diffusion.py                   # Simulation loop
├── visualization/
│   └── animate.py                     # Adoption curves, stage flow, network plots
├── transmission_layer.json            # 24 historical figures, 155 edges
├── run_sim.py                         # Generic diffusion simulation entry point
└── examples/                          # Sample outputs
```

---

## Data sources

| Dataset | Location | Description |
|---|---|---|
| darshana-graph | [HuggingFace](https://huggingface.co/datasets/joyboseroy/darshana-graph) | 28,322 typed relationship edges, 2,349 concept nodes |
| temporal_source_layer.json | `temporal/` | 38 scholarly sources, ~120 concept-datings with full citations |
| transmission_layer.json | repo root | 24 historical figures, 155 typed transmission edges |
| homologues_v7.json | `temporal/` | Top-200 cross-tradition structural homologue pairs |

The temporal source layer covers the period from the Rigveda (c.1500 BCE) through the 20th century. Every dating is accompanied by a scholarly citation. The layer includes Pali-Sanskrit equivalents (nibbana/nirvana, kamma/karma, dhamma/dharma, panna/prajna) and practice vocabulary drawn from a review by a practitioner teacher (nirodha, nivarana, asrava, bodhyanga, dhyana, maitri, karuna, vedana, cetana).

---

## Related projects

| Project | Connection |
|---|---|
| [darshana-graph](https://huggingface.co/datasets/joyboseroy/darshana-graph) | Source dataset (arXiv:2606.18222) |
| [vada-simulator](https://github.com/joyboseroy/vada-simulator) | Multi-agent philosophical debate engine |
| [bengal-dharma-corpus](https://github.com/joyboseroy/bengal-dharma-corpus) | Computational study of Bengali religious lexicon |
| [digital-buddhism](https://github.com/joyboseroy/digital-buddhism) | Buddhist teacher network analysis on Reddit |
| [falkor-irac](https://github.com/joyboseroy/falkor-irac) | Graph-RAG legal reasoning |

---

## Citation

```bibtex
@article{bose2026attribution,
  author = {Bose, Joy},
  title = {Attribution Bias in Philosophical Knowledge Graphs:
           Corpus Frequency versus Temporal Sourcing},
  year = {2026}
}

@software{bose2026agentic,
  author = {Bose, Joy},
  title = {darshana-temporal-analysis: Computational Studies of
           Indian Philosophical Traditions},
  year = {2026},
  url = {https://github.com/joyboseroy/darshana-temporal-analysis}
}

@dataset{bose2025darshana,
  author = {Bose, Joy},
  title = {darshana-graph: A Knowledge Graph of Indian Philosophy},
  year = {2025},
  url = {https://huggingface.co/datasets/joyboseroy/darshana-graph}
}
```

---

## Medium article

[How Ideas Travel Across Religions: A Computational Journey Through 2,000 Years of Indian Philosophy](https://joyboseroy.medium.com/how-ideas-travel-across-religions-a-computational-journey-through-2000-years-of-indian-philosophy-ce91b089e888)
