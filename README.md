# Coralai: Emergent Ecosystems of Evolved Neural Cellular Automata

Coralai is a research framework for studying emergent collective behavior in evolving neural cellular automata. Each cell in a 2D toroidal grid runs a NEAT-evolved neural network, competing for energy and territory. Selection is purely spatial — cells that can't hold energy die, and successful strategies spread through directional colonization and in-place mutation. No fitness function. No explicit generations.

Builds on **EINCASM** (ALIFE 2023). Active development resumed 2026.

**Key design:**
- GPU-accelerated via [Taichi Lang](https://docs.taichi-lang.org/) + PyTorch (Mac Metal, CUDA, CPU)
- Continuous spatial NEAT — mutation and crossover happen as in-place radiation events
- Physics-as-experiment: the physics pipeline *is* the research question
- Pluggable experiments — each defines its own channel layout, physics, and NEAT config

---

## Setup

```bash
./setup.sh          # auto-detects conda or venv
conda activate coralai    # if conda
source .venv/bin/activate # if venv
```

Requires Python 3.10+. See `AGENTS.md` for full setup details and known issues.

---

## Running

```bash
# Interactive GUI (Mac Metal):
python experiments/coral/run.py

# Headless — N steps, print profiling breakdown:
python experiments/coral/run.py --no-gui --steps 1000 --profile

# Benchmark (seeded, worst-case throughput):
python experiments/coral/run.py --benchmark --steps 200 --shape 400

# Smaller grid for quick iteration:
python experiments/coral/run.py --shape 100 --steps 5000

# CPU / headless Linux:
python experiments/coral/run.py --backend cpu --device cpu --no-gui --steps 100

# Headless REPL (video output, JSON log):
python headless_repl.py --experiment coral --shape 64 --auto 500
```

---

## Structure

```
coralai/                    ← repo root
│
├── experiments/            ← create and run experiments here
│   ├── _template/          ← copy this to start a new experiment
│   ├── coral/              ← active main experiment
│   │   ├── run.py          ← entry point
│   │   ├── physics.py      ← physics pipeline (the experiment definition)
│   │   └── neat.config     ← NEAT hyperparameters
│   └── minimal/
│       └── neat.config
│
├── coralai/                ← Python package (the engine — don't edit per-experiment)
│   ├── substrate.py        ← world memory manager
│   ├── evolver.py          ← SpaceEvolver + apply_weights_and_biases kernel
│   ├── visualization.py    ← Taichi GGUI real-time renderer
│   ├── checkpointer.py     ← NEAT checkpoint reporter (stub — wiring TBD)
│   ├── channel.py
│   ├── nn_lib.py
│   ├── substrate_index.py
│   └── utils/
│       └── ti_struct_factory.py
│
├── analysis/               ← post-hoc tools for understanding saved runs
│   ├── multiscale_complexity.py   ← RG-flow complexity metrics
│   └── population_analysis.py    ← SAD, genome distance, KNN network
│
├── headless_repl.py        ← cross-platform headless runner + REPL
├── pyproject.toml          ← canonical dependency/package config
├── Makefile
├── setup.sh
│
├── runs/                   ← gitignored — written by experiment runs
├── logs/                   ← session notes (human + AI)
└── archive/                ← frozen history
```

**To create a new experiment:** copy `experiments/_template/`, define your channels in `run.py` and your physics in `physics.py`. The engine (`coralai/`) stays unchanged.

---

## Architecture notes

For a full write-up of the architecture, design issues, and research direction see:
- `logs/2026-02-26 Thu/2026-02-26 Thu 19.21 coralai deep dive.md` — full system overview
- `logs/2026-02-26 Thu/2026-02-26 Thu 20.17 energy dynamics exploration and architecture.md` — physics analysis
- `logs/2026-02-26 Thu/2026-02-26 Thu 22.09 session capstone and next steps.md` — current state and priorities

---

## EINCASM

Predecessor to CoralAI, published at ALIFE 2023 in Sapporo, Japan. Featured muscles, ports, capital mining, and waste physics. Design preserved in `archive/instances/eincasm/`.

- **Paper:** https://direct.mit.edu/isal/proceedings/isal/35/82/116945
- **Talk:** https://www.youtube.com/watch?v=RuLQRgi6YSU&t=514s

---

aidanbx@gmail.com
