# Coralai: Emergent Ecosystems of Evolved Neural Cellular Automata

This repo contains the code for:
- **Coralai** (Master's Thesis 2024)
- **EINCASM** (ALIFE 2023) — see historical notes below

Coralai is a framework for simulating and studying emergent ecosystems in neural cellular collectives resembling slime molds. Each cell in a 2D toroidal grid runs the same NEAT-evolved neural network, competing for energy and space. Selection pressure is purely spatial — cells that can't accumulate enough energy die, and successful strategies spread through directional colonization and mutation.

**Key capabilities:**
- Evolves and visualizes Neural Cellular Automata (NCA) in real-time with custom physics and NEAT-evolved architectures
- GPU-accelerated on Mac Metal, CUDA, and CPU via [Taichi Lang](https://docs.taichi-lang.org/) + PyTorch
- Continuous spatial NEAT — no explicit generations; mutation and crossover happen in-place like radiation events
- Configurable channel layout, kernel shape, physics pipeline, and NEAT hyperparameters per experiment
- Headless REPL for cross-platform use (Linux/CI), outputs video and JSON logs

For a full architectural write-up see `logs/2026-02-26 Thu/2026-02-26 Thu 20.21 coralai deep dive.md`.

---

## Setup

Run all commands from the **repo root** (the directory containing `setup.sh`).

**Conda (recommended):**
```bash
./setup.sh conda
conda activate coralai
```

**Venv:**
```bash
./setup.sh venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows
```

`./setup.sh` auto-detects conda or venv if called without arguments. Requires Python 3.10+.

---

## Running

**GUI (Mac Metal only):**
```bash
python coral_runner_space.py    # full coral ecosystem — most advanced runner
python minimal_runner.py        # single-channel NCA
python nca_runner.py            # NCA with RGB + hidden channels
```

**Headless (any platform):**
```bash
make run-minimal    # 64×64 minimal NCA
make run-nca        # 48×48 NCA with RGB channels
make run-coral      # 16×16 coral ecosystem
make demo           # non-interactive 300-step demo, saves video to runs/
# or directly:
python headless_repl.py --experiment minimal --shape 64
python headless_repl.py --experiment coral   --shape 32 --auto 500
```

See `AGENTS.md` for full REPL commands, profiling instructions, and known issues.

---

## Code Organization

```
./                              ← repo root — runners live here
├── coral_runner_space.py       ← ACTIVE: continuous spatial NEAT ecosystem
├── minimal_runner.py           ← simple single-channel NCA (Mac only)
├── nca_runner.py               ← RGB NCA (Mac only)
├── headless_repl.py            ← cross-platform headless runner + REPL
├── xor_runner.py               ← NEAT XOR demo (sanity check)
│
└── coralai/                    ← main package
    ├── substrate/              ← world memory: Substrate, Channel, visualization
    ├── evolution/              ← evolvers and organism types
    │   ├── space_evolver.py    ← ACTIVE evolver (spatial NEAT)
    │   ├── neat_evolver.py     ← older explicit-generation evolver
    │   └── ecosystem.py        ← older per-organism sequential evolver
    ├── instances/              ← self-contained experiment configurations
    │   ├── coral/              ← ACTIVE instance (physics, NEAT config)
    │   ├── minimal/            ← single-channel NCA
    │   ├── nca/                ← multi-channel NCA
    │   ├── xor/                ← NEAT XOR demo
    │   └── eincasm/            ← historical (broken refs, not runnable)
    ├── dependencies/
    │   └── PyTorch-NEAT/       ← vendored (no submodule init needed)
    └── utils/
        └── ti_struct_factory.py
```

**Instances** are self-contained experiment configurations — each folder defines its own channel layout, physics, and NEAT config. To create a new experiment: add a subfolder under `instances/`, define your physics and NEAT config, and write a runner script in the repo root.

---

## EINCASM: Emergent Intelligence in Neural Cellular Automata Slime Molds

EINCASM is the predecessor to CoralAI, published at ALIFE 2023 in Sapporo, Japan. It featured a more elaborate biophysical model (muscles, ports, capital mining, waste) that has since been refactored. The `instances/eincasm/` folder preserves the design intent but references deleted modules and is not currently runnable.

- **Paper:** https://direct.mit.edu/isal/proceedings/isal/35/82/116945
- **Presentation recording:** https://www.youtube.com/watch?v=RuLQRgi6YSU&t=514s
- **Workshop — Machine Love and Human Flourishing:** https://www.youtube.com/watch?v=tfQhXOBchKY

---

## Contact

Feel free to reach out to aidanbx@gmail.com if you have questions or ideas.
