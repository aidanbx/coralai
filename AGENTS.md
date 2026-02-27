# AGENTS.md

---

## Logs and Output Convention

All analysis, design thinking, and substantial AI responses belong in the **logs directory** at `logs/` inside this repo.

### Directory structure

```
logs/
├── YY-MM-DD ddd - Session Title.md         ← human session log (one per day, append-only)
└── YYYY-MM-DD ddd/                         ← AI output directory for that day
    └── YYYY-MM-DD ddd HH.MM descriptive title.md
```

### Human session log (`YY-MM-DD ddd - Title.md`)
- One file per day, named with short year + 3-letter weekday: `26-02-26 Thu - Title.md`
- Title represents the full day's content; update it as the day progresses
- Sections separated by `---`, ordered chronologically
- Captures questions, decisions, and directions to explore

### AI output files (`YYYY-MM-DD ddd/YYYY-MM-DD ddd HH.MM descriptive title.md`)
- Written to the dated subdirectory: `YYYY-MM-DD ddd/`
- Filename: full ISO 8601 date + ddd + `HH.MM` time + space-separated descriptive title
- Note: use `HH.MM` (period separator) not `HH:MM`; colons are invalid in filenames on macOS/Windows
- Example: `2026-02-26 Thu/2026-02-26 Thu 14.30 energy dynamics and architecture.md`
- Write a new file for any substantial analysis, design exploration, or technical write-up

---

## Project Overview

CoralAI is a Python research framework for simulating emergent ecosystems of evolved Neural Cellular Automata (NCA). It uses PyTorch for tensor operations, Taichi Lang for GPU-parallel kernels (Mac Metal, CUDA, or CPU), and NEAT-Python for neuroevolution. It was developed as a Master's Thesis (2024), building on EINCASM (ALIFE 2023).

See `logs/2026-02-26 Thu/2026-02-26 Thu 20.21 coralai deep dive.md` for a full architectural write-up.

**Active runner**: `coral_runner_space.py` - continuous spatial NEAT, no explicit generations.
**Active instance**: `coralai/instances/coral/` - energy/infra economy with directional colonization.
**Active physics**: `coralai/instances/coral/coral_physics.py`.
**Active evolver**: `coralai/evolution/space_evolver.py`.

---

## Setup

Run all commands from the **repo root** (the directory containing `setup.sh` and `Makefile`).

### One-command install (auto-detects conda or venv)
```bash
./setup.sh            # auto
./setup.sh conda      # force conda
./setup.sh venv       # force venv
```

- **Conda** (recommended): creates/updates the `coralai` env from `environment.yml`
- **Venv**: requires Python 3.10+, installs from `requirements.txt`, installs the package with `pip install -e .`

PyTorch-NEAT is vendored at `coralai/dependencies/PyTorch-NEAT/` — no submodule init needed. It is installed as a local editable package via the `-e coralai/dependencies/PyTorch-NEAT` line in `requirements.txt`.

### Activate
```bash
conda activate coralai          # conda
source .venv/bin/activate       # venv (Linux/macOS)
.venv\Scripts\activate          # venv (Windows)
```

### Make targets
```bash
make help           # show all targets
make setup          # run setup.sh
make run-minimal    # headless REPL, minimal 1-channel NCA, 64×64
make run-nca        # headless REPL, NCA with RGB + hidden channels, 48×48
make run-coral      # headless REPL, coral ecosystem, 16×16
make run-xor        # XOR NEAT evolution (no GUI)
make demo           # non-interactive 300-step demo, saves video to runs/
make test           # run PyTorch-NEAT tests
make lint           # flake8 critical errors only
make clean          # remove runs/, history/, __pycache__, etc.
```

---

## Running

### GUI runners (Mac only — requires Metal)
```bash
python coral_runner_space.py    # full coral ecosystem, 400×400, Mac Metal
python minimal_runner.py        # single-channel NCA, Mac Metal
python nca_runner.py            # RGB NCA, Mac Metal
```
These use Taichi GGUI and are hardcoded to `ti.metal` + `torch.device("mps")`. Not portable.

### Headless REPL (cross-platform)
```bash
python headless_repl.py --experiment minimal --shape 64
python headless_repl.py --experiment nca     --shape 48
python headless_repl.py --experiment coral   --shape 16
```

Options:
- `--shape N` — grid side length (N×N)
- `--steps-per-frame N` — simulation steps between rendered frames (default 1)
- `--max-frames N` — cap on saved frames (default 2000)
- `--auto N` — run N steps non-interactively, save video, exit
- `--fps N` — output video frame rate (default 30)

REPL commands (interactive mode):
```
step [N]           advance N steps (default 10), prints steps/s
mutate [strength]  perturb organism weights
status             print per-channel statistics (mean/std/min/max/sum)
channels R G B     set which channels map to RGB display (by name or index)
paint X Y VAL CH   add VAL to substrate.mem[0, CH, X, Y]
clear              zero all substrate memory
save_frame         save current frame as PNG to runs/<run>/frames/
snapshot           save substrate.mem tensor to runs/<run>/substrate_step<N>.pt
quit / exit        compile video, write log.json, exit
```

Output is written to `runs/<experiment>_<timestamp>/`:
- `simulation.mp4` — compiled video
- `log.json` — per-step stats and event log
- `frames/` — individual PNG frames

---

## Profiling and Benchmarking

### Steps per second
The headless REPL prints `steps/s` after every `step [N]` command and after `--auto` runs:
```bash
python headless_repl.py --experiment minimal --shape 64 --auto 1000
# Prints: "1000 steps in 4.23s (236 steps/s)"
```

Use `--steps-per-frame 10` or higher to reduce rendering overhead and get pure sim throughput:
```bash
python headless_repl.py --experiment minimal --shape 128 --auto 500 --steps-per-frame 10
```

### Comparing backends
The headless REPL uses `ti.cpu` and `torch.device("cpu")`. To benchmark Metal or CUDA, run the GUI runner or modify the device init at the top of `headless_repl.py`:
```python
ti.init(ti.metal)                    # Mac
ti.init(ti.cuda)                     # NVIDIA
TORCH_DEVICE = torch.device("mps")   # Mac
TORCH_DEVICE = torch.device("cuda")  # NVIDIA
```

### Substrate memory scaling
Grid size scales as O(W×H×C) where C is total channel count. For coral: C=15 channels, 400×400 grid ≈ 2.4M floats ≈ 9.6MB per substrate tensor. The stacked weight tensor for SpaceEvolver scales as O(n_genomes × n_acts × n_inputs) — with 100 genomes, n_acts=10, n_inputs=12 this is ~12K floats, negligible.

---

## Known Issues and Vestigial Code

### Broken / not runnable
- `coralai/instances/coral/coral_evolver.py` — empty stub, imports a deleted `simulation.evolver` module
- `coralai/instances/coral/dumb_test_org.py` — imports deleted `dynamics.Organism`
- `coralai/instances/eincasm/eincasm.py` — references `World`, `dynamics.pcg`, `dynamics.ein_physics`, all deleted
- `coral_runner.py` / `coral_runner_old.py` — legacy, use `coral_runner_space.py` instead

### Partially implemented
- `SpaceEvolver.save_checkpoint()` — stub (`pass`), no checkpointing
- `SpaceEvolver.report_if_necessary()` — body exists but call is commented out in `run()`
- The genome culling strategy (`reduce_population_to_threshold`) kills lowest-cell-count genomes when count >100 — bad for biodiversity, a memory management workaround

### Known design issues
- `activate_outputs` applies `softmax(dim=0)` on invest/liquidate, forcing `invest + liquidate = 1` per cell; cells can never "do nothing," wastes energy at equilibrium. See `../logs/2026-02-26 Thu/2026-02-26 Thu invest liquidate softmax analysis.md`.
- `explore_physics` uses `argmax` on direction activations — discards magnitude, forces single-direction expansion per step
- No spatially-structured energy sources — energy injection is spatially homogeneous (uniform noise + sinusoidal offset)
- `LinearNet` in SpaceEvolver ignores NEAT hidden nodes — the network is always purely linear regardless of `num_hidden` in the config

### Taichi CPU gotcha
On the CPU backend, any cell with `genome = -1` causes out-of-bounds array access in `apply_weights_and_biases` (negative index into the weight array). The coral runner seeds genomes correctly at init, but mutations can momentarily create unowned cells. The GPU backends handle negative indices differently (wrap/clamp). Always initialize genome values to valid keys (≥ 0) when using CPU.

---

## Tests and Lint

```bash
make test    # runs test_cppn.py and test_multi_env_eval.py from PyTorch-NEAT
make lint    # flake8 E9/F63/F7/F82 (critical errors only)
```

- 5/6 PyTorch-NEAT tests pass on CPU-only (`test_cppn_unconnected` fails — hardcoded CUDA device in test, not a real issue)
- No test suite for `coralai` itself; `test.py` at root is a scratch script
- `make clean` removes `runs/`, `history/`, all `__pycache__`
