# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Coralai is a Python research framework for simulating emergent ecosystems of evolved Neural Cellular Automata (NCA). It uses PyTorch, Taichi Lang (GPU compute), and NEAT-Python for neuroevolution. See `README.md` for details.

### Quick Setup

One command — auto-detects conda or venv:
```bash
./setup.sh
```
Or explicitly: `./setup.sh conda` / `./setup.sh venv`. Then `source .venv/bin/activate` (or `conda activate coralai`).

Dependencies are in `requirements.txt` (pip) and `environment.yml` (conda). PyTorch-NEAT is vendored directly — no git submodules needed.

### Running

Use `make help` to see all targets. Key commands:
- `make run-minimal` — interactive headless REPL (minimal NCA)
- `make run-nca` — interactive headless REPL (NCA with RGB/hidden channels)
- `make run-xor` — XOR NEAT evolution (no GUI needed)
- `make demo` — non-interactive auto demo, outputs video
- `make test` / `make lint`

The headless REPL (`headless_repl.py`) replaces the Taichi GGUI. It supports interactive commands (`step`, `mutate`, `paint`, `status`, `channels`, `save_frame`, `snapshot`, `quit`) and outputs video + JSON log to `runs/`.

### Gotchas

- On Taichi CPU backend, substrate cells with `genome = -1` cause out-of-bounds array access in Taichi kernels (negative index into weight arrays). Initialize all genome values to valid keys (>= 0).
- Original GUI runners (`minimal_runner.py`, `nca_runner.py`, `coral_runner.py`) are hardcoded for Mac Metal. Use the headless REPL on Linux.
- `test_cppn_unconnected` test fails on CPU-only machines (hardcoded CUDA device in PyTorch-NEAT test). Not a real issue.

### Tests & Lint

- `make test` — runs PyTorch-NEAT tests (5/6 pass on CPU-only)
- `make lint` — flake8 critical error checks
- No test suite for `coralai` itself; `test.py` at root is a scratch script
