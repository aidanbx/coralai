# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Coralai is a Python research framework for simulating emergent ecosystems of evolved Neural Cellular Automata (NCA). It uses PyTorch, Taichi Lang (GPU compute), and NEAT-Python for neuroevolution. See `README.md` for details.

### Environment Setup

- **Python venv** at `.venv/` — always activate with `source .venv/bin/activate`.
- **Git submodule** `coralai/dependencies/PyTorch-NEAT` must be on `master` branch (includes `linear_net.py`). Run `git submodule update --init --recursive` then `cd coralai/dependencies/PyTorch-NEAT && git checkout master`.
- The pinned dependency versions in `setup.py` (e.g. `torch==2.2.0.dev20230926`, `numpy==1.23.2`) are outdated nightly/dev builds incompatible with Python 3.12. The update script installs current compatible versions and then installs `coralai` in editable mode with `--no-deps`.

### Running the Application

- **Headless REPL** (`python headless_repl.py`): replaces the Taichi GGUI for headless Linux environments. Supports `--experiment minimal|nca|coral`, `--auto N` for non-interactive runs, and interactive commands (`step`, `mutate`, `paint`, `status`, `channels`, `save_frame`, `snapshot`, `quit`). Outputs video (`simulation.mp4`), frame PNGs, and a JSON log of all commands/stats to `runs/`.
- **XOR runner** (`python xor_runner.py`): simplest demo, evolves a NEAT network to solve XOR. No GPU or display required.
- Original GUI runners (`minimal_runner.py`, `nca_runner.py`, `coral_runner.py`) are hardcoded for Mac Metal (`ti.init(ti.metal)`, `torch.device("mps")`). Use the headless REPL instead on Linux.

### Gotchas

- On Taichi CPU backend, substrate cells with `genome = -1` cause out-of-bounds array access in Taichi kernels (negative index into weight arrays). Initialize all genome values to valid genome keys (>= 0).
- `coral_runner.py` imports `apply_physics` which only exists in `coral_physics_old.py` (fixed to import from there).
- `coralai.evolution.torch_organism` had a broken relative import for `nn_lib` (fixed to `from ..substrate.nn_lib`).
- Several subdirectories lack `__init__.py` files but work via Python 3.3+ implicit namespace packages.

### Tests

- No test suite for `coralai` itself. `test.py` at root is a scratch/debug script.
- PyTorch-NEAT submodule tests: `python -m pytest coralai/dependencies/PyTorch-NEAT/tests/test_cppn.py coralai/dependencies/PyTorch-NEAT/tests/test_multi_env_eval.py -v`. Some adaptive_linear tests fail due to shape changes on master branch. Maze tests require `gym`.

### Lint

- No lint configuration in the project. Use `flake8 coralai/ --select=E9,F63,F7,F82` for critical error checks.
