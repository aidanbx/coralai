# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

Coralai is a Python research framework for simulating emergent ecosystems of evolved Neural Cellular Automata (NCA). It uses PyTorch, Taichi Lang (GPU compute), and NEAT-Python for neuroevolution. See `README.md` for details.

### Environment Setup

- **Python venv** at `.venv/` — always activate with `source .venv/bin/activate`.
- **Git submodule** `coralai/dependencies/PyTorch-NEAT` must be initialized: `git submodule update --init --recursive`.
- The pinned dependency versions in `setup.py` (e.g. `torch==2.2.0.dev20230926`, `numpy==1.23.2`) are outdated nightly/dev builds incompatible with Python 3.12. The update script installs current compatible versions and then installs `coralai` in editable mode with `--no-deps`.

### Running the Application

- **XOR runner** (`python xor_runner.py`): the simplest demo. Evolves a NEAT neural network to solve XOR. No GPU or display required. Best "hello world" for headless environments.
- Other runners (`minimal_runner.py`, `nca_runner.py`, `coral_runner.py`) require a GUI (Taichi GGUI) and are hardcoded for Mac Metal (`ti.init(ti.metal)`, `torch.device("mps")`). On Linux, these would need `ti.init(ti.cpu)` or `ti.init(ti.cuda)` and `torch.device("cpu")` or `torch.device("cuda")`.

### Known Codebase Issues

- `pytorch_neat.linear_net` module does not exist in the PyTorch-NEAT submodule (neither in the fork nor upstream). This causes `ImportError` when importing `coralai.evolution.hyper_organism`, `coralai.evolution.neat_evolver`, or `coralai.evolution.cppn_organism` (transitively). Modules that depend on `LinearNet` cannot be loaded.
- `coralai.evolution.torch_organism` has a broken relative import (`from .nn_lib import ch_norm`) — `nn_lib.py` is actually in `coralai/substrate/`, not `coralai/evolution/`.
- Several subdirectories (`coralai/evolution/`, `coralai/utils/`, `coralai/instances/xor/`, `coralai/instances/minimal/`) lack `__init__.py` files but work via Python 3.3+ implicit namespace packages.

### Tests

- No test suite for `coralai` itself. The `test.py` at root is a scratch/debug script.
- PyTorch-NEAT submodule has tests at `coralai/dependencies/PyTorch-NEAT/tests/`. Run with: `python -m pytest coralai/dependencies/PyTorch-NEAT/tests/test_adaptive_linear.py coralai/dependencies/PyTorch-NEAT/tests/test_cppn.py coralai/dependencies/PyTorch-NEAT/tests/test_multi_env_eval.py -v`. Maze tests require the `gym` package (not installed by default).

### Lint

- No lint configuration in the project. Use `flake8 coralai/ --select=E9,F63,F7,F82` for critical error checks.
