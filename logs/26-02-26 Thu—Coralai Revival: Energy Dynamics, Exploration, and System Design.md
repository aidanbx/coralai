# 26-02-26 Thu — Coralai Revival: Energy Dynamics, Exploration, and System Design

---

## Revisiting the project after a long break

Coming back to CoralAI after time away. Reading through the codebase with Claude. Key realization: `coral_runner_space.py` is the most advanced runner and the right place to start. The EINCASM instance is broken (deleted module refs). A lot of the older runners are vestigial.

Observation: the invest/liquidate softmax in `coral_physics.py` is a significant design problem — cells can never "do nothing," magnitude information is lost, and the economy is zero-sum per step. See AI analysis file.

---

## Questions and directions opened up today

- How does energy actually move through the system? Does exploration move energy or just genome ownership?
- Can cells discover / harvest energy from the environment, or is it just injected as noise?
- Multi-directional exploration: why pick only the highest direction? Could bids be proportional and simultaneous?
- Infrastructure as defense: high infra = harder to colonize? Immune cell types?
- Symbiosis as default in biology (Lewis Thomas, *Lives of a Cell*) — the current model has no cooperation mechanism between genomes. Everything is adversarial.
- Energy vs Infrastructure: what is really the difference? Both spread to neighbors. Is the total (energy + infra) the real "cell capacity"?
- Energy attracted to infrastructure: is this like osmosis? Can you drain neighboring cells passively by building infra?
- The weather (sinusoidal energy offset) and random chunk killing: is this too much to disentangle? Hard to run controlled experiments.
- The genome culling behavior (kill lowest cell-count genomes when >100 genomes exist) is bad for biodiversity and is really a memory management hack.

---

## Intent going forward

Revival of the project as an active research effort. Goals:
- Fix known design issues (softmax, culling, no checkpointing)
- Make the system modular enough to run controlled experiments
- Build up from minimal complexity rather than all-in
- Keep biological analogy honest and intentional

See AI response files in `2026-02-26 Thu/` for full technical write-up.
