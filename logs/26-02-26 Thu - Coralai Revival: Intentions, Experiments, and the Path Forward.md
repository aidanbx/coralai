# 26-02-26 Thu - Coralai Revival: Intentions, Experiments, and the Path Forward

---

## Revisiting the project after a long break

Coming back to CoralAI after time away. Reading through the codebase with Claude. Key realization: `coral_runner_space.py` is the most advanced runner and the right place to start. The EINCASM instance is broken (deleted module refs). A lot of the older runners are vestigial.

Observation: the invest/liquidate softmax in `coral_physics.py` is a significant design problem - cells can never "do nothing," magnitude information is lost, and the economy is zero-sum per step. See AI analysis file.

---

## Questions and directions opened up today

- How does energy actually move through the system? Does exploration move energy or just genome ownership?
- Can cells discover / harvest energy from the environment, or is it just injected as noise?
- Multi-directional exploration: why pick only the highest direction? Could bids be proportional and simultaneous?
- Infrastructure as defense: high infra = harder to colonize? Immune cell types?
- Symbiosis as default in biology (Lewis Thomas, *Lives of a Cell*) - the current model has no cooperation mechanism between genomes. Everything is adversarial.
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

---

## Revival intentions (evening)

There are unfinished Git branches: one around analysis/checkpointing/experiments, one around refactoring the codebase into a cleaner module. Neither is complete.

**Core vision for the revival:**

The instances pattern is right — write some physics, configure channels and a kernel, run an experiment. That should be as fast as possible. The missing piece is the analysis/experiment layer that sits on top: every run should be fully reproducible, with the physics code, substrate state, genomes, and channel config all saved alongside the output data. Not just logs of what happened, but everything needed to rerun it exactly, even if the codebase evolves later.

**Experimental directions to explore:**
- Fix known physics issues: softmax on invest/liquidate, argmax-only exploration, homogeneous energy injection
- Spatial resource patches (Perlin/Lévy distributed, regenerating) to create navigation pressure
- Infra-as-defense in colonization; proportional multi-directional bidding
- Symbiosis mechanisms: mixed genome cell occupancy, kin recognition via com channels
- Multi-scale complexity metrics to quantify how "alive" a run looks — compare across physics variants
- Parallel substrate batching: run N experiments simultaneously on one GPU by expanding the batch dim

**Longer-horizon ideas:**
- 3D substrates (computationally expensive but feasible at smaller grid sizes)
- Better visualization: the current RGB projection loses almost all the information in the substrate. Need interactive channel selection, heatmaps, time-series plots, maybe a web frontend
- C++ / game engine port for better interactivity and rendering — though Python + Taichi may be sufficient if the visualization layer improves
- Parameter sweep infrastructure: spin up a GPU somewhere, run N physics variants in parallel, collect complexity metrics, compare

**Overarching question:** Can you build a framework where writing a new physics rule and getting a full experimental result — with reproducibility, analysis, and comparison to baselines — takes an afternoon rather than weeks? That's the goal.
