# EINCASM: Emergent Intelligence in Neural Cellular Automaton Slime Molds

This repo contains the complete code for:

- EINCASM (ALIFE 2023)

EINCASM is a prototype system for studying emergent intelligence in organisms resembling slime molds. The repository contains code to evolve neural cellular automata with NEAT and assess growth and adaptation in complex environments.

# EINCASM

Conference paper presented at ALIFE 2023, Sapporo, Japan: ["EINCASM: Emergent Intelligence in Neural Cellular Automaton Slime Molds"](https://direct.mit.edu/isal/proceedings/isal/35/82/116945)

### Current Status: `Under Development/Rewriting`

- **Fleshed Out**:
    - Environment Generation 
    - Life Cycle
    - Visualization

- **In Progress**:
    - Local Physics/Constraints
    - Interactive Sim

- **TODO**:
    - NEAT Evolution
    - Checkpointing
    - Weather/Parameter Search
    - Analysis

- **Long Goals**:
    - Convert to Torch or JAX
    - Multi-agent Environments
    - Unindividuated/Neural Soup Environments
    - Large Scale Worlds
    - LociBrain Integration
    - Critical State/Power Law/Scale Free Analysis

# Coding Standards
## Coding Process:
- Find something to do
- Does it deserve a file? More or less?
- e.g file: 
    - Assume F is a function that does everything exactly as you think you want
    - Write a simple test case for F, below F
    - Define F and sketch it out with a series of subcomponents
    - Write F's dependencies, its subcomponents above F following the same process

In the end your file is runnable from the top down. Once you hit F, everything F needs you have already seen.

## Example:

### F's Subcomponent Tree:
The leaves (x) are axioms/predefined:
```python
        F
       / \
      /   \
     L     R
    / \   / \
   LL LR RL RR
  /   |   |   \
 x1  x2   x3   x4
```


## Resultant Program:

The end program's order would be a mix of BFS and DFS depending on how:
- The programmer (incl. AI) balances exploration and exploitation and
- The amount of context/excitement available for different components.

### BFS:
```python
# %% BFS: Initialize Variables ------------------------------------------------
x4
x3
.
.
# %% LR -----------------------------------------------------------------------
LR(x2)
.
.
# %% R ------------------------------------------------------------------------
R(LL,LR)
if __name__ == "__main__":
    test(R)

# %% L ------------------------------------------------------------------------
L(LL,LR)
if __name__ == "__main__":
    test(L)


# %% F ------------------------------------------------------------------------
F(LR)

if __name__ == "__main__":
    test(F)
```

### DFS:

**File Function Structure:**
```python
x4
# %% R-------------------------------------------------------------------------
RR(x4)
.
.
.
# %% LL------------------------------------------------------------------------
LL(x1)
if __name__ == "__main__":
    test(LL)


# %% L-------------------------------------------------------------------------
L(LL,LR)
if __name__ == "__main__":
    test(L)


# %% F-------------------------------------------------------------------------
F(LR)
if __name__ == "__main__":
    test(F)

```

# Contact
Feel free to reach out to aidanbx@gmail.com if you have questions or ideas
