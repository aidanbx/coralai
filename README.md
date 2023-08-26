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

# Coding Style Standards
- Start with the biggest method - the most abstract, the one that does everything. If all of its helper functions were implemented the function would complete the task of that file.
- Take all the unimplemented sub methods and create their function declaration ABOVE the main method in reverse order like: 

Write a test for each function as you create it

Thinking order is characterized by doing a bottom up sweep of the functions. You first ideate what F is and then break it down....

I think mixing together the two is acceptable. You should be guided by what is the clearest given you current set of information, your priors. So, to make it algorithmic:

**Base Case:**
I know x1, I have a transistor, I have an axiom, etc.

**Induction**
Step 1:
1. I want to do: F
2. I need to break down F
3. Do I know enough about F to break it down?
4. Yes -> Do so
5. No -> Search for and expand priors to F, goto (3.)

Step 2:
F is made of subcomponents S
1. I want to do S
2. Pull all of S into memory
	1. Find most supported, established, obvious, highest attention, most enticing thing on the stack
		1. Begin to outline it and push subparts of outline onto stack 
		2. (EXPLORE BFS) Goto 3
	2. Find the most important thing on the stack
4. (EXPLOIT) Choose the most important thing on the stack and finish it
    1. Finish means give it its own fresh stack and compute until it is empty.


## Code:
```python
F(
	L(
		LL(x1),
		LR(x2)
	),
	R(
		RL(x3),
		RR(x4)
		)
)
```

The end program's order would be a mix of BFS and DFS depending on how:
- The programmer (incl. AI) balances exploration and exploitation and
- The amount of context/excitement available for different components.

## BFS:
```python
x4
x3
...
test(LL)
R(RL,RR)
test(R)
L(LL,LR)
test(L)
F(LR)
test(F)
```

## DFS:

**File Function Structure:**
```python
x4
...
LL(x1)
test(LL)
L(LL,LR)
test(L)
F(LR)
tesT(F)
```

# Contact
Feel free to reach out to aidanbx@gmail.com if you have questions or ideas
