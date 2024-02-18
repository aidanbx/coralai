# Coralai: Emergent Ecosystems of Evolved Neural Cellular Automata


This repo contains the code for:
- Coralai (Master's Thesis 2024)
- EINCASM (ALIFE 2023)


Coralai is a framework for simulating and studying emergent ecosystem in neural cellular collectives resembling slime molds. The code in this repository is under heavy development.

Coralai....
- Can evolve and visualize, in real-time, Neural Cellular Automata (NCA) with custom physics and neural architectures.
- Runs on GPU (+ Mac Metal) using [Taichi Lang](https://docs.taichi-lang.org/) and PyTorch NEAT.
- Selects organisms that successfully mine, distribute, and utilize capital/energy from other organisms or resources.
- Can generate customizable weather patterns to determine resource productivity and physical parameters
- Contains physics for organisms to transport capital from cell to cell using muscle contractions, all at a cost of capital.   

The goal of coralai is to produce robust and adaptable organisms that can survive complexifying weather/rules and emergent interactions with other organisms. By doing so, the organisms develop instrumental goals, such as efficiently solving mazes and optimizing transport networks.

More work is to be done on experimentation, analysis, parameter searches, and the evolution of physics/rules.

# Code Organization
### ./
- Here is where you create a new experiment and run it. 
- ./nca_slim.py is a simple example experiment
### ./coralai
- This module contains all the necessary code for running coralai experiments
### ./coralai/requirements
- Currently only Pytorch NEAT, normal pip install are included in requirements.txt
### ./coralai/substrate
- The universe the coralai runs in. Mostly memory management and manipulation/evolution of organism architectures
### ./coralai/dynamics
- Code that modifies memory incl. physics, organisms (NCA), weather, and procedural content generation.
### ./coralai/analysis
- Visualizations, statistics, and experimental frameworks that track and evolve organisms over time.
### ./coralai/instances
- Specific instances of coralai, including basic random NCA and EINCASM (see below)

## EINCASM: Emergent Intelligence in Neural Cellular Automata Slime Molds

This is the old title of this research, which has since been expanded to capture a larger diversity of emergent behaviors

Conference paper presented at ALIFE 2023, Sapporo, Japan
- **Paper:** https://direct.mit.edu/isal/proceedings/isal/35/82/116945
- **Recording of Presentation:** https://www.youtube.com/watch?v=RuLQRgi6YSU&t=514s
- **Workshop Discussion on Machine Love and Human Flourishing:** https://www.youtube.com/watch?v=tfQhXOBchKY

# Contact

Feel free to reach out to aidanbx@gmail.com if you have questions or ideas