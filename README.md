# EINCASM: Emergent Intelligence in Neural Cellular Automaton Slime Molds

This repo contains the code for:

- EINCASM (ALIFE 2023)

EINCASM is a framework for studying emergent intelligence in cellular collectives resembling slime molds. The code in this repository is under heavy development. 

Here you will find code to evolve and visualize in real-time Neural Cellular Automata (NCA) with custom physics and neural architectures. Most of the heavy lifting is entirely on the GPU (+ Mac Metal) using [Taichi Lang](https://docs.taichi-lang.org/) and PyTorch. Organisms in EINCASM survive by mining capital (nutrients/energy) from resources whose productivity is subject to customizable weather. To grow, organisms must transport capital from cell to cell using muscle contractions, all at a cost. As organisms grow, their neural network weights and, soon, architecture mutate. This can be controlled via flexible mechanisms, including weather dynamics or generational merging/cross over using NEAT + PyTorch. 

The goal of EINCASM is to produce robust and adaptable organisms that can survive complexifying weather/rules and emergent interactions with other organisms. By doing so, the organisms develop instrumental goals, such as efficiently solving mazes and optimizing transport networks.

More work is to be done on experimentation, analysis, parameter searches, and the evolution of physics/rules.

# EINCASM

Conference paper presented at ALIFE 2023, Sapporo, Japan
- **Paper:** https://direct.mit.edu/isal/proceedings/isal/35/82/116945
- **Recording of Presentation:** https://www.youtube.com/watch?v=RuLQRgi6YSU&t=514s

# Contact
Feel free to reach out to aidanbx@gmail.com if you have questions or ideas
