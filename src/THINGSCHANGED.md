# Things I changed in the

## Before - 09/09/23:

- Moved initial tests from {filename} to {filename}_tests
- Added types to function headers
- Streamlined loading of config.yaml so it only needs to be read in once
- Moved older and probably unused (or soon to be) code to OLD_OTHER dir
    - videowriter.py
    - interactive_sim.py
    - graph.py
- Burned down (moved to OLD_HEXAGONS) all object oriented hexagons stuffs
- simulate_lifecycle.py -> simulate.py
- passing visualization as function to 

### Questions:
- Would stuff be sent to the GPU in run_lifecycle()?
- Do we need two processes? One for visualization and one for simulation
- I think I'm not following your testing structure very well... not using if __name__ in main file at all
