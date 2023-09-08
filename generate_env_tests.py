from generate_env import *

# %% Test Generate Env
yaml_config = """
    environment:
    radius = 20
    boundary_condition: "torus"
    channels: 
        - "food"
        - "poison"
        - "obstacle"
        - "chemoattractant"
        - "chemorepellant"

    food_generation: "levy_dust"
    food_generation_params:
        pad: [0, 5] # these are ranges, low high
        alpha: [0.1, 2]
        
        beta: [-1, 1] 
        num_food: [50, 500]

    poison_generation: "levy_dust"
    poison_generation_params:
        pad: [0, 5]
        alpha: [0.1, 2]
        beta: [-1, 1]
        num_poison: [20, 200]

    obstacle_generation: "perlin_noise"
    obstacle_generation_params:
        threshold: [0.05, 0.2]
        frequency: [4.0, 16.0]
        octaves: [1, 4]
        persistence: [0.25, 1.0]
        lacunarity: [1.5, 3.0]

    chemoattractant_params:
        iterations: 300
        dropoff: 1
        
    chemorepellant_params:
        iterations: 300
        dropoff: 1

    visualize:
    colormaps:
        food: "Greens"
        poison: "Oranges"
        obstacle: "binary"
        chemoattractant: "Greens"
        chemorepellant: "Oranges"
    chemo_alpha: 0.9
"""
config = yaml.safe_load(yaml_config)
tester.test(lambda: generate_env(config, visualize=True),
            "Generate Environment",
            verbose,
            lambda result, title: visualize.show_image(result[1]))
        
# %%