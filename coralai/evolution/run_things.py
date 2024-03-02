def run_evolvable(vis, substrate, organism, genome_map):
    while vis.window.running:
        substrate.mem[:, organism.act_chinds] = organism.forward(substrate.mem, genome_map)

        vis.update()
        if vis.mutating:
            new_genome = organism.mutate(vis.perturbation_strength)
            organism.set_genome(organism.genome_key, new_genome) # mutates all cells at once
            organism.create_torch_net()


def run_cnn(vis, organism, substrate):
    while vis.window.running:
        substrate.mem[:, organism.act_chinds] = organism.forward(substrate.mem)

        vis.update()
        if vis.mutating:
            organism.mutate(vis.perturbation_strength)