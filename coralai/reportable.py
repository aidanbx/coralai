class Reportable:
    def __init__(self, metadata):
        self.metadata = metadata

    def report(self):
        pass

    def load_from_report(self, filepath):
        pass




    def save_genomes(self, genomes, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.genomes, f)
    
    def load_genomes(self, filename):
        with open(filename, 'rb') as f:
            genomes = pickle.load(f)
        return genomes
    
    def report_if_necessary(self, timestep):
        if timestep % 300 == 0:
            timestep_dir = os.path.join(self.checkpoint_dir, f"step_{timestep}")
            os.makedirs(timestep_dir, exist_ok=True)
            self.substrate.save_mem_to_pt(filepath = os.path.join(timestep_dir, "sub_mem"))
            self.save_genomes(self.genomes, os.path.join(timestep_dir, "genomes"))
            with open(os.path.join(timestep_dir, "ages"), 'wb') as f:
                pickle.dump(self.ages, f)

        # for i in range(len(self.genomes)):
        #     # org['genome'].fitness += self.substrate.mem[0, inds.genome].eq(i).sum().item()
        #     self.genomes[i].fitness = fitness_function(self.genomes[i], i)
        # self.reporters.start_generation(self.generation)

        # # Evaluate all genomes using the user-provided function.
        # fitness_function(list(iteritems(self.population)), self.config)

        # # Gather and report statistics.
        # best = None
        # for g in itervalues(self.population):
        #     if best is None or g.fitness > best.fitness:
        #         best = g

        # self.reporters.post_evaluate(self.config, self.population, self.species, best)

        # # # Create the next generation from the current generation.
        # # self.population = self.reproduction.reproduce(self.config, self.species,
        # #                                               self.config.pop_size, self.generation)

        # # Check for complete extinction.
        # if not self.species.species:
        #     self.reporters.complete_extinction()
        #     # TODO: re-seed env

        # # Divide the new population into species.
        # self.species.speciate(self.config, self.population, self.generation)

        # self.reporters.end_generation(self.config, self.population, self.species)

        # self.generation += 1

        # if self.config.no_fitness_termination:
        #     self.reporters.found_solution(self.config, self.generation, self.best_genome)

        # return self.best_genome
    
    def add_reporter(self, reporter):
        self.reporters.add(reporter)

    def remove_reporter(self, reporter):
        self.reporters.remove(reporter)    

    def get_genome_infra_sum(self, genome_key):
        inds = self.substrate.ti_indices[None]
        infra_sum = torch.where(self.substrate.mem[0, inds.genome] == genome_key, self.substrate.mem[0, inds.infra], 0).sum()
        return infra_sum


    def load_checkpoint(self, folderpath):
        config_path = os.path.join(folderpath, "..", "neat_config")
        self.neat_config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                        config_path)
        self.load_genomes(os.path.join(folderpath, "genomes"))
        with open(os.path.join(folderpath, "ages"), 'rb') as f:
            self.ages = pickle.load(f)

        self.combined_biases = []
        self.combined_weights = []
        for genome in self.genomes:
            genome.fitness = 0.0
            net = self.create_torch_net(genome)
            self.combined_biases.append(net.biases)
            self.combined_weights.append(net.weights) 
        loaded_mem = torch.load(os.path.join(folderpath, "sub_mem"))
        self.substrate.mem = loaded_mem