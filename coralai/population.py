import torch
import neat

from coralai.coralai_cor import CoralaiCor
from pytorch_neat.activations import identity_activation
from pytorch_neat.linear_net import LinearNet


class Population():
    def __init__(self, cor: CoralaiCor, num_genomes=1000, weight_thresh=0.0, weight_max=3.0):
        """Manages a population of genome, their networks, and their metadata
        Args:
            cor: a CoralaiCor instance (main params)
            num_genomes: int
            weight_thresh: min network weight value (to be clamped)
            weight_max: max network weight value (to be clamped)
        Returns:
            None
        """
        self.cor = cor
        self.weight_thresh = weight_thresh
        self.weight_max = weight_max
        self.genomes = [None] * num_genomes
        self.ages = [0] * num_genomes
        self.free_indices = list(range(num_genomes))
        self.num_genomes = num_genomes

        self.input_coords, self.output_coords = self.init_hyperneat_coords()
        self.combined_weights, self.combined_biases = self.init_combined_weights_biases()
        self.init_random_pop()


    def init_combined_weights_biases(self):
        """Allocates memory for all the networks of the simulation"""
        combined_weights = torch.zeros(
            self.num_genomes,
            1,
            self.cor.n_acts,
            self.cor.n_senses,
            device=self.cor.torch_device
        )
        combined_biases = torch.zeros(
            self.num_genomes,
            1,
            self.cor.n_acts,
            1,
            device=self.cor.torch_device
        )
        return combined_weights, combined_biases
    

    def init_hyperneat_coords(self):
        """Creates input/output mapping for HyperNEAT using the kernel, sensors, and actuators

        This is important for HyperNEAT as it can enable a network to learn the
          geometry of its input/output space. This means the order of the kernel
          can introduce bias. 
        """
        input_coords = []
        for ch in range(self.cor.n_senses):
             input_coords.append([0, 0, self.cor.sense_chinds[ch]])
             for offset_i in range(self.cor.kernel.shape[0]):
                offset_x = self.cor.kernel[offset_i, 0]
                offset_y = self.cor.kernel[offset_i, 1]
                input_coords.append([offset_x, offset_y, self.cor.sense_chinds[ch]])

        output_coords = []
        for ch in range(self.cor.n_acts):
            output_coords.append([0, 0, self.cor.act_chinds[ch]])
        
        return input_coords, output_coords


    def init_random_pop(self):
        """Creates random genomes based on the NEAT Config and generates their networks"""
        for i in range(self.num_genomes):
            genome = neat.DefaultGenome(str(i))
            genome.configure_new(self.cor.neat_config.genome_config)
            self.genomes[i] = genome
            net = self.create_torch_net(genome)
            self.combined_weights[i] = net.weights
            self.combined_biases[i] = net.biases


    def create_torch_net(self, genome):
        """Uses the CPPN of a genome to produce the weights and biases of a network"""
        net = LinearNet.create(
            genome,
            self.cor.neat_config,
            input_coords=self.input_coords,
            output_coords=self.output_coords,
            weight_threshold=self.weight_thresh,
            weight_max=self.weight_max,
            activation=identity_activation,
            cppn_activation=identity_activation,
            device=self.cor.torch_device,
        )
        return net

    
    def report(self):
        # self.weight_max
        # self.weight_thresh
        pass
