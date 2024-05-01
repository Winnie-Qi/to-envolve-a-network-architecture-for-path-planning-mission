"""Handles genomes (individuals in the population)."""
from __future__ import division, print_function

from itertools import count
from random import choice, random, shuffle, randint
import copy

import sys

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGeneCNN, DefaultNodeGeneFC
from neat.graphs import creates_cycle
from neat.six_util import iteritems, iterkeys


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('conn_add_num', int),
                        ConfigParameter('conn_delete_num', int),
                        ConfigParameter('node_add_num', int),
                        ConfigParameter('node_delete_num', int),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected'),
                        ConfigParameter('num_cnn_layer', int),
                        ConfigParameter('dense_after_cnn', int),
                        ConfigParameter('num_gnn_layer', int),
                        ConfigParameter('dense_after_gnn', int),
                        ConfigParameter('nodes_every_layer', str),
                        ConfigParameter('kernel_size', int),
                        ConfigParameter('input_size', int),
                        ConfigParameter('full_connect_input', bool),
                        ConfigParameter('mutate_add_layer', float),
                        ConfigParameter('add_cnn_layer', float),
                        ConfigParameter('add_layer_double', float),
                        ConfigParameter('add_layer_halve', float),
                        ConfigParameter('add_fc_before_gnn', float),
                        ConfigParameter('node_add_one_layer', float),
                        ConfigParameter('node_delete_one_layer', float),
                        ConfigParameter('add_direct_conn', float),
                        ConfigParameter('direct_conn_num', int),
                        ConfigParameter('parameter_cost', float)]

        # Gather configuration data from the gene classes.
        self.cnn_node_gene_type = params['cnn_node_gene_type']
        self.fc_node_gene_type = params['fc_node_gene_type']
        self._params += self.cnn_node_gene_type.get_config_params()
        self._params += self.fc_node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1','yes','true','on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0','no','false','off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write('initial_connection      = {0} {1}\n'.format(self.initial_connection,
                                                                 self.connection_fraction))
        else:
            f.write('initial_connection      = {0}\n'.format(self.initial_connection))

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if not 'initial_connection' in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(iterkeys(node_dict))) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = "Invalid structural_mutation_surer {!r}".format(
                self.structural_mutation_surer)
            raise RuntimeError(error_string)

class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['cnn_node_gene_type'] = DefaultNodeGeneCNN
        param_dict['fc_node_gene_type'] = DefaultNodeGeneFC
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key, config):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}
        self.layer = []
        self.maxpooling_mask = [0] * config.num_cnn_layer
        for i in range(config.num_cnn_layer):
            if i%2 == 0:
                self.maxpooling_mask[i] = 1
        self.padding_mask = [1] * config.num_cnn_layer # padding of cnn, not maxpooling

        # Create layer: cnn layer + dense layer + gnn layer + dense layer
        for i in range(config.num_cnn_layer):
            self.layer.append(['cnn', set()])
        for i in range(len(self.layer), len(self.layer) + config.dense_after_cnn):
            self.layer.append(['fc', set()])
        for i in range(len(self.layer), len(self.layer) + config.num_gnn_layer):
            self.layer.append(['gnn', set()])
        for i in range(len(self.layer), len(self.layer) + config.dense_after_gnn):
            self.layer.append(['fc', set()])

        self.nodes_every_layers = [int(x) for x in config.nodes_every_layer.split(',')]
        self.num_cnn_layer = config.num_cnn_layer
        self.dense_after_cnn = config.dense_after_cnn
        self.num_gnn_layer = config.num_gnn_layer
        self.dense_after_gnn = config.dense_after_gnn

        self.cnn_nodes_conv_layer = []
        for i in range(self.num_cnn_layer):
            self.cnn_nodes_conv_layer.append([-1] * self.nodes_every_layers[i])

        # Compute output size of cnn
        convW = [config.input_size]
        FilterTaps = int(pow(config.kernel_size,0.5))
        for i in range(config.num_cnn_layer):
            W_tmp = int((convW[i] - FilterTaps + 2 * self.padding_mask[i])) + 1
            if i % 2 == 0:
                W_tmp = int((W_tmp - 2) / 2) + 1
            convW.append(W_tmp)
        self.size_width_every_cnn = convW[1:]
        self.size_output_cnn = convW[-1] * convW[-1]

        self.direct_conn = {}

        # Fitness results.
        self.fitness = 0

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key, len(self.layer)-1, 'fc', '_')
        # Add output layer nodes
        self.layer[-1][1] = set(config.output_keys)

        # Create node genes for the cnn nodes
        config.node_indexer = None # reset node_indexer
        for i in range(config.num_cnn_layer):
            for j in range(self.nodes_every_layers[i]):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                if i == 0:
                    node = self.create_node(config, node_key, i, 'cnn', [3,self.nodes_every_layers[i]])
                else:
                    node = self.create_node(config, node_key, i, 'cnn', self.nodes_every_layers[i-1:i+1])
                self.nodes[node_key] = node
                self.layer[i][1].add(node_key)

        # Create node genes for the fc nodes after cnn layer and for the gnn nodes
        for i in range(config.num_cnn_layer, config.num_cnn_layer + config.dense_after_cnn + config.num_gnn_layer):
            for j in range(self.nodes_every_layers[i]):
                node_key = config.get_new_node_key(self.nodes)
                node = self.create_node(config, node_key, i, 'fc', '_')
                self.nodes[node_key] = node
                self.layer[i][1].add(node_key)

        # Create connection genes for fc nodes
        if config.initial_connection == 'full':
            self.connect_full(config)
        elif config.initial_connection == 'partial':
            self.connect_partial(config)
        else:
            print("Only full and partial connection allowed in CNN!")

    def configure_crossover(self, genome1, genome2):
        """ Configure a new genome by crossover from two parent genomes. """

        if genome1.fitness < genome2.fitness:  # make sure parent1.fitness > parent2.fitness
            genome1, genome2 = copy.deepcopy(genome2), copy.deepcopy(genome1)

        # Inherit connection genes
        for key, cg1 in iteritems(genome1.connections):
            cg2 = genome2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        for key, cg1 in iteritems(genome1.direct_conn):
            cg2 = genome2.direct_conn.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.direct_conn[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.direct_conn[key] = cg1.crossover(cg2)

        # Inherit node genes
        for key, ng1 in iteritems(genome1.nodes):
            ng2 = genome2.nodes.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

        # inherit other property from genome1
        self.dense_after_cnn = copy.deepcopy(genome1.dense_after_cnn)
        self.dense_after_gnn = copy.deepcopy(genome1.dense_after_gnn)
        self.num_cnn_layer = copy.deepcopy(genome1.num_cnn_layer)
        self.layer = copy.deepcopy(genome1.layer)
        self.maxpooling_mask = copy.deepcopy(genome1.maxpooling_mask)
        self.nodes_every_layers = copy.deepcopy(genome1.nodes_every_layers)
        self.padding_mask = copy.deepcopy(genome1.padding_mask)
        self.size_output_cnn = copy.deepcopy(genome1.size_output_cnn)
        self.size_width_every_cnn = copy.deepcopy(genome1.size_width_every_cnn)
        self.direct_conn = copy.deepcopy(genome1.direct_conn)
        self.cnn_nodes_conv_layer = copy.deepcopy(self.cnn_nodes_conv_layer)

    def mutate(self, config, gid):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1,(config.node_add_prob + config.node_delete_prob +
                         config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob/div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob)/div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob)/div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob)/div):
                self.mutate_delete_connection(config)
        else:
            # pass
            # if random() < config.node_add_prob:
            #     self.mutate_add_node(config, gid)
            #
            # if random() < config.node_delete_prob:
            #     self.mutate_delete_node(config, gid)

            # if random() < config.mutate_add_layer:
            #     self.mutate_add_layer(config, gid)
            #
            # if random() < config.add_direct_conn:
            #     self.mutate_add_direct_conn(config, gid)

            if random() < config.conn_add_prob:
                self.mutate_conv_single_channel(config)
            #
            # if random() < config.conn_delete_prob:
            #     self.mutate_delete_connection(config)

        # Mutate connection genes (weight).
        for cg in self.connections.values():
            if len(cg.key) == 3:
                cg.mutate(config, [self.nodes_every_layers[cg.connect_layer[0]] * self.size_output_cnn, self.nodes_every_layers[cg.connect_layer[1]]])
            else:
                cg.mutate(config, self.nodes_every_layers[cg.connect_layer[0]:cg.connect_layer[1]+1])

        # for dg in self.direct_conn.values():
        #     dg.mutate(config, [self.nodes_every_layers[dg.connect_layer[0]],self.nodes_every_layers[dg.connect_layer[1]]])

        # Mutate node genes (bias, kernels).
        for ng in self.nodes.values():
            ng.mutate(config, [self.nodes_every_layers[ng.layer-1], self.nodes_every_layers[ng.layer]])

    def mutate_add_node(self, config, gid):
        info = open("info.txt", "a")
        if self.num_cnn_layer == 3:
            pass
        old_fitness = self.fitness
        in_one_layer = True if random() < config.node_add_one_layer else False # add nodes to one layer or randomly to all layers
        if in_one_layer:
            layer_num = randint(0, len(self.nodes_every_layers) - 2)

        for i in range(config.node_add_num):
            # Choose the layer to add node (not the last layer)
            if not in_one_layer:
                layer_num = randint(0, len(self.nodes_every_layers)-2)
            node_type = self.layer[layer_num][0]

            # Revise the nodes_every_layers list
            self.nodes_every_layers[layer_num] += 1

            new_node_id = config.get_new_node_key(self.nodes)
            if node_type == 'cnn':
                if layer_num == 0:
                    ng = self.create_node(config, new_node_id, layer_num, 'cnn',
                                          [config.num_inputs, self.nodes_every_layers[layer_num]])
                else:
                    ng = self.create_node(config, new_node_id, layer_num, 'cnn',
                                          self.nodes_every_layers[layer_num-1:layer_num+1])
                self.fitness -= (len(ng.kernel) + 1) * config.parameter_cost # The cost of parameters in this added convolution kernel
            else:
                ng = self.create_node(config, new_node_id, layer_num, 'fc', '_')
                self.fitness -= 1 * config.parameter_cost  # The cost of this added fc node (bias)

            self.layer[layer_num][1].add(new_node_id)
            self.nodes[new_node_id] = ng

            # if the added node in fc/gnn layer ---- add connections
            if layer_num > self.num_cnn_layer - 1:
                connections = []
                #  Add connections to the next layer
                if layer_num == len(self.nodes_every_layers)-2: # Add connection to output, if the added node in last layer
                    for output_id in config.output_keys:
                        connections.append((new_node_id, output_id))
                else:
                    for j in list(self.layer[layer_num + 1][1]):
                        connections.append((new_node_id, j))
                for node_id in connections:
                    connection = self.create_connection(config, node_id,
                                                        self.nodes_every_layers[layer_num : layer_num + 2],
                                                        (layer_num, layer_num + 1))
                    self.connections[connection.key] = connection
                self.fitness -= len(connections) * config.parameter_cost

                #  Add connections to the previous layer
                connections = []
                if layer_num == self.num_cnn_layer: # if the added node in the first fc layer after cnn
                    for i in list(self.layer[layer_num - 1][1]):
                        for j in range(self.size_output_cnn):
                            connections.append((i, new_node_id, j))
                else:
                    for j in list(self.layer[layer_num - 1][1]):
                        connections.append((j, new_node_id))
                for node_id in connections:
                    connection = self.create_connection(config, node_id,
                                                        self.nodes_every_layers[layer_num - 1:layer_num + 1],
                                                        (layer_num - 1, layer_num))
                    self.connections[connection.key] = connection
                self.fitness -= len(connections) * config.parameter_cost

            # if the added node in cnn layer but not the last cnn layer ---- add one layer to each kernel of next cnn layer
            elif layer_num < self.num_cnn_layer - 1:
                for node_key in self.layer[layer_num+1][1]:
                    kernel_attribute = next((a for a in self.nodes[node_key]._gene_attributes if a.name == 'kernel'),None)
                    new_kernel_layer = kernel_attribute.add_layer(config, self.nodes_every_layers[layer_num:layer_num+2])
                    self.nodes[node_key].kernel.extend(new_kernel_layer)
                    self.fitness -= len(new_kernel_layer) * config.parameter_cost # The cost of parameters in next layer convolution kernel

            # if the added node in the last cnn layer ---- add connections
            else:
                connections = []
                for i in list(self.layer[layer_num+1][1]):
                    for j in range(self.size_output_cnn):
                        connections.append((new_node_id, i, j))
                self.fitness -= len(connections) * config.parameter_cost
                for node_id in connections:
                    connection = self.create_connection(config, node_id,
                                                        [self.nodes_every_layers[layer_num] * self.size_output_cnn,
                                                         self.nodes_every_layers[layer_num + 1]], (layer_num, layer_num + 1))
                    self.connections[connection.key] = connection

        if in_one_layer:
            print("Genome No.{} add {} nodes in layer{}, with fitness reward {}".format(gid, config.node_add_num,
                                                                                        layer_num,
                                                                                        self.fitness - old_fitness))
            info.write("Genome No.{} add {} nodes in layer{}, with fitness reward {}\n".format(gid, config.node_add_num,
                                                                                        layer_num,
                                                                                        self.fitness - old_fitness))
        else:
            print("Genome No.{} add {} nodes randomly, with fitness reward {}".format(gid, config.node_add_num,
                                                                                        self.fitness - old_fitness))
            info.write("Genome No.{} add {} nodes randomly, with fitness reward {}\n".format(gid, config.node_add_num,
                                                                                        self.fitness - old_fitness))
        info.close()

    def mutate_add_direct_conn(self, config, gid):
        old_fitness = self.fitness
        in_layer = self.num_cnn_layer + self.dense_after_cnn - 1
        out_layer_list = [x for x in range(len(self.nodes_every_layers) - self.dense_after_gnn, len(self.nodes_every_layers))]
        out_layer = choice(out_layer_list)
        available_in_nodes = list(self.layer[in_layer][1])
        available_out_nodes = list(self.layer[out_layer][1])
        for i in range(config.direct_conn_num):
            in_node = choice(available_in_nodes)
            out_node = choice(available_out_nodes)
            if (in_node, out_node) in self.direct_conn: # do not duplicate connections
                continue
            connection = self.create_connection(config, (in_node, out_node),
                                                [(self.nodes_every_layers[out_layer - 1] + 1), self.nodes_every_layers[out_layer]],
                                                (in_layer, out_layer))
            self.direct_conn[connection.key] = connection
        self.fitness -= (i + 1) * config.parameter_cost
        print("Genome No.{} adds {} connections between layer {} and {}, with fitness reward {}".format(gid, i + 1,
                                                                                                        in_layer, out_layer,
                                                                                                        self.fitness - old_fitness))

    def mutate_delete_node(self, config, gid):
        info = open("info.txt", "a")
        old_fitness = self.fitness
        deleted_nodes = set()
        deleted_cnn_not_last_layer = {}
        in_one_layer = True if random() < config.node_delete_one_layer else False  # add nodes to one layer or randomly to all layers
        if in_one_layer:
            layer_num = randint(0, len(self.layer) - self.dense_after_gnn - 1)
            available_nodes = list(self.layer[layer_num][1])
            if len(available_nodes) < config.node_delete_num:
                in_one_layer = False
        else:
            available_nodes = [k for k in iterkeys(self.nodes) if k not in config.output_keys]

        for i in range(config.node_delete_num):
            if not available_nodes:
                break

            del_key = choice(available_nodes)
            available_nodes.remove(del_key)
            layer_num = self.nodes[del_key].layer
            deleted_nodes.add((del_key, layer_num))
            # If there is only one node
            while len(self.layer[layer_num][1]) <= 1:
                del_key = choice(available_nodes)
                layer_num = self.nodes[del_key].layer

            if layer_num < self.num_cnn_layer - 1: # deleted node in cnn layers but not in the last cnn layer
                # The number of layer of next-layer convolution kernel needs to be reduced
                del_node_index = sorted(list(self.layer[layer_num][1])).index(del_key)
                if layer_num not in deleted_cnn_not_last_layer:
                    deleted_cnn_not_last_layer[layer_num] = [del_node_index]
                else:
                    deleted_cnn_not_last_layer[layer_num].append(del_node_index)
                self.fitness += (len(self.nodes[del_key].kernel) + 1) * config.parameter_cost

            else: # deleted node in fc layers or the last cnn layer
                if self.direct_conn:
                    pass
                if layer_num < self.num_cnn_layer: # the last cnn layer
                    self.fitness += (len(self.nodes[del_key].kernel) + 1) * config.parameter_cost
                connections_to_delete = set()
                for connection in iteritems(self.connections):
                    if del_key in connection[0][:2]:
                        connections_to_delete.add(connection[0])
                for connection in iteritems(self.direct_conn):
                    if del_key in connection[0][:2]:
                        connections_to_delete.add(connection[0])

                for key in connections_to_delete:
                    try:
                        del self.connections[key]
                    except:
                        del self.direct_conn[key]

                self.fitness += (len(connections_to_delete) + 1) * config.parameter_cost

        # Reduce the number of layer in the next cnn kernel
        for layer_num in deleted_cnn_not_last_layer:
            ranges = [(config.kernel_size * i, config.kernel_size * (i + 1)) for i in deleted_cnn_not_last_layer[layer_num]]
            indices_to_remove = []
            for start, end in ranges:
                indices_to_remove.extend(range(start, end))
            for node_key in self.layer[layer_num + 1][1]:
                self.nodes[node_key].kernel = [elem for idx, elem in enumerate(
                    self.nodes[node_key].kernel) if idx not in indices_to_remove]
                self.fitness += config.kernel_size * config.parameter_cost


        for del_key, layer_num in deleted_nodes:
            self.layer[layer_num][1].remove(del_key)
            self.nodes_every_layers[layer_num] -= 1
            del self.nodes[del_key]

        if in_one_layer:
            print("Genome No.{} delete {} nodes in layer{}, with fitness reward {}".format(gid, i + 1, layer_num,
                                                                                        self.fitness - old_fitness))
            info.write("Genome No.{} delete {} nodes in layer{}, with fitness reward {}\n".format(gid, i + 1, layer_num,
                                                                                        self.fitness - old_fitness))
        else:
            print("Genome No.{} delete {} nodes randomly, with fitness reward {}".format(gid, i + 1,
                                                                                           self.fitness - old_fitness))
            info.write("Genome No.{} delete {} nodes randomly, with fitness reward {}\n".format(gid, i + 1,
                                                                                           self.fitness - old_fitness))
        info.close()

    def mutate_add_layer(self, config, gid):
        old_fitness = self.fitness
        a = random()
        if a < config.add_cnn_layer and self.num_cnn_layer < 5: # add cnn layer
            a = random()
            if a < config.add_layer_halve:
                node_add_num = int(self.nodes_every_layers[self.num_cnn_layer-1] / 2) # number of added nodes halves the last cnn layer
            elif a < config.add_layer_double:
                node_add_num = self.nodes_every_layers[self.num_cnn_layer - 1] * 2 # number of added nodes doubles the last cnn layer
            else:
                node_add_num = self.nodes_every_layers[self.num_cnn_layer - 1] # number of added nodes duplicates the last cnn layer

            # update configuration
            self.padding_mask.append(1)
            self.maxpooling_mask.append(1) if self.maxpooling_mask[-1] == 0 else self.maxpooling_mask.append(0)
            layer_num = self.num_cnn_layer
            self.num_cnn_layer += 1
            self.nodes_every_layers.insert(layer_num, node_add_num)
            # Add one to the number of layer in the attributes of every node in subsequent layers
            for _, layer in self.layer[layer_num:]:
                for node_key in layer:
                    try:
                        self.nodes[node_key].layer += 1
                    except:
                        self.nodes[node_key].layer += 1
            self.layer.insert(layer_num, ['cnn', set()])
            # recompute num_cnn_output
            FilterTaps = int(pow(config.kernel_size, 0.5))
            W_tmp = int((self.size_width_every_cnn[-1] - FilterTaps + 2 * self.padding_mask[-1])) + 1
            if self.maxpooling_mask[-1]:
                W_tmp = int((W_tmp - 2) / 2) + 1
            self.size_width_every_cnn.append(W_tmp)
            self.size_output_cnn = self.size_width_every_cnn[-1] * self.size_width_every_cnn[-1]

            # Delete connections between last cnn layer and the next fc layer
            connections_to_delete = set()
            for connection in iteritems(self.connections):
                if layer_num == connection[1].connect_layer[1]:
                    connections_to_delete.add(connection[0])
                connection[1].connect_layer = [x + 1 for x in connection[1].connect_layer]
            for key in connections_to_delete:
                del self.connections[key]
            self.fitness += (len(connections_to_delete) + 1) * config.parameter_cost

            # add nodes
            connections = []
            for i in range(node_add_num):
                new_node_id = config.get_new_node_key(self.nodes)
                ng = self.create_node(config, new_node_id, layer_num, 'cnn',
                                      [self.nodes_every_layers[layer_num - 1], node_add_num])
                self.nodes[new_node_id] = ng
                self.layer[layer_num][1].add(new_node_id)
                self.fitness -= (len(ng.kernel) + 1) * config.parameter_cost
                for j in list(self.layer[layer_num + 1][1]):
                    for k in range(self.size_output_cnn):
                        connections.append((new_node_id, j, k))

            # add new connections
            for node_id in connections:
                connection = self.create_connection(config, node_id,
                                                    [self.nodes_every_layers[layer_num] * self.size_output_cnn,
                                                     self.nodes_every_layers[layer_num + 1]],
                                                    (layer_num, layer_num + 1))
                self.connections[connection.key] = connection
            self.fitness -= len(connections) * config.parameter_cost

            print("Genome No.{} adds a cnn layer with {} nodes, with fitness reward {}".format(gid, node_add_num,
                                                                                            self.fitness - old_fitness))

        else: # add fc layer
            a = random()
            if a < config.add_fc_before_gnn: # add a fc layer before the gnn layer
                layer_num = self.num_cnn_layer + self.dense_after_cnn
                self.dense_after_cnn += 1
                a_log = 'before gnn layer'
            else: # add a fc layer before the output layer
                layer_num = len(self.nodes_every_layers) - 1
                self.dense_after_gnn += 1
                a_log = 'after gnn layer'

            a = random()
            if a < config.add_layer_halve:
                node_add_num = int(self.nodes_every_layers[layer_num - 1] / 2)  # number of added nodes halves the previous layer
            elif a < config.add_layer_double:
                node_add_num = self.nodes_every_layers[layer_num - 1] * 2  # number of added nodes doubles the previous layer
            else:
                node_add_num = self.nodes_every_layers[layer_num - 1]  # number of added nodes duplicates the previous layer

            # Add one to the number of layer in the attributes of every node in subsequent layers
            for _, layer in self.layer[layer_num:]:
                for node_key in layer:
                    try:
                        self.nodes[node_key].layer += 1
                    except:
                        self.nodes[node_key].layer += 1

            self.layer.insert(layer_num, ['fc', set()])

            # Delete connections between previous layer and next layer
            connections_to_delete = set()
            for connection in iteritems(self.connections):
                if layer_num == connection[1].connect_layer[1]:
                    connections_to_delete.add(connection[0])
                if connection[1].connect_layer[1] >= layer_num:
                    connection[1].connect_layer = [x + 1 for x in connection[1].connect_layer]
            for key in connections_to_delete:
                del self.connections[key]
            self.fitness += len(connections_to_delete) * config.parameter_cost

            self.nodes_every_layers.insert(layer_num, node_add_num)

            # Add nodes and connections
            for i in range(node_add_num):
                new_node_id = config.get_new_node_key(self.nodes)
                ng = self.create_node(config, new_node_id, layer_num, 'fc',
                                      [self.nodes_every_layers[layer_num - 1], node_add_num])
                self.nodes[new_node_id] = ng
                self.layer[layer_num][1].add(new_node_id)
                for i in list(self.layer[layer_num - 1][1]):
                    node_id = (i, new_node_id)
                    connection = self.create_connection(config, node_id,
                                                        [self.nodes_every_layers[layer_num - 1], node_add_num],
                                                        (layer_num - 1, layer_num))
                    self.connections[connection.key] = connection
                for i in list(self.layer[layer_num + 1][1]):
                    node_id = (new_node_id, i)
                    connection = self.create_connection(config, node_id,
                                                        [node_add_num, self.nodes_every_layers[layer_num + 1]],
                                                        (layer_num, layer_num + 1))
                    self.connections[connection.key] = connection
            self.fitness -= node_add_num * config.parameter_cost
            self.fitness -= (self.nodes_every_layers[layer_num - 1] * node_add_num *
                             self.nodes_every_layers[layer_num + 1]) * config.parameter_cost
            print("Genome No.{} adds a fc layer {} with {} nodes, with fitness reward {}".format(self.key, a_log,
                                                                                               node_add_num,
                                                                                               self.fitness - old_fitness))

    def mutate_conv_single_channel(self, config):
        # If it's the first time, only mutate nodes in the first cnn layer
        if not any(item != -1 for sublist in self.cnn_nodes_conv_layer for item in sublist):
            for j in self.layer[0][1]:

        for i in range(self.num_cnn_layer):
            for j in self.layer[i][1]:
                pass

    def mutate_depthwise_conv(self, config):
        available_in_nodes = list(self.layer[0][1])
        for i in range(config.depthwise_conv_num):
            choice(available_in_nodes)

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in iterkeys(other.nodes):
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in iteritems(self.nodes):
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in iterkeys(other.connections):
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in iteritems(self.connections):
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values()])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nNodes:".format(self.key, self.fitness)
        for k, ng in iteritems(self.nodes):
            s += "\n\t{0} {1!s}".format(k, ng)

        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)

        s += "\nLayers:"
        for i in range(len(self.layer)):
            s += "\n\t" + self.layer[i][0] + ": "
            l = list(self.layer[i][1])
            l.sort()
            for node in l:
                s += " {0}".format(node)
        return s

    @staticmethod
    def create_node(config, node_id, layer, node_type, num_nodes):
        if node_type == 'cnn':
            node = config.cnn_node_gene_type(node_id, layer)
        else:
            node = config.fc_node_gene_type(node_id, layer)
        node.init_attributes(config, num_nodes)
        return node

    @staticmethod
    def create_connection(config, node_id, num_nodes,connect_layer):
        connection = config.connection_gene_type(node_id, connect_layer)
        connection.init_attributes(config,num_nodes)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in iterkeys(self.nodes) if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in iterkeys(self.nodes) if i not in config.output_keys]
        output = [i for i in iterkeys(self.nodes) if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def compute_full_connections_with_layer(self, config, i):
        """
        Compute connections for a fully-connected cnn genome--each node in one
        layer connected to all nodes in the next layer
        """
        connections = []

        if self.layer[i-1][0] == 'cnn' and self.size_output_cnn != 1: # previous layer is cnn
            for node_i in self.layer[i-1][1]:
                for node_j in self.layer[i][1]:
                    for n in range(self.size_output_cnn):
                        connections.append((node_i, node_j, n))
        else:
            for node_i in self.layer[i-1][1]:
                for node_j in self.layer[i][1]:
                    connections.append((node_i, node_j))

        '''
        # Original none dense connention
        for i in range(len(self.layer) - 1):
             for node1 in self.layer[i][1]:
                    for node2 in self.layer[i+1][1]:
                        connections.append((node1, node2))
        '''

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in iterkeys(self.nodes):
                connections.append((i, i))

        return connections

    def connect_full(self, config):
        """
        Create a fully-connected cnn genome
        """
        fc_layer = [i for i in range(config.num_cnn_layer,len(self.layer))]
        for i in fc_layer:
            for node_id in self.compute_full_connections_with_layer(config, i):
                if len(node_id) == 3:
                    connection = self.create_connection(config, node_id,
                                                        [self.nodes_every_layers[i - 1] * self.size_output_cnn,
                                                         self.nodes_every_layers[i]], (i - 1, i))
                else:
                    connection = self.create_connection(config, node_id, self.nodes_every_layers[i-1:i+1],(i-1,i))
                self.connections[connection.key] = connection

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections_with_layer(config)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections."""
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection
