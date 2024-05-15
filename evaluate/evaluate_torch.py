import copy

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from scipy.sparse import coo_matrix

import neat.genome


class Net(nn.Module):

    def __init__(self, config, genome: neat.genome.DefaultGenome):
        super(Net, self).__init__()

        # Save useful neat parameters
        self.num_inputs = config.genome_config.num_inputs
        self.num_outputs = config.genome_config.num_outputs
        self.input_size = config.genome_config.input_size

        self.connections = genome.connections
        self.direct_conn = genome.direct_conn
        self.layers = genome.layers
        self.nodes = genome.nodes
        self.num_cnn_layer = genome.num_cnn_layer
        self.dense_afer_cnn = genome.dense_after_cnn
        self.num_gnn_layer = genome.num_gnn_layer
        self.dense_after_gnn = genome.dense_after_gnn
        self.padding_mask = genome.padding_mask
        self.maxpooling_mask = genome.maxpooling_mask
        self.nodes_every_layer = genome.nodes_every_layer
        self.size_output_cnn = genome.size_output_cnn
        self.size_width_every_cnn = genome.size_width_every_cnn
        self.cnn_nodes_conv_layer = genome.cnn_nodes_conv_layer
        self.num_parallel_fc = genome.num_parallel_fc
        self.par_connections = genome.par_connections
        self.par_layers = genome.par_layers
        self.par_nodes = genome.par_nodes

        self.cnn_layers = self._make_cnn_layers
        self.fc_layers_after_cnn = self._make_fc_layers_after_cnn()
        self.gnn_layer = self._make_gnn_layers()
        self.fc_layers_after_gnn = self._make_fc_layers_after_gnn()
        self.parallel_fc = self._make_parallel_fc()

        self.set_parameters()

    @property
    def _make_cnn_layers(self):

        layers = []

        for i in range(self.num_cnn_layer):
            layer = []
            if i == 0:
                for j in self.cnn_nodes_conv_layer[i]:
                    node = []
                    if j != -1:
                        node.append(nn.Conv2d(in_channels=1, out_channels=1,
                                            kernel_size=3, stride=1, padding=self.padding_mask[i], bias=True))
                    else:
                        node.append(nn.Conv2d(in_channels=self.num_inputs, out_channels=1,
                                              kernel_size=3, stride=1, padding=self.padding_mask[i], bias=True))
                    node.append(nn.BatchNorm2d(num_features=1))
                    node.append(nn.ReLU(inplace=True))
                    if self.maxpooling_mask[i]:
                        node.append(nn.MaxPool2d(kernel_size=2))
                    node = nn.Sequential(*node)
                    layer.append(node)
            else:
                for j in self.cnn_nodes_conv_layer[i]:
                    node = []
                    if j != -1:
                        node.append(nn.Conv2d(in_channels=1, out_channels=1,
                                      kernel_size=3, stride=1, padding=self.padding_mask[i], bias=True))
                    else:
                        node.append(
                            nn.Conv2d(in_channels=self.nodes_every_layer[i - 1], out_channels=1,
                                      kernel_size=3, stride=1, padding=self.padding_mask[i], bias=True))
                    node.append(nn.BatchNorm2d(num_features=1))
                    node.append(nn.ReLU(inplace=True))
                    if self.maxpooling_mask[i]:
                        node.append(nn.MaxPool2d(kernel_size=2))
                    node = nn.Sequential(*node)
                    layer.append(node)
            layers.append(layer)

        return layers

    def _make_fc_layers_after_cnn(self):
        layers = []

        for i in range(self.num_cnn_layer, self.num_cnn_layer + self.dense_afer_cnn):
            if i == self.num_cnn_layer and self.size_output_cnn != 1:
                layers.append(nn.Linear(in_features=self.nodes_every_layer[i - 1] * self.size_output_cnn,
                                        out_features=self.nodes_every_layer[i], bias=True))
            else:
                layers.append(nn.Linear(in_features=self.nodes_every_layer[i - 1],
                                        out_features=self.nodes_every_layer[i], bias=True))
            layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _make_gnn_layers(self):
        layers = []
        for i in range(self.num_cnn_layer + self.dense_afer_cnn, self.num_cnn_layer +
                                                                                  self.dense_afer_cnn + self.num_gnn_layer):
            layers.append(GCNConv(self.nodes_every_layer[i-1], self.nodes_every_layer[i], bias=True))
        self.gcn_layers = layers
        return nn.Sequential(*layers)

    def _make_fc_layers_after_gnn(self):
        layers = []
        for i in range(self.num_cnn_layer + self.dense_afer_cnn + self.num_gnn_layer, len(self.nodes_every_layer)):
            if i == len(self.nodes_every_layer) - 1:
                layers.append(nn.Linear(in_features=self.nodes_every_layer[i - 1],
                                        out_features=self.num_outputs, bias=True))
            else:
                layers.append(nn.Linear(in_features=self.nodes_every_layer[i - 1],
                                        out_features=self.nodes_every_layer[i], bias=True))
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_parallel_fc(self):
        layers = []
        for i in range(self.num_parallel_fc + 1):
            if i == 0:
                in_features = 2
            else:
                in_features = len(self.par_layers[i - 1])
            if i == self.num_parallel_fc:
                out_features = self.num_outputs
            else:
                out_features = len(self.par_layers[i])
            layers.append(nn.Linear(in_features, out_features, bias=True))
            if i != self.num_parallel_fc:
                layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x, x_p, gso):
        num_agent = x.shape[1]
        num_batch = x.shape[0]

        if 1:
            # CNN layers + fc layer
            direct_output = [[] for _ in range(num_agent)]
            extractFeatureMap = torch.zeros(num_batch, num_agent, self.nodes_every_layer[self.num_cnn_layer+self.dense_afer_cnn - 1]) # (64,10,x)
            if torch.cuda.is_available():
                extractFeatureMap = extractFeatureMap.cuda()
            for id_agent in range(num_agent):
                input_currentAgent = x[:, id_agent]
                for i in range(self.num_cnn_layer):
                    if i != 0:
                        input_currentAgent = cnn_output.clone()
                    cnn_output = torch.zeros(num_batch, self.nodes_every_layer[i], self.size_width_every_cnn[i], self.size_width_every_cnn[i])
                    if torch.cuda.is_available():
                        cnn_output = cnn_output.cuda()
                    for j in range(self.nodes_every_layer[i]):
                        try:
                            if self.cnn_nodes_conv_layer[i][j] == -1:
                                cnn_output_node = self.cnn_layers[i][j](input_currentAgent)
                            else:
                                cnn_output_node = self.cnn_layers[i][j](input_currentAgent[:, self.cnn_nodes_conv_layer[i][j]].unsqueeze(1))
                            cnn_output[:, j] = cnn_output_node.squeeze(1)
                        except:
                            pass

                cnn_output_flatten = cnn_output.view(cnn_output.size(0), -1)
                l = 0
                for layer in self.fc_layers_after_cnn:
                    cnn_output_flatten = layer(cnn_output_flatten)
                    if isinstance(layer, nn.Linear):
                        l += 1
                        nodes_list = self.layers[self.num_cnn_layer + l - 1][1]
                        for con in self.direct_conn:
                            if con[0] in nodes_list:
                                index = nodes_list.index(con[0])
                                direct_output[id_agent].append((con, cnn_output_flatten[:,index] * self.direct_conn[con].weight))
                extractFeatureMap[:, id_agent, :] = cnn_output_flatten

            # GNN layers
            feature_gcn = torch.zeros(num_batch, num_agent, self.nodes_every_layer[self.num_cnn_layer
                                                                                   +self.dense_afer_cnn+self.num_gnn_layer-1]) # (64,10,128)
            if torch.cuda.is_available():
                feature_gcn = feature_gcn.cuda()
            for batch in range(num_batch):
                adj_currentBatch = coo_matrix(gso[batch])
                edge_index_currentBatch = [adj_currentBatch.row, adj_currentBatch.col]
                edge_index_currentBatch = torch.tensor(np.array(edge_index_currentBatch), dtype=torch.int64)
                edge_weight_currentBatch = [adj_currentBatch.data]
                edge_weight_currentBatch = torch.Tensor(np.array(edge_weight_currentBatch)).squeeze()
                feature_currentBatch = extractFeatureMap[batch]
                if torch.cuda.is_available():
                    edge_index_currentBatch = edge_index_currentBatch.cuda()
                    edge_weight_currentBatch = edge_weight_currentBatch.cuda()
                for index, gcn in enumerate(self.gcn_layers):
                    if index == 0:
                        gcn_output = gcn(feature_currentBatch, edge_index_currentBatch, edge_weight_currentBatch)
                        gcn_output = torch.relu(gcn_output)
                    else:
                        gcn_output = gcn(gcn_output,edge_index_currentBatch,edge_weight_currentBatch)
                        gcn_output = torch.relu(gcn_output)
                feature_gcn[batch] = gcn_output

            # fc layers
            actions = []
            for id_agent in range(num_agent):
                l = 0
                action_currentAgent = feature_gcn[:, id_agent]
                for layer in self.fc_layers_after_gnn:
                    action_currentAgent = layer(action_currentAgent)
                    if isinstance(layer, nn.Linear):
                        l += 1
                        nodes_list = self.layers[len(self.layers) - self.dense_after_gnn + l - 1][1]
                        if direct_output[id_agent]:
                            for con in direct_output[id_agent]:
                                if con[0][1] in nodes_list:
                                    index = nodes_list.index(con[0][1])
                                    action_currentAgent[:, index] += con[1]
                actions.append(action_currentAgent)

        # parallel network
        if 0:
            actions_p = []
            output_parallel_fc = torch.zeros(num_batch, num_agent, self.num_outputs)
            if torch.cuda.is_available():
                output_parallel_fc = output_parallel_fc.cuda()
            for id_agent in range(num_agent):
                input_currentAgent = x_p[:, id_agent]
                output_currentAgent = self.parallel_fc(input_currentAgent)
                output_parallel_fc[:, id_agent, :] = output_currentAgent
                actions_p.append(output_currentAgent)
            actions = actions_p

        if 0:
            for a in range(len(actions)):
                actions[a] = actions[a] + actions_p[a]

        return actions

    def set_parameters(self):

        # layers that contain trainable parameters in fc layers
        layer = list()
        for module in self.children():
            for block in module:
                if isinstance(block, nn.Conv2d):
                    layer.append(block)
                elif isinstance(block, nn.Linear):
                    layer.append(block)
                elif isinstance(block, GCNConv):
                    layer.append(block)

        nodes = {}

        # add the input node to nodes dict
        for index, i in enumerate(range(-self.num_inputs, 0)):
            position = [-1, index]  # -1 means input node
            nodes.update({i: position})

        # add every layer to nodes dict
        for i in range(len(self.nodes_every_layer)):
            l = self.layers[i][1]
            for j in range(len(l)):
                position = [i, j]
                nodes.update({l[j]: position})

                # add conv kernel and bias to pytorch module
                if i < self.num_cnn_layer:
                    a = np.array(self.nodes[l[j]].kernel)
                    if i == 0:
                        try:
                            if self.cnn_nodes_conv_layer[i][j] == -1:
                                self.cnn_layers[0][j][0].weight.data = torch.FloatTensor(a.reshape(1, self.num_inputs, 3, 3))
                            else:
                                a = a[self.cnn_nodes_conv_layer[i][j] * 9 : (self.cnn_nodes_conv_layer[i][j] + 1) * 9]
                                self.cnn_layers[0][j][0].weight.data = torch.FloatTensor(a.reshape(1, 1, 3, 3))
                        except:
                            pass
                    else:
                        try:
                            if self.cnn_nodes_conv_layer[i][j] == -1:
                                self.cnn_layers[i][j][0].weight.data = torch.FloatTensor(a.reshape(1, self.nodes_every_layer[i - 1], 3, 3))
                            else:
                                a = a[self.cnn_nodes_conv_layer[i][j] * 9: (self.cnn_nodes_conv_layer[i][j] + 1) * 9]
                                self.cnn_layers[i][j][0].weight.data = torch.FloatTensor(a.reshape(1, 1, 3, 3))
                        except:
                            pass
                    b = self.nodes[l[j]].bias
                    self.cnn_layers[i][j][0].bias.data = torch.FloatTensor([b])
                else:
                    b = self.nodes[l[j]].bias
                    layer[i - self.num_cnn_layer].bias.data[j] = torch.FloatTensor([b])

        for node_id in self.connections:

            in_node = node_id[0]
            out_node = node_id[1]

            out_layer = nodes[out_node][0]   # out_node layer number

            if len(node_id) == 3:
                in_num = nodes[in_node][1] * self.size_output_cnn + node_id[2]
            else:
                in_num = nodes[in_node][1]
            out_num = nodes[out_node][1]

            if hasattr(layer[out_layer - self.num_cnn_layer], 'lin'): # connection within gcn layers
                (layer[out_layer - self.num_cnn_layer].lin.weight.data[out_num])[in_num] = \
                    torch.FloatTensor([self.connections[(in_node, out_node)].weight])

            elif len(node_id) == 3: # connection with the last cnn layer and fc layer
                try:
                    (layer[out_layer - self.num_cnn_layer].weight.data[out_num])[in_num] = \
                        torch.FloatTensor([self.connections[(in_node, out_node, node_id[2])].weight])
                except:
                    print(out_num, in_num)
                    print(layer)
                    print((layer[out_layer - self.num_cnn_layer].weight.data[out_num])[in_num])
                    print(torch.FloatTensor([self.connections[(in_node, out_node, node_id[2])].weight]))
            else:
                (layer[out_layer - self.num_cnn_layer].weight.data[out_num])[in_num] = \
                    torch.FloatTensor([self.connections[(in_node, out_node)].weight])

        # parallel part
        layer = list()
        for module in self.parallel_fc.children():
            if isinstance(module, nn.Linear):
                layer.append(module)

        nodes = {}
        for index, i in enumerate(range(-2, 0)):
            position = [-1, index]  # -1 means input node
            nodes.update({i: position})
        for i in range(len(self.par_layers)):
            l = self.par_layers[i]
            for j in range(len(l)):
                position = [i, j]
                nodes.update({l[j]: position})
                b = self.par_nodes[l[j]].bias
                layer[i].bias.data[j] = torch.FloatTensor([b])


        for node_id in self.par_connections:

            in_node = node_id[0]
            out_node = node_id[1]

            out_layer = nodes[out_node][0]   # out_node layer number
            in_num = nodes[in_node][1]
            out_num = nodes[out_node][1]

            (layer[out_layer].weight.data[out_num])[in_num] = \
                torch.FloatTensor([self.par_connections[(in_node, out_node)].weight])