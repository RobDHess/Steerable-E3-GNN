import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_scatter import scatter_mean
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps, spherical_harmonics

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm
from .balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from .node_attribute_network import NodeAttributeNetwork


class SEGNNModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, N, norm, lmax_h, lmax_pos=None):
        super(SEGNNModel, self).__init__()

        if lmax_pos == None:
            lmax_pos = lmax_h

        # Irreps for the node features
        if lmax_h == 0:
            node_in_irreps = Irreps("{0}x0e".format(input_features.dim))       # Irreps for the input
            node_out_irreps = Irreps("{0}x0e".format(output_features.dim))     # Irreps for the output
        else:
            node_in_irreps = input_features
            node_out_irreps = output_features

        # Irreps for the edge and node attributes
        attr_irreps = Irreps.spherical_harmonics(lmax_pos)
        self.attr_irreps = attr_irreps

        # Irreps for the hidden activations (s.t. the nr of weights in the TP approx that of a standard linear layer)
        hidden_irreps_scalar = Irreps("{0}x0e".format(hidden_features))
        hidden_irreps = WeightBalancedIrreps(
            hidden_irreps_scalar, attr_irreps, True, lmax=lmax_h)  # True: copies of sh

        # Network for computing the node attributes
        self.node_attribute_net = NodeAttributeNetwork()

        # The embedding layer (acts point-wise, no orientation information so only use trivial/scalar irreps)
        self.embedding_layer = O3TensorProduct(node_in_irreps,         # in
                                               hidden_irreps,     # out
                                               attr_irreps)       # steerable attribute

        # The main layers
        self.layers = []
        for i in range(N):
            self.layers.append(SEGNN(hidden_irreps,  # in
                                     hidden_irreps,  # hidden
                                     hidden_irreps,  # out
                                     attr_irreps,    # steerable attributes
                                     norm=norm))
        self.layers = nn.ModuleList(self.layers)

        # The output network (again via point-wise operation via scalar irreps)
        self.output_layer_1 = O3TensorProductSwishGate(hidden_irreps,           # in
                                                       hidden_irreps,           # out
                                                       attr_irreps)             # steerable attribute
        self.output_layer_2 = O3TensorProduct(hidden_irreps,
                                              node_out_irreps,
                                              attr_irreps)

    def forward(self, graph):
        """ Embed, pass messages, graph-pool and output """
        # Unpack the graph
        pos, vel, charges, edge_index, batch = graph.x, graph.vel, graph.charges, graph.edge_index, graph.batch

        prod_charges = charges[edge_index[0]] * charges[edge_index[1]]
        rel_pos = pos[edge_index[0]] - pos[edge_index[1]]
        edge_dist = torch.sqrt(rel_pos.pow(2).sum(-1, keepdims=True))
        vel_abs = torch.sqrt(vel.pow(2).sum(-1, keepdims=True))
        mean_pos = scatter_mean(pos, batch, dim=0)
        mean_pos = mean_pos.repeat_interleave(5, dim=0)

        # Construct the node and edge attributes
        edge_attr = spherical_harmonics(self.attr_irreps, rel_pos, normalize=True, normalization='integral')
        vel_embedding = spherical_harmonics(self.attr_irreps, vel, normalize=True, normalization='integral')
        node_attr = self.node_attribute_net(edge_index, edge_attr) + vel_embedding

        # The embedding layer
        x = self.embedding_layer(torch.cat((pos-mean_pos, vel, vel_abs), -1), node_attr)

        # The main layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_dist, prod_charges, edge_attr, node_attr, batch)

        # The output layers
        x = self.output_layer_1(x, node_attr)
        x = self.output_layer_2(x, node_attr)

        # The output is a difference vector
        return graph.x + x


class SEGNN(MessagePassing):
    """
        E(3) equivariant message passing layer.
    """

    def __init__(self, node_in_irreps, hidden_irreps, node_out_irreps, attr_irreps, norm):
        super(SEGNN, self).__init__(node_dim=-2, aggr="add")

        self.norm = norm

        # The message network layers
        irreps_message_in = (node_in_irreps + node_in_irreps + Irreps("2x0e")).simplify()
        self.message_layer_1 = O3TensorProductSwishGate(irreps_message_in,
                                                        hidden_irreps,
                                                        attr_irreps)
        self.message_layer_2 = O3TensorProductSwishGate(hidden_irreps,
                                                        hidden_irreps,
                                                        attr_irreps)

        # The node update layers
        irreps_update_in = (node_in_irreps + hidden_irreps).simplify()
        self.update_layer_1 = O3TensorProductSwishGate(irreps_update_in,
                                                       hidden_irreps,
                                                       attr_irreps)
        self.update_layer_2 = O3TensorProduct(hidden_irreps,
                                              node_out_irreps,
                                              attr_irreps)

        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(hidden_irreps)
            self.message_norm = BatchNorm(hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(hidden_irreps)

    def forward(self, x, edge_index, edge_dist, prod_charges, edge_attr, node_attr, batch):
        """ Propagate messages along edges """
        x = self.propagate(edge_index, x=x, edge_dist=edge_dist, prod_charges=prod_charges, node_attr=node_attr, edge_attr=edge_attr)
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_dist, prod_charges, edge_attr):
        """ Create messages """
        message = self.message_layer_1(torch.cat((x_i, x_j, edge_dist, prod_charges), dim=-1), edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """ Update note features """
        update = self.update_layer_1(torch.cat((x, message), dim=-1), node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x
