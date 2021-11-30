import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from torch_scatter import scatter_mean
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps, spherical_harmonics

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate, O3SwishGate
from .instance_norm import InstanceNorm
from .balanced_irreps import BalancedIrreps, WeightBalancedIrreps
from .node_attribute_network import NodeAttributeNetwork


class SEConvModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_features, N, norm, lmax_h, lmax_pos=None, linear=True):
        super(SEConvModel, self).__init__()

        if lmax_pos == None:
            lmax_pos = lmax_h

        # Irreps for the node features
        if lmax_h == 0:
            node_in_irreps = Irreps("{0}x0e".format(input_features.dim))    # Irreps for the input
            node_out_irreps = Irreps("{0}x0e".format(output_features.dim))  # Irreps for the output
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
                                               hidden_irreps,          # out
                                               attr_irreps)            # steerable attribute

        # The main layers
        self.layers = []
        for i in range(N):
            self.layers.append(SEConv(hidden_irreps,  # in
                                      hidden_irreps,  # hidden
                                      hidden_irreps,  # out
                                      attr_irreps,    # steerable attribute
                                      linear=linear,
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
        pos, x, edge_attr, node_attr, edge_index, batch = graph.pos, graph.x, graph.edge_attr, graph.node_attr, graph.edge_index, graph.batch
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        # The embedding layer
        x = self.embedding_layer(x, node_attr)

        # The main layers
        for layer in self.layers:
            x = layer(x, edge_index, additional_message_features, edge_attr, batch)

        # The output layers
        x = self.output_layer_1(x, node_attr)
        x = self.output_layer_2(x, node_attr)

        # The output is a difference vector
        return x


class SEConv(MessagePassing):
    """
        SE(3) equivariant convolutional layer.
    """

    def __init__(self, node_in_irreps, hidden_irreps, node_out_irreps, attr_irreps, linear, norm):
        super(SEConv, self).__init__(node_dim=-2, aggr="add")

        self.norm = norm
        self.linear = linear

        # The message network layers
        irreps_message_in = (node_in_irreps + node_in_irreps + Irreps("2x0e")).simplify()

        # Specifiy irreps for gating after the aggregation of messages
        irreps_g_scalars = Irreps(str(hidden_irreps[0]))
        irreps_g_gate = Irreps("{}x0e".format(hidden_irreps.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(hidden_irreps[1:]))
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        if self.linear:
            self.message_net_linear = O3TensorProduct(irreps_message_in,
                                                      irreps_g,
                                                      attr_irreps)

        else:
            self.message_net_nonlinear_1 = O3TensorProductSwishGate(irreps_message_in,
                                                                    hidden_irreps,
                                                                    attr_irreps)
            self.message_net_nonlinear_2 = O3TensorProduct(hidden_irreps,
                                                           irreps_g,
                                                           attr_irreps)

        # The node update layers is just a swish gate
        irreps_update_in = (node_in_irreps + hidden_irreps).simplify()
        self.update_layer = O3SwishGate(irreps_g_scalars, irreps_g_gate, irreps_g_gated)

        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(hidden_irreps)
            self.message_norm = BatchNorm(hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(hidden_irreps)

    def forward(self, x, edge_index, additional_message_features, edge_attr, batch):
        """ Propagate messages along edges """
        x = self.propagate(edge_index, x=x, additional_message_features=additional_message_features,
                           edge_attr=edge_attr)
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, additional_message_features, edge_attr):
        """ Create messages """
        if self.linear:
            message = self.message_net_linear(torch.cat((x_i, x_j, additional_message_features), dim=-1), edge_attr)
        else:
            message = self.message_net_nonlinear_1(
                torch.cat((x_i, x_j, additional_message_features), dim=-1), edge_attr)
            message = self.message_net_nonlinear_2(message, edge_attr)
        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x):
        """ Update note features """
        update = self.update_layer(message)
        x += update  # Residual connection
        return x
