import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps

from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate, O3SwishGate
from .instance_norm import InstanceNorm


class SEConv(nn.Module):
    """Steerable E(3) equivariant (non-linear) convolutional network"""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
        conv_type="linear",
    ):
        super().__init__()
        self.task = task
        # Create network, embedding first
        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEConvLayer(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                    conv_type=conv_type,
                )
            )
        self.layers = nn.ModuleList(layers)

        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # Embed
        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x


class SEConvLayer(MessagePassing):
    """E(3) equivariant (non-linear) convolutional layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
        conv_type="linear",
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps
        self.conv_type = conv_type

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        self.setup_gate(hidden_irreps)
        if self.conv_type == "linear":
            self.message_layer = O3TensorProduct(
                message_input_irreps, self.irreps_g, edge_attr_irreps
            )
        elif self.conv_type == "nonlinear":
            self.message_layer_1 = O3TensorProductSwishGate(
                message_input_irreps, hidden_irreps, edge_attr_irreps
            )
            self.message_layer_2 = O3TensorProduct(
                hidden_irreps, self.irreps_g, edge_attr_irreps
            )
        else:
            raise Exception("Invalid convolution type for SEConvLayer")

        self.setup_normalisation(norm)

    def setup_gate(self, hidden_irreps):
        """Add necessary scalar irreps for gate to output_irreps, similar to O3TensorProductSwishGate"""
        irreps_g_scalars = Irreps(str(hidden_irreps[0]))
        irreps_g_gate = Irreps(
            "{}x0e".format(hidden_irreps.num_irreps - irreps_g_scalars.num_irreps)
        )
        irreps_g_gated = Irreps(str(hidden_irreps[1:]))
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()
        self.gate = O3SwishGate(irreps_g_scalars, irreps_g_gate, irreps_g_gated)
        self.irreps_g = irreps_g

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.irreps_g)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""
        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        if self.conv_type == "linear":
            message = self.message_layer(input, edge_attr)
        elif self.conv_type == "nonlinear":
            message = self.message_layer_1(input, edge_attr)
            message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        update = self.gate(message)
        x += update  # Residual connection
        return x
