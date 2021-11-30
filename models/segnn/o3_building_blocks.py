import torch
import torch.nn as nn
from math import sqrt

import e3nn.o3 as o3
from e3nn.o3 import Irreps, Linear, spherical_harmonics, TensorProduct, FullyConnectedTensorProduct
from e3nn.nn import Gate

import models.segnn.initialisation as tp_init


class UnnormalisedFullyConnectedTensorProduct(TensorProduct):
    """
    Unnormalised TensorProduct. See e3nn for documentation. e3nn initialises weights U(-1, 1) and
    normalises using constants based on the fan_in and number of paths. We would like to remove this
    normalisation and initialise weights using standard strategies.
    """

    def __init__(
        self,
        irreps_in1,
        irreps_in2,
        irreps_out,
        init=None,
        **kwargs
    ):
        irreps_in1 = o3.Irreps(irreps_in1)
        irreps_in2 = o3.Irreps(irreps_in2)
        irreps_out = o3.Irreps(irreps_out)

        instr = [
            (i_1, i_2, i_out, 'uvw', True, 1.0)
            for i_1, (mul1, ir_1) in enumerate(irreps_in1)
            for i_2, (mul2, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]

        # Undo e3nn normalisation by setting path_weight so it negates alpha. This assumes
        # the in_var and out_var are set to 1. Might break in the new normalisation of e3nn.
        # The normalisation is done in the _codegen.py file.
        unnormalised_instr = []

        for ins in instr:

            if init == "erik":
                alpha = sum([irreps_in1[ins[0]].mul*irreps_in2[ins[1]].mul for i in instr if i[2] == ins[2]])
            else:
                # Undo e3nns path normalisation.
                mul_ir_in1 = irreps_in1[ins[0]]
                mul_ir_in2 = irreps_in2[ins[1]]
                alpha = (mul_ir_in1.mul*mul_ir_in2.mul)  # No sqrt, since that happens in _codegen.py
                alpha *= len([i for i in instr if i[2] == ins[2]])

            ins = list(ins)
            ins[5] = alpha
            unnormalised_instr.append(tuple(ins))
        super().__init__(irreps_in1, irreps_in2, irreps_out, unnormalised_instr,
                         irrep_normalization="component", path_normalization="path", **kwargs)


class O3TensorProduct(torch.nn.Module):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, init="kaiming_uniform") -> None:
        super().__init__()
        self.irreps_in1 = irreps_in1
        self.irreps_out = irreps_out
        # Init irreps_in2
        if irreps_in2 == None:
            self.irreps_in2_provided = False
            self.irreps_in2 = Irreps("1x0e")
        else:
            self.irreps_in2_provided = True
            self.irreps_in2 = irreps_in2

        # Build the layers
        if init == "e3nn":
            self.tp = FullyConnectedTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out, shared_weights=True, irrep_normalization='component', path_normalization="element")
        else:
            self.tp = UnnormalisedFullyConnectedTensorProduct(
                irreps_in1=self.irreps_in1,
                irreps_in2=self.irreps_in2,
                irreps_out=self.irreps_out, shared_weights=True, normalization='component', init=init)  # We still use component normalisation.

        self.bias_slices = []
        biases = []
        for irrep, slice in zip(irreps_out, irreps_out.slices()):
            if irrep[1][0] == 0:  # only scalars
                biases.append(nn.Parameter(torch.zeros(irrep.dim)))
                self.bias_slices.append(slice)
        self.biases = nn.ParameterList(biases)

        # Initialize tensor product
        if init in ["glorot", "xavier_uniform"]:
            tp_init.xavier_uniform(self.tp)
        elif init == "kaiming_uniform":
            tp_init.kaiming_uniform(self.tp)
        elif init == "erik":
            tp_init.erik(self.tp)
            tp_init.bias_erik(self.biases, self.tp)
        elif init == "kaiming_normal":
            tp_init.kaiming_normal(self.tp)
        elif init == "e3nn":
            pass
        else:
            raise Exception("Unknown initialisation method for O3TensorProduct")

    def forward_with_bias(self, data_in1, data_in2=None) -> torch.Tensor:
        if data_in2 == None:
            data_in2 = torch.ones_like(data_in1[:, 0:1])
        data_out = self.tp(data_in1, data_in2)

        # Add the biases
        for (slice, bias) in zip(self.bias_slices, self.biases):
            data_out[:, slice] += bias
        return data_out

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_with_bias(data_in1, data_in2)
        return data_out


class O3TensorProductSwishGate(O3TensorProduct):
    def __init__(self, irreps_in1, irreps_out, irreps_in2=None, init="kaiming_uniform") -> None:
        # For the gate the output of the linear needs to have an extra number of scalar irreps equal to the amount of
        # non scalar irreps:
        # The first type is assumed to be scalar and passed through the activation
        irreps_g_scalars = Irreps(str(irreps_out[0]))
        # The remaining types are gated
        irreps_g_gate = Irreps("{}x0e".format(irreps_out.num_irreps - irreps_g_scalars.num_irreps))
        irreps_g_gated = Irreps(str(irreps_out[1:]))
        # So the gate needs the following irrep as input, this is the output irrep of the tensor product
        irreps_g = (irreps_g_scalars + irreps_g_gate + irreps_g_gated).simplify()

        # Build the layers
        super(O3TensorProductSwishGate, self).__init__(irreps_in1, irreps_g, irreps_in2, init)
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in1, data_in2=None) -> torch.Tensor:
        # Apply the tensor product, the rescaling and the bias
        data_out = self.forward_with_bias(data_in1, data_in2)
        # Apply the gate
        data_out = self.gate(data_out)
        # Return result
        return data_out


class O3SwishGate(torch.nn.Module):
    def __init__(self, irreps_g_scalars, irreps_g_gate, irreps_g_gated) -> None:
        super().__init__()
        if irreps_g_gated.num_irreps > 0:
            self.gate = Gate(irreps_g_scalars, [nn.SiLU()], irreps_g_gate, [torch.sigmoid], irreps_g_gated)
        else:
            self.gate = nn.SiLU()

    def forward(self, data_in) -> torch.Tensor:
        data_out = self.gate(data_in)
        return data_out
