import torch
from math import sqrt

# All initialisations are adapted from the Pytorch nn.init module.


def get_fans(tp):
    """ Get fan_in and fan_out with corresponding slices """
    slices_fan_in = {}  # fan_in per slice
    slices_fan_out = {}
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        mul_1, mul_2, mul_out = weight.shape
        fan_in = mul_1 * mul_2
        fan_out = mul_out
        slices_fan_in[slice_idx] = (slices_fan_in[slice_idx] +
                                    fan_in if slice_idx in slices_fan_in.keys() else fan_in)
        slices_fan_out[slice_idx] = fan_out
    return slices_fan_in, slices_fan_out


@torch.no_grad()
def xavier_uniform(tp, gain=1):
    slices_fan_in, slices_fan_out = get_fans(tp)
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        a = gain * sqrt(6 / (slices_fan_in[slice_idx] + slices_fan_out[slice_idx]))
        weight.data.uniform_(-a, a)


@torch.no_grad()
def xavier_normal(tp, gain=1):
    slices_fan_in, slices_fan_out = get_fans(tp)
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        a = gain * sqrt(2 / (slices_fan_in[slice_idx] + slices_fan_out[slice_idx]))
        weight.data.normal_(-a, a)


def glorot(tp):
    xavier_uniform(tp)


@torch.no_grad()
def kaiming_uniform(tp, gain=1):
    slices_fan_in, _ = get_fans(tp)
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        a = gain * sqrt(3 / slices_fan_in[slice_idx])
        weight.data.uniform_(-a, a)


@torch.no_grad()
def erik(tp, gain=1):
    slices_fan_in, _ = get_fans(tp)
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        a = gain * sqrt(1 / slices_fan_in[slice_idx])
        weight.data.uniform_(-a, a)


@torch.no_grad()
def bias_erik(biases, tp):
    slices_fan_in, _ = get_fans(tp)
    i = 0
    for instr in tp.instructions:
        slice_idx = instr[2]
        if tp.irreps_out[slice_idx][1][0] == 0:
            a = sqrt(1 / slices_fan_in[slice_idx])
            biases[i].uniform_(-a, a)


@torch.no_grad()
def kaiming_normal(tp, gain=1):
    slices_fan_in, _ = get_fans(tp)
    for weight, instr in zip(tp.weight_views(), tp.instructions):
        slice_idx = instr[2]
        a = gain / sqrt(slices_fan_in[slice_idx])
        weight.data.normal_(0, a)
