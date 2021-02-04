# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import emulate_int


class IntLinearSuper(nn.Linear):
    """
    Quantized counterpart of the nn.Linear module that applies QuantNoise during training.

    Args:
        - in_features: input features
        - out_features: output features
        - bias: bias or not
        - p: amount of noise to inject (0 = no quantization, 1 = quantize all the weights)
        - bits: number of bits
        - method: choose among {"tensor", "histogram", "channel"}
        - update_step: recompute scale and zero_point every update_steps iterations

    Remarks:
        - We use the straight-through estimator so that the gradients
          back-propagate nicely in the network, this is implemented with
          the detach() trick.
        - Parameters scale and zero_point are recomputed every update_step
          forward pass to reduce the overhead
        - At test time, the weights are fully quantized
    """

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', 
        p=0,
        update_step=3000,
        bits=8,
        method="histogram"):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        if self.bias is not None:
            self.samples['bias'] = sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        weight = self.samples['weight']

        p = self.p if self.training else 1
        if self.counter % self.update_step == 0:
            self.scale = None
            self.zero_point = None
        self.counter += 1
        weight_quantized, self.scale, self.zero_point = emulate_int(
            weight.detach(),
            bits=self.bits,
            method=self.method,
            scale=self.scale,
            zero_point=self.zero_point,
        )
        mask = torch.zeros_like(weight)
        mask.bernoulli_(1 - p)
        noise = (weight_quantized - weight).masked_fill(mask.bool(), 0)

        # using straight-through estimator (STE)
        clamp_low = - self.scale * self.zero_point
        clamp_high = self.scale * (2 ** self.bits - 1 - self.zero_point)
        weight = torch.clamp(weight, clamp_low.item(), clamp_high.item()) + noise.detach()
        return F.linear(x, weight, self.samples['bias'])

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias