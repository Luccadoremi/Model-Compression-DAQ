# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import emulate_int


class IntEmbeddingSuper(nn.Embedding):

    def __init__(self, num_embeddings, super_embed_dim, padding_idx, 
        p=0,
        update_step=1000,
        bits=8,
        method="histogram", *args, **kwargs):
        super().__init__(num_embeddings, super_embed_dim, padding_idx, *args, **kwargs)

        # the largest embed dim
        self.super_embed_dim = {'encoder': super_embed_dim, 'decoder': super_embed_dim}

        # the current sampled embed dim
        self.sample_embed_dim = {'encoder': None, 'decoder': None}

        self.samples = {'encoder': {}, 'decoder': {}}
        self.profiling = False
        self.p = p
        self.bits = bits
        self.method = method
        self.update_step = update_step
        self.counter = 0
        self.reset_parameters()

    def profile(self, mode=True):
        self.profiling = mode

    def reset_parameters(self):
        super().reset_parameters()
        nn.init.normal_(self.weight, mean=0, std=self.embedding_dim ** -0.5)
        nn.init.constant_(self.weight[self.padding_idx], 0)

    def set_sample_config(self, sample_embed_dim, part):
        self.sample_embed_dim[part] = sample_embed_dim
        self._sample_parameters(part)

    def _sample_parameters(self, part):
        weight = self.weight[..., :self.sample_embed_dim[part]]
        self.samples[part]['weight'] = weight

        return self.samples

    def sample_parameters(self, part, resample=False):
        return self._sample_parameters(part) if self.profiling or resample else self.samples

    def sampled_weight(self, part):
        weight = self.sample_parameters(part)[part]['weight']
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

        return weight

    def forward(self, input, part='encoder'):
        return F.embedding(input, self.sampled_weight(part), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)
