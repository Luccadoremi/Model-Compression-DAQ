from functools import partial as _apply_partial

import torch.nn as nn
import torch.nn.functional as F

from .quant_base import PQBase


def _p(a, b):
    return _apply_partial(b, a)

class PQLinearSuper(nn.Module, PQBase):

    def __init__(self, super_in_dim, super_out_dim, bias=None, uniform_=None, non_linear='linear',
                 centroid_buckets=None, assignment_buckets=None, in_features=None, out_features=None):
        self.in_features = in_features
        self.out_features = out_features
        self.buc_x, self.buc_y = centroid_buckets.shape[:2]
        self.x_dim = out_features
        self.n_centroids = centroid_buckets[0][0].size(0)
        self.block_size = centroid_buckets[0][0].size(1)
        self.out_dim = int(out_features // self.buc_x)
        if self.x_dim % self.block_size != 0:
            raise ValueError("Wrong PQ sizes (not divisible) (out:%d, block: %d)" % (self.x_dim, self.block_size))

        super(PQLinearSuper, self).__init__()
        super(PQBase, self).__thisclass__.__init__(self, centroid_buckets, assignment_buckets, bias)
        # check compatibility

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    @property
    def weight(self):
        return super(PQLinearSuper, self).weight()


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
        return F.linear(x, self.samples['weight'], self.samples['bias'])

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def extra_repr(self):
        return f"in_features={self.in_features}, out_features={self.out_features}, n_centroids={self.n_centroids}, block_size={self.block_size}, bias={self.bias is not None}"

def sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    return sample_weight[:sample_out_dim, :]

def sample_bias(bias, sample_out_dim):
    return bias[:sample_out_dim]
