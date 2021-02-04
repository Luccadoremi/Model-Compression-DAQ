from functools import partial as _apply_partial

import torch.nn as nn
import torch.nn.functional as F

from .quant_base import PQBase


def _p(a, b):
    return _apply_partial(b, a)

class PQEmbeddingSuper(nn.Embedding, PQBase):

    def __init__(self, num_embeddings, super_embed_dim, padding_idx, centroid_buckets, assignment_buckets, embedding_dim,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False, sparse=False, _weight=None, *args, **kwargs):
        self.buc_x, self.buc_y = centroid_buckets.shape[:2]
        self.bucket_num = self.buc_x * self.buc_y
        self.n_centroids = centroid_buckets[0][0].size(0)
        self.block_size = centroid_buckets[0][0].size(1)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.x_dim = num_embeddings
        self.out_dim = int(num_embeddings // self.buc_x)
        super(PQEmbeddingSuper, self).__init__(num_embeddings, super_embed_dim, padding_idx, *args, **kwargs)
        super(PQBase, self).__thisclass__.__init__(self, centroid_buckets, assignment_buckets)

        # Delete weight of super class since it doubles with quantization code
        del self.weight
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        # check compatibility
        if self.x_dim % self.block_size != 0:
            raise ValueError("Wrong PQ sizes (not divisible) (out:%d, block: %d)" % (self.x_dim, self.block_size))

        # the largest embed dim
        self.super_embed_dim = {'encoder': super_embed_dim, 'decoder': super_embed_dim}

        # the current sampled embed dim
        self.sample_embed_dim = {'encoder': None, 'decoder': None}

        self.samples = {'encoder': {}, 'decoder': {}}
        self.profiling = False
        self.reset_parameters()

    @property
    def weight(self):
        return super(PQEmbeddingSuper, self).weight()

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
        return self.sample_parameters(part)[part]['weight']

    def forward(self, input, part='encoder'):
        return F.embedding(input, self.sampled_weight(part), self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse)

    def extra_repr(self):
        s = '{num_embeddings}, {embedding_dim}'
        if self.padding_idx is not None:
            s += ', padding_idx={padding_idx}'
        if self.max_norm is not None:
            s += ', max_norm={max_norm}'
        if self.norm_type != 2:
            s += ', norm_type={norm_type}'
        if self.scale_grad_by_freq is not False:
            s += ', scale_grad_by_freq={scale_grad_by_freq}'
        if self.sparse is not False:
            s += ', sparse=True'
        if "n_centroids" in self.__dict__:
            s += ', n_centroids={n_centroids}'
        if "block_size" in self.__dict__:
            s += ', block_size={block_size}'
        return s.format(**self.__dict__)