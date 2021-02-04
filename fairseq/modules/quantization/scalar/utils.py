# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from operator import attrgetter

import torch.nn as nn
import torch.distributed as dist

from ..pq.utils import get_layers, attrsetter
from .modules import IntConv2d, IntLinear, IntEmbedding, ActivationQuantizer, IntEmbeddingSuper, IntLinearSuper
from fairseq.modules.embedding_super import EmbeddingSuper
from fairseq.modules.linear_super import LinearSuper

MAPPING = {nn.Linear: IntLinear, nn.Embedding: IntEmbedding, nn.Conv2d: IntConv2d, EmbeddingSuper: IntEmbeddingSuper, LinearSuper: IntLinearSuper}

from fairseq.modules.quantization.scalar import ops

import torch

def quantize_weights(model, p=0.2, bits=8, update_step=3000):
    # quantize all layers
    assert False, "Scalar Quantization is disabled!"
    quantized_layers = get_layers(model, "(.*?)")

    for layer in quantized_layers:
        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)

        # recover module
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}")

        # instantiate the quantized counterpart
        if isinstance(module, tuple(MAPPING.values())):
            _, scale, zero_point = ops.emulate_int(module.weight.detach(),bits=bits,method="histogram")
            # q_w = torch.clamp(torch.round(module.weight / scale + zero_point), 0, 255)
            # q_w = q_w.type(torch.ByteTensor)
            q_w = torch.quantize_per_tensor(module.weight, scale[0].item(), int(zero_point[0].item()), torch.quint8)
            # if(hasattr(module, 'bias')):
            #     # q_b = torch.clamp(torch.round(module.bias / scale + zero_point), 0, 255)
            #     # q_b = q_b.type(torch.ByteTensor)
            #     q_b = torch.quantize_per_tensor(module.bias, scale[0].item(), int(zero_point[0].item()), torch.quint8)
            #     module.bias = torch.nn.Parameter(q_b, False)
            #import pdb; pdb.set_trace()
            module.weight = torch.nn.Parameter(q_w, False)

        else:
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

    # return name of quantized layers
    return quantized_layers
def dequantize_weights(model, p=0.2, bits=8, update_step=3000):
    # quantize all layers
    quantized_layers = get_layers(model, "(.*?)")

    for layer in quantized_layers:
        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)

        # recover module
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}")

        # instantiate the quantized counterpart
        if isinstance(module, tuple(MAPPING.values())):
            dq_w = torch.dequantize(module.weight)
            module.weight = torch.nn.Parameter(dq_w, False)

        else:
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

    # return name of quantized layers
    return quantized_layers
def quantize_model_(model, p=0.2, bits=8, update_step=3000):
    """
    Replaces all modules with their scalar quantized counterpart and
    registers hooks to quantize the post-ativations of those modules.

    Args:
        - model: a nn.Module
        - p: amount of noise (0 for no noise, 1 to quantize all the weights/activations)
        - bits: number of bits
        - update_step: update quantization parameters every update_step steps
    """

    # quantize all layers
    quantized_layers = get_layers(model, "(.*?)")

    for layer in quantized_layers:
        # book-keeping
        is_master_process = (not dist.is_initialized()) or (dist.is_initialized() and dist.get_rank() == 0)

        # recover module
        module = attrgetter(layer)(model)
        if is_master_process:
            logging.info(f"Quantizing layer {layer} with bits={bits} and QuantNoise={p}")

        # quantization params
        q_params = {"p": p, "update_step": update_step, "bits": bits, "method": "histogram", "counter": 0}

        # instantiate the quantized counterpart
        if isinstance(module, tuple(MAPPING.keys())):
            QuantizedModule = MAPPING[module.__class__]
            quantized_module = QuantizedModule.__new__(QuantizedModule)
            params = module.__dict__
            params.update(q_params)
            quantized_module.__dict__.update(params)
            # weight_quantized, scale, zero_point = ops.emulate_int(quantized_module.weight.detach(),bits=8,method="histogram")
            # torch.clamp(torch.round(weight_quantized / scale + zero_point), 0, 255)

        else:
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

        # activation quantization
        a_q = ActivationQuantizer(quantized_module, p=0, bits=bits, method="histogram")

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_module)

    # return name of quantized layers
    return quantized_layers
