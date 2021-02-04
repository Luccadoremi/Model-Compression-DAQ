import contextlib
from io import StringIO
import logging
import os
import random
import tempfile
import unittest

import torch

from fairseq import options
from fairseq import trainer as train
from fairseq_cli import preprocess
# from tests.utils import (
#     create_dummy_data,
#     preprocess_lm_data,
#     preprocess_translation_data,
#     train_translation_model,
#     generate_main,
# )

def create_dummy_data(data_dir, num_examples=100, maxlen=20, alignment=False):
    def _create_dummy_data(filename):
        data = torch.rand(num_examples * maxlen)
        data = 97 + torch.floor(26 * data).int()
        with open(os.path.join(data_dir, filename), 'w') as h:
            offset = 0
            for _ in range(num_examples):
                ex_len = random.randint(1, maxlen)
                ex_str = ' '.join(map(chr, data[offset:offset+ex_len]))
                print(ex_str, file=h)
                offset += ex_len

    def _create_dummy_alignment_data(filename_src, filename_tgt, filename):
        with open(os.path.join(data_dir, filename_src), 'r') as src_f, \
             open(os.path.join(data_dir, filename_tgt), 'r') as tgt_f, \
             open(os.path.join(data_dir, filename), 'w') as h:
            for src, tgt in zip(src_f, tgt_f):
                src_len = len(src.split())
                tgt_len = len(tgt.split())
                avg_len = (src_len + tgt_len) // 2
                num_alignments = random.randint(avg_len // 2, 2 * avg_len)
                src_indices = torch.floor(torch.rand(num_alignments) * src_len).int()
                tgt_indices = torch.floor(torch.rand(num_alignments) * tgt_len).int()
                ex_str = ' '.join(["{}-{}".format(src, tgt) for src, tgt in zip(src_indices, tgt_indices)])
                print(ex_str, file=h)

    _create_dummy_data('train.in')
    _create_dummy_data('train.out')
    _create_dummy_data('valid.in')
    _create_dummy_data('valid.out')
    _create_dummy_data('test.in')
    _create_dummy_data('test.out')

    if alignment:
        _create_dummy_alignment_data('train.in', 'train.out', 'train.align')
        _create_dummy_alignment_data('valid.in', 'valid.out', 'valid.align')
        _create_dummy_alignment_data('test.in', 'test.out', 'test.align')

def preprocess_lm_data(data_dir):
    preprocess_parser = options.get_preprocessing_parser()
    preprocess_args = preprocess_parser.parse_args([
        '--only-source',
        '--trainpref', os.path.join(data_dir, 'train.out'),
        '--validpref', os.path.join(data_dir, 'valid.out'),
        '--testpref', os.path.join(data_dir, 'test.out'),
        '--destdir', data_dir,
    ])
    preprocess.main(preprocess_args)

def quantize_language_model(data_dir, arch, extra_flags=None, run_validation=False):
    train_parser = options.get_training_parser()
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--task', 'language_modeling',
            data_dir,
            '--arch', arch,
            '--optimizer', 'adam',
            '--lr', '0.0001',
            '--criterion', 'adaptive_loss',
            '--adaptive-softmax-cutoff', '5,10,15',
            '--max-tokens', '500',
            '--tokens-per-sample', '500',
            '--save-dir', data_dir,
            '--max-epoch', '1',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--ddp-backend', 'no_c10d',
            '--num-workers', 0,
        ] + (extra_flags or []),
    )
    train.main(train_args)

    # try scalar quantization
    scalar_quant_train_parser = options.get_training_parser()
    scalar_quant_train_args = options.parse_args_and_arch(
        scalar_quant_train_parser,
        [
            '--task', 'language_modeling',
            data_dir,
            '--arch', arch,
            '--optimizer', 'adam',
            '--lr', '0.0001',
            '--criterion', 'adaptive_loss',
            '--adaptive-softmax-cutoff', '5,10,15',
            '--max-tokens', '500',
            '--tokens-per-sample', '500',
            '--save-dir', data_dir,
            '--max-update', '3',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--ddp-backend', 'no_c10d',
            '--num-workers', 0,
            '--quant-noise-scalar', '0.5',
        ] + (extra_flags or []),
    )
    train.main(scalar_quant_train_args)

    # try iterative PQ quantization
    quantize_parser = options.get_training_parser()
    quantize_args = options.parse_args_and_arch(
        quantize_parser,
        [
            '--task', 'language_modeling',
            data_dir,
            '--arch', arch,
            '--optimizer', 'adam',
            '--lr', '0.0001',
            '--criterion', 'adaptive_loss',
            '--adaptive-softmax-cutoff', '5,10,15',
            '--max-tokens', '50',
            '--tokens-per-sample', '50',
            '--max-update', '6',
            '--no-progress-bar',
            '--distributed-world-size', '1',
            '--ddp-backend', 'no_c10d',
            '--num-workers', 0,
            '--restore-file', os.path.join(data_dir, 'checkpoint_last.pt'),
            '--reset-optimizer',
            '--quantization-config-path', os.path.join(os.path.dirname(__file__), 'transformer_quantization_config.yaml'),
        ] + (extra_flags or []),
    )
 #   train.main(quantize_args)


class TestQuantization(unittest.TestCase):
    def setUp(self):
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        logging.disable(logging.NOTSET)

    @unittest.skipIf(not torch.cuda.is_available(), 'test requires a GPU')
    def test_quantization(self):
        with contextlib.redirect_stdout(StringIO()):
            with tempfile.TemporaryDirectory('test_quantization') as data_dir:
                create_dummy_data(data_dir)
                preprocess_lm_data(data_dir)
                # tests both scalar and iterative PQ quantization
                quantize_language_model(data_dir, 'transformer_lm')

if __name__ == "__main__":
    unittest.main()
