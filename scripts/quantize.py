#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.meters import StopwatchMeter, TimeMeter
import sys
import pdb
import numpy as np
import os

from fairseq.modules.quantization import scalar
from fairseq.models.transformer import TransformerModel


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

def quantize_model(model):
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
       
        else:
            if is_master_process:
                logging.info(f"Module {module} not yet supported for quantization")
            continue

        # activation quantization
        a_q = ActivationQuantizer(quantized_module, p=0, bits=bits, method="histogram")

        # replace layer by its quantized counterpart
        attrsetter(layer)(model, quantized_module)
def main(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

    #use_cuda = torch.cuda.is_available() and not args.cpu
    use_cuda = False

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    print('| loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )
    # args.arch= "transformer"
    # model = task.build_model(args)

    #nmodel = task.build_model(args,task)
    scalar.quantize_model_(models[0])
    #import pdb; pdb.set_trace()
    torch.manual_seed(args.seed)

    # Optimize ensemble for generation
    for model in models:
        if use_cuda:
            model.cuda()

        config = utils.get_subtransformer_config(args)

        model.set_sample_config(config)
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()
        print(model, file=sys.stderr)
        print(args.path, file=sys.stderr)

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    num_sentences = 0
    has_target = True
    decoder_times_all = []
    input_len_all = []
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos, decoder_times = task.inference_step(generator, models, sample, prefix_tokens)
            input_len_all.append(np.mean(sample['net_input']['src_lengths'].cpu().numpy()))

            print(decoder_times)
            decoder_times_all.append(decoder_times)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str))
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str))

                # Process top predictions
                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'].int().cpu() if hypo['alignment'] is not None else None,
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        print('H-{}\t{}\t{}'.format(sample_id, hypo['score'], hypo_str))
                        print('P-{}\t{}'.format(
                            sample_id,
                            ' '.join(map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
                            ))
                        ))

                        if args.print_alignment:
                            print('A-{}\t{}'.format(
                                sample_id,
                                ' '.join(map(lambda x: str(utils.item(x)), alignment))
                            ))

            wps_meter.update(num_generated_tokens)
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument('--encoder-embed-dim-subtransformer', type=int, help='subtransformer encoder embedding dimension',
                        default=None)
    parser.add_argument('--decoder-embed-dim-subtransformer', type=int, help='subtransformer decoder embedding dimension',
                        default=None)

    parser.add_argument('--encoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-ffn-embed-dim-all-subtransformer', nargs='+', default=None, type=int)

    parser.add_argument('--encoder-layer-num-subtransformer', type=int, help='subtransformer num encoder layers')
    parser.add_argument('--decoder-layer-num-subtransformer', type=int, help='subtransformer num decoder layers')

    parser.add_argument('--encoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-self-attention-heads-all-subtransformer', nargs='+', default=None, type=int)
    parser.add_argument('--decoder-ende-attention-heads-all-subtransformer', nargs='+', default=None, type=int)

    parser.add_argument('--decoder-arbitrary-ende-attn-all-subtransformer', nargs='+', default=None, type=int)

    args = options.parse_args_and_arch(parser)

    if args.pdb:
        pdb.set_trace()

    main(args)


if __name__ == '__main__':
    cli_main()
