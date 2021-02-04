import collections
import math
import random
import torch
import pdb

from fairseq import checkpoint_utils, distributed_utils, quantization_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter
from copy import deepcopy
from fairseq.modules.quantization import scalar

def main(args, init_distributed=False):
    utils.import_user_module(args)
    utils.handle_save_path(args)
    task = tasks.setup_task(args)
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    trainer = Trainer(args, task, model, criterion)
    # print("Saving Uncompressed...")
    # trainer.save_checkpoint(args.save_dir + "/uncompressed.pt", {})
    # print("Saved at: %s/uncompressed.pt" % args.save_dir)
    print("Starting Quantizing...")
    q = quantization_utils.Quantizer(args.quantization_config_path, 6, 0)
    q.dry_run(model, flag=False)
    print("Saving Compressed...")
    trainer.save_checkpoint(args.save_dir + "/compressed.pt", {})
    print("Saved at: %s/compressed.pt" % args.save_dir)
def cli_main():
    parser = options.get_training_parser()
    parser.add_argument('--train-subtransformer', action='store_true', default=False, help='whether train SuperTransformer or SubTransformer')
    parser.add_argument('--sub-configs', required=False, is_config_file=True, help='when training SubTransformer, use --configs to specify architecture and --sub-configs to specify other settings')

    # for profiling
    parser.add_argument('--profile-flops', action='store_true', help='measure the FLOPs of a SubTransformer')

    parser.add_argument('--latgpu', action='store_true', help='measure SubTransformer latency on GPU')
    parser.add_argument('--latcpu', action='store_true', help='measure SubTransformer latency on CPU')
    parser.add_argument('--latiter', type=int, default=300, help='how many iterations to run when measure the latency')
    parser.add_argument('--latsilent', action='store_true', help='keep silent when measure latency')

    parser.add_argument('--validate-subtransformer', action='store_true', help='evaluate the SubTransformer on the validation set')

    options.add_generation_args(parser)

    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
    