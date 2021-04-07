#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.

We basically copy this file from `fairseq.generate.py` and
change this file a little bit to generate attention scores and other stuff.
"""

import ast
import logging
import os
import sys
from itertools import chain

import numpy as np
import torch
from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.logging import progress_bar


def main(args):
    assert args.path is not None, "--path required for generation!"
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"

    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok=True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering=1, encoding="utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def _main(args, output_file):
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        stream=output_file,
    )
    logger = logging.getLogger("fairseq_cli.generate")

    utils.import_user_module(args)

    if args.max_tokens is None and args.batch_size is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=overrides,
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
        strict=(args.checkpoint_shard_count == 1),
        num_shards=args.checkpoint_shard_count,
    )

    if args.lm_path is not None:
        overrides["data"] = args.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({args.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
        data_buffer_size=args.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=("tqdm" if not args.no_progress_bar else "none"),
    )

    output = []
    for sample in progress:
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        if "net_input" not in sample:
            continue

        loss, _ = task.inference_step(
            models,
            sample
        )
        print(loss)
        output.append(loss)
    return output

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--print-attention", action="store_true", help="print attention matrix as jsonline")
    args = options.parse_args_and_arch(parser)
    main(args)

if __name__ == "__main__":
    cli_main()