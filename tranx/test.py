# coding=utf-8

import argparse
import os

import six.moves.cPickle as pickle
import torch

import common.cli_logger as cli_logger
import evaluation
from common.registerable import Registrable
from common.utils import dump_cfg, init_cfg
from components.dataset import Dataset


def prologue(cfg: argparse.Namespace, *varargs) -> None:
    # sanity checks
    assert cfg.load_model not in [None, ""]
    assert cfg.exp_name not in [None, ""]
    assert not cfg.cuda or (cfg.cuda and torch.cuda.is_available())

    # dirs
    base_dir = f"./experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)

    dump_cfg(f"{base_dir}/test_config.txt", vars(cfg))


def epilogue(cfg: argparse.Namespace, *varargs) -> None:
    pass


def test(cfg: argparse.Namespace):
    cli_logger.info("=== Testing ===")
    prologue(cfg)

    test_set = Dataset.from_bin_file(cfg.test_file)

    cli_logger.info('load model from [%s]' % cfg.load_model)
    params = torch.load(cfg.load_model, map_location=lambda storage, loc: storage)

    transition_system = params['transition_system']
    saved_args = params['args']
    saved_args.cuda = cfg.cuda
    # set the correct domain from saved arg
    cfg.lang = saved_args.lang

    parser_cls = Registrable.by_name(cfg.parser)
    parser = parser_cls.load(model_path=cfg.load_model, cuda=cfg.cuda)
    parser.eval()

    evaluator = Registrable.by_name(cfg.evaluator)(transition_system, args=cfg)
    eval_results, decode_results = evaluation.evaluate(
        test_set.examples, parser, evaluator, cfg,
        verbose=cfg.verbose, return_decode_result=True
    )

    cli_logger.info(eval_results)

    if cfg.save_decode_to:
        pickle.dump(decode_results, open(cfg.save_decode_to, 'wb'))

    epilogue(cfg)


if __name__ == '__main__':
    test(cfg=init_cfg("./configs/conala.yaml", do_seed=True))
