# coding=utf-8

import argparse
import os
import sys

import pickle
import torch
from pprint import pprint

import common.cli_logger as cli_logger
import evaluation
from common.registerable import Registrable
from common.utils import dump_cfg, init_cfg
from components.dataset import Dataset
import model.parser

# Evaluators
import datasets.conala.conala_evaluator as conala_evaluator
import datasets.django.django_evaluator as django_evaluator


def prologue(cfg: argparse.Namespace, *varargs) -> str:
    # sanity checks
    assert cfg.load_model not in [None, ""]
    assert cfg.exp_name not in [None, ""]
    assert not cfg.cuda or (cfg.cuda and torch.cuda.is_available())
    assert cfg.mode == "test"

    # dirs
    base_dir = f"./experiments/{cfg.exp_name}"

    os.makedirs(f"{base_dir}/out", exist_ok=True)

    dump_cfg(f"{base_dir}/test_config.txt", vars(cfg))

    return base_dir


def epilogue(cfg: argparse.Namespace, *varargs) -> None:
    pass


def test(cfg: argparse.Namespace):
    cli_logger.info("=== Testing ===")
    experiment_base_dir = prologue(cfg)

    test_set = Dataset.from_bin_file(cfg.test_file)
    cli_logger.info(f"Loaded test file from [{cfg.test_file}]")

    params = torch.load(cfg.load_model, map_location=lambda storage, loc: storage)
    cli_logger.info(f"Loaded model from [{cfg.load_model}]")

    transition_system = params['transition_system']
    cli_logger.info(f"Loaded transition system [{type(transition_system)}]")

    saved_args: argparse.Namespace = params['args']
    saved_args.cuda = cfg.cuda
    # FIXME ?? set the correct domain from saved arg
    cfg.lang = saved_args.lang

    dump_cfg(experiment_base_dir + "/was_trained_with.txt", cfg=saved_args.__dict__)

    parser_cls = Registrable.by_name(cfg.parser)
    parser = parser_cls.load(model_path=cfg.load_model, cuda=cfg.cuda)
    parser.eval()
    cli_logger.info(f"Loaded parser model [{cfg.parser}]")

    evaluator = Registrable.by_name(cfg.evaluator)(transition_system, args=cfg)
    cli_logger.info(f"Loaded evaluator [{cfg.evaluator}]")

    # Do the evaluation
    eval_results, decoded_results = evaluation.evaluate(
        examples=test_set.examples, model=parser, evaluator=evaluator, args=cfg, verbose=cfg.verbose, return_decoded_result=True
    )

    cli_logger.info(eval_results)

    if cfg.save_decode_to:
        pickle.dump(decoded_results, open(cfg.save_decode_to, 'wb'))
        cli_logger.info(f"Saved decoded results to [{cfg.save_decode_to}]")

    epilogue(cfg)
    cli_logger.info("=== Done ===")


if __name__ == '__main__':
    test(cfg=init_cfg("./configs/django.yaml", do_seed=True))
