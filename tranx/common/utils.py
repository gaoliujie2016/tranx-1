# coding=utf-8

import argparse

import numpy as np
import torch
import yaml


def init_cfg(yaml_file: str, do_seed=True) -> argparse.Namespace:
    try:
        cfg = argparse.Namespace(**yaml.load(open(yaml_file, 'rt')))
    except:
        raise RuntimeError(f'Error while parsing {yaml_file} config. Exiting...')

    # seed the RNG
    if do_seed:
        torch.manual_seed(cfg.seed)
        if cfg.cuda: torch.cuda.manual_seed(cfg.seed)
        np.random.seed(int(cfg.seed * 13 / 7))

    return cfg


def dump_cfg(file: str, cfg: dict) -> None:
    fp = open(file, "wt")
    for k, v in cfg.items():
        fp.write("%32s: %s\n" % (k, v))
    fp.close()


class CachedProperty(object):
    """
    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value
