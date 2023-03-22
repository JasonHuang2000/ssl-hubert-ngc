#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import copy
import functools
import pickle
import warnings
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, List, Optional

from argparse import Namespace

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra._internal.utils import  get_args_parser
from hydra.core.utils import _flush_loggers, configure_log
from hydra.types import TaskFunction

from fairseq_cli.hydra_utils.utils import _run_hydra


import torch
from hydra.core.hydra_config import HydraConfig
from hydra._internal.hydra import Hydra
import omegaconf
from omegaconf import OmegaConf, open_dict, DictConfig, read_write

from fairseq import distributed_utils, metrics
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.initialize import add_defaults, hydra_init
from fairseq.dataclass.utils import omegaconf_no_object_check, convert_namespace_to_omegaconf
from fairseq.utils import reset_logging
from fairseq_cli.train import main as pre_main

logger = logging.getLogger("fairseq_cli.hydra_train")

def main(
    config_path: Optional[str] = None,
    config_name: Optional[str] = None,
    strict: Optional[str] = None,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: The config path, a directory relative to the declaring python file.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:

                args_parser = get_args_parser()
                args = Namespace(help=False, hydra_help=False, 
                    overrides=['task.data=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/manifest', 
                    'task.label_dir=/tmp2/willymen/librispeech/LibriSpeech/dev-clean/labels', 
                    'task.labels=["km"]', 'model.label_rate=100'], cfg=None, package=None, run=False, 
                    multirun=False, shell_completion=False, config_path=None, 
                    config_name='hubert_base_librispeech', 
                    config_dir='/tmp2/willymen/fairseq/examples/hubert/config/pretrain', 
                    info=False)
                
                # no return value from run_hydra() as it may sometime actually run the task_function
                # multiple times (--multirun)
                _run_hydra(
                    args_parser=args_parser,
                    task_function=task_function,
                    config_path=config_path,
                    config_name=config_name,
                    strict=strict,
                    args=args,
                )

        return decorated_main

    return main_decorator

fairseq_config = 0

@hydra.main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def hydra_main(cfg: FairseqConfig) -> float:
    _hydra_main(cfg)

@main(config_path=os.path.join("..", "fairseq", "config"), config_name="config")
def set_fairseq_config(cfg: FairseqConfig) -> float:
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)
    
    config = copy.deepcopy(cfg)
    global fairseq_config
    fairseq_config = config

def get_fairseq_config():
    return fairseq_config

def _hydra_main(cfg: FairseqConfig, **kwargs):
    add_defaults(cfg)

    if cfg.common.reset_logging:
        reset_logging()  # Hydra hijacks logging, fix that
    else:
        # check if directly called or called through hydra_main
        if HydraConfig.initialized():
            with open_dict(cfg):
                # make hydra logging work with ddp (see # see https://github.com/facebookresearch/hydra/issues/1126)
                cfg.job_logging_cfg = OmegaConf.to_container(
                    HydraConfig.get().job_logging, resolve=True
                )

    with omegaconf_no_object_check():
        cfg = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
        )
    OmegaConf.set_struct(cfg, True)

    try:
        if cfg.common.profile:
            with torch.cuda.profiler.profile():
                with torch.autograd.profiler.emit_nvtx():
                    distributed_utils.call_main(cfg, pre_main, **kwargs)
        else:
            distributed_utils.call_main(cfg, pre_main, **kwargs)
    except BaseException as e:
        if not cfg.common.suppress_crashes:
            raise
        else:
            logger.error("Crashed! " + str(e))

    # get best val and return - useful for sweepers
    try:
        best_val = metrics.get_smoothed_value(
            "valid", cfg.checkpoint.best_checkpoint_metric
        )
    except:
        best_val = None

    if best_val is None:
        best_val = float("inf")

    return best_val


def cli_main():
    try:
        from hydra._internal.utils import get_args

        cfg_name = get_args().config_name or "config"
    except:
        logger.warning("Failed to get config name from hydra args")
        cfg_name = "config"

    print(cfg_name)

    hydra_init(cfg_name)
    hydra_main()


if __name__ == "__main__":
    cli_main()
