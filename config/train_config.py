import argparse
import os

from .base import PACKAGE_ROOT
from .task_configs import get_task_choices


MODEL_TYPE = "rsmil_stage1"
REQUIRED_FEAT_MAG = "hr"
DEFAULT_TASK = "task_tls_vs_notls_coad335"
DEFAULT_RESULTS_DIR = os.path.join(PACKAGE_ROOT, "results")
TRAIN_SCRIPT_DESCRIPTION = f"Standalone training entry for {MODEL_TYPE}"


CLI_ARGUMENTS = [
    {
        "flags": ("--feat_root_dir",),
        "kwargs": {"type": str, "required": True, "help": "feature root or direct hr feature directory"},
    },
    {
        "flags": ("--feat_mag",),
        "kwargs": {"type": str, "default": REQUIRED_FEAT_MAG, "choices": ["lr", "hr"], "help": "RSMIL must use hr"},
    },
    {
        "flags": ("--task",),
        "kwargs": {
            "type": str,
            "default": DEFAULT_TASK,
            "choices": get_task_choices(),
            "help": "dataset task preset",
        },
    },
    {
        "flags": ("--dataset_csv_path",),
        "kwargs": {"type": str, "default": None, "help": "optional override for the dataset csv"},
    },
    {
        "flags": ("--split_dir",),
        "kwargs": {"type": str, "default": None, "help": "optional override for the split directory"},
    },
    {
        "flags": ("--results_dir",),
        "kwargs": {"type": str, "default": DEFAULT_RESULTS_DIR, "help": "results directory"},
    },
    {
        "flags": ("--exp_code",),
        "kwargs": {"type": str, "default": MODEL_TYPE, "help": "experiment code"},
    },
    {"flags": ("--seed",), "kwargs": {"type": int, "default": 1, "help": "random seed"}},
    {"flags": ("--k",), "kwargs": {"type": int, "default": 10, "help": "number of folds"}},
    {"flags": ("--k_start",), "kwargs": {"type": int, "default": -1, "help": "start fold index"}},
    {"flags": ("--k_end",), "kwargs": {"type": int, "default": -1, "help": "end fold index"}},
    {"flags": ("--max_epochs",), "kwargs": {"type": int, "default": 10, "help": "maximum number of epochs"}},
    {"flags": ("--lr",), "kwargs": {"type": float, "default": 1e-4, "help": "learning rate"}},
    {"flags": ("--reg",), "kwargs": {"type": float, "default": 1e-5, "help": "weight decay"}},
    {"flags": ("--lambda_reg",), "kwargs": {"type": float, "default": 1e-5, "help": "L1 regularization strength"}},
    {"flags": ("--reg_type",), "kwargs": {"type": str, "default": "TLS", "help": "regularization mode"}},
    {"flags": ("--opt",), "kwargs": {"type": str, "default": "adam", "choices": ["adam", "sgd"], "help": "optimizer"}},
    {"flags": ("--drop_out",), "kwargs": {"type": float, "default": 0.25, "help": "dropout"}},
    {
        "flags": ("--weighted_sample",),
        "kwargs": {"action": "store_true", "default": False, "help": "use weighted sampler"},
    },
    {
        "flags": ("--log_data",),
        "kwargs": {"action": "store_true", "default": False, "help": "enable tensorboardX logging"},
    },
    {
        "flags": ("--testing",),
        "kwargs": {"action": "store_true", "default": False, "help": "use 10 percent subset for debugging"},
    },
    {
        "flags": ("--early_stopping",),
        "kwargs": {"action": "store_true", "default": False, "help": "enable early stopping"},
    },
    {"flags": ("--bag_loss",), "kwargs": {"type": str, "default": "ce", "choices": ["ce"], "help": "slide-level loss"}},
    {"flags": ("--in_dim",), "kwargs": {"type": int, "default": 1024, "help": "input feature dimension"}},
    {
        "flags": ("--embed_dim",),
        "kwargs": {"type": int, "default": 1024, "help": "kept for compatibility with the original script"},
    },
    {"flags": ("--rsmil_inner_dim",), "kwargs": {"type": int, "default": 512, "help": "hidden dimension inside RSMIL"}},
    {
        "flags": ("--rsmil_attn_dim",),
        "kwargs": {"type": int, "default": 128, "help": "attention hidden dimension inside RSMIL"},
    },
    {"flags": ("--rsmil_n_token_1",), "kwargs": {"type": int, "default": 1, "help": "stage-1 token count"}},
    {"flags": ("--rsmil_n_token_2",), "kwargs": {"type": int, "default": 6, "help": "stage-2 token count"}},
    {
        "flags": ("--rsmil_n_masked_patch_2",),
        "kwargs": {"type": int, "default": 20, "help": "stage-2 masked patch count"},
    },
    {"flags": ("--rsmil_mask_drop",), "kwargs": {"type": float, "default": 0.4, "help": "random masking ratio"}},
    {
        "flags": ("--rsmil_dim_reduction",),
        "kwargs": {"action": "store_true", "default": False, "help": "enable RSMIL dim reduction"},
    },
    {
        "flags": ("--rsmil_mixer_embed_dim",),
        "kwargs": {"type": int, "default": 256, "help": "RSMIL mixer embedding dimension"},
    },
    {
        "flags": ("--rsmil_mixer_depths",),
        "kwargs": {"type": int, "nargs": "+", "default": [2, 2, 2, 2], "help": "RSMIL mixer stage depths"},
    },
    {
        "flags": ("--rsmil_mixer_drop_path_rate",),
        "kwargs": {"type": float, "default": 0.1, "help": "RSMIL mixer drop path rate"},
    },
    {
        "flags": ("--rsmil_mixer_act",),
        "kwargs": {"type": str, "default": "relu", "choices": ["relu", "gelu"], "help": "RSMIL mixer activation"},
    },
    {
        "flags": ("--rsmil_mixer_dropout",),
        "kwargs": {"type": float, "default": None, "help": "RSMIL mixer dropout, defaults to --drop_out"},
    },
]


SETTINGS_EXPORT_MAP = (
    ("model_type", "model_type"),
    ("task", "task"),
    ("task_type", "task_type"),
    ("num_splits", "k"),
    ("k_start", "k_start"),
    ("k_end", "k_end"),
    ("max_epochs", "max_epochs"),
    ("lr", "lr"),
    ("reg", "reg"),
    ("lambda_reg", "lambda_reg"),
    ("seed", "seed"),
    ("opt", "opt"),
    ("drop_out", "drop_out"),
    ("feat_root_dir", "feat_root_dir"),
    ("feat_mag", "feat_mag"),
    ("split_dir", "split_dir"),
    ("results_dir", "results_dir"),
    ("weighted_sample", "weighted_sample"),
    ("early_stopping", "early_stopping"),
    ("rsmil_inner_dim", "rsmil_inner_dim"),
    ("rsmil_attn_dim", "rsmil_attn_dim"),
    ("rsmil_n_token_1", "rsmil_n_token_1"),
    ("rsmil_n_token_2", "rsmil_n_token_2"),
    ("rsmil_n_masked_patch_2", "rsmil_n_masked_patch_2"),
    ("rsmil_mask_drop", "rsmil_mask_drop"),
    ("rsmil_dim_reduction", "rsmil_dim_reduction"),
    ("rsmil_mixer_embed_dim", "rsmil_mixer_embed_dim"),
    ("rsmil_mixer_depths", "rsmil_mixer_depths"),
    ("rsmil_mixer_drop_path_rate", "rsmil_mixer_drop_path_rate"),
    ("rsmil_mixer_act", "rsmil_mixer_act"),
    ("rsmil_mixer_dropout", "rsmil_mixer_dropout"),
)


def build_train_argparser():
    parser = argparse.ArgumentParser(description=TRAIN_SCRIPT_DESCRIPTION)
    for spec in CLI_ARGUMENTS:
        parser.add_argument(*spec["flags"], **spec["kwargs"])
    return parser


def normalize_train_args(args):
    args.model_type = MODEL_TYPE
    args.rsmil_mixer_depths = tuple(args.rsmil_mixer_depths)
    if args.rsmil_mixer_dropout is None:
        args.rsmil_mixer_dropout = args.drop_out
    return args


def collect_training_settings(args, dataset_csv_path):
    settings = {setting_key: getattr(args, attr_name, None) for setting_key, attr_name in SETTINGS_EXPORT_MAP}
    settings["dataset_csv_path"] = dataset_csv_path
    return settings
