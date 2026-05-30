import os

from .base import PACKAGE_ROOT


TASK_CONFIGS = {
    "task_tls_vs_notls_coad335": {
        "default_csv": os.path.join(PACKAGE_ROOT, "dataset_csv", "COAD335.csv"),
        "default_feature_subdir": "TLS_NOTLS_UNI",
        "default_split_dir": os.path.join(PACKAGE_ROOT, "splits", "task_coad335_100"),
        "label_dict": {"NOTLS": 0, "TLS": 1},
        "n_classes": 2,
        "task_type": "TLS_vs_NOT",
    },
    "task_1_TLS_vs_NOTLS_535": {
        "default_csv": os.path.join(PACKAGE_ROOT, "dataset_csv", "535filtered_data.csv"),
        "default_feature_subdir": "TLS_NOTLS_UNI",
        "default_split_dir": None,
        "label_dict": {"NOTLS": 0, "TLS": 1},
        "n_classes": 2,
        "task_type": "TLS_vs_NOT",
    },
}


def get_task_config(task_name):
    return TASK_CONFIGS[task_name]


def get_task_choices():
    return sorted(TASK_CONFIGS.keys())
