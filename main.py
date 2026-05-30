from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    from RSMIL.config import (
        MODEL_TYPE,
        REQUIRED_FEAT_MAG,
        build_train_argparser,
        collect_training_settings,
        get_task_config,
        normalize_train_args,
    )
    from RSMIL.dataset_modules.dataset_generic import Generic_MIL_Dataset
    from RSMIL.utils.core_utils import train
    from RSMIL.utils.file_utils import save_pkl
else:
    from .config import (
        MODEL_TYPE,
        REQUIRED_FEAT_MAG,
        build_train_argparser,
        collect_training_settings,
        get_task_config,
        normalize_train_args,
    )
    from .dataset_modules.dataset_generic import Generic_MIL_Dataset
    from .utils.core_utils import train
    from .utils.file_utils import save_pkl


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=7):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def is_feature_leaf_dir(path):
    if path is None:
        return False
    return os.path.isdir(os.path.join(path, "pt_files")) or os.path.isdir(os.path.join(path, "h5_files"))


def resolve_feature_data_dir(feat_root_dir, feat_mag, default_subdir):
    if feat_root_dir is None:
        return None

    feat_root_dir = os.path.abspath(feat_root_dir)
    if is_feature_leaf_dir(feat_root_dir):
        return feat_root_dir

    mag_dir = os.path.join(feat_root_dir, feat_mag)
    if is_feature_leaf_dir(mag_dir):
        return mag_dir

    subdir_dir = os.path.join(feat_root_dir, default_subdir)
    if is_feature_leaf_dir(subdir_dir) or os.path.isdir(subdir_dir):
        return subdir_dir

    subdir_mag_dir = os.path.join(subdir_dir, feat_mag)
    if is_feature_leaf_dir(subdir_mag_dir):
        return subdir_mag_dir

    return mag_dir


def resolve_optional_path(path_value, default_value):
    if path_value is None:
        return default_value
    return os.path.abspath(path_value)


def build_dataset(args):
    task_config = get_task_config(args.task)
    args.task_type = task_config["task_type"]
    args.n_classes = task_config["n_classes"]

    dataset_csv_path = resolve_optional_path(args.dataset_csv_path, task_config["default_csv"])
    feature_dir = resolve_feature_data_dir(args.feat_root_dir, args.feat_mag, task_config["default_feature_subdir"])

    dataset = Generic_MIL_Dataset(
        csv_path=dataset_csv_path,
        data_dir=feature_dir,
        shuffle=False,
        seed=args.seed,
        print_info=True,
        label_dict=task_config["label_dict"],
        patient_strat=False,
        ignore=[],
    )
    return dataset, dataset_csv_path


def run_training(args, dataset):
    start = 0 if args.k_start == -1 else args.k_start
    end = args.k if args.k_end == -1 else args.k_end
    folds = np.arange(start, end)
    summary_rows = []

    for fold in folds:
        seed_torch(args.seed)
        split_csv = os.path.join(args.split_dir, f"splits_{fold}.csv")
        train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, csv_path=split_csv)
        fold_results = train((train_dataset, val_dataset, test_dataset), fold, args)

        summary_rows.append(
            {
                "folds": fold,
                "test_orin_auc": fold_results["test_orin_auc"],
                "test_acc": fold_results["test_acc"],
                "test_f1": fold_results["test_f1"],
                "test_recall": fold_results["test_recall"],
                "test_precision_score": fold_results["test_precision"],
                "test_accuracy": fold_results["test_accuracy"],
                "test_specificity": fold_results["test_specificity"],
                "test_eval_auc": fold_results["test_eval_auc"],
                "val_orin_auc": fold_results["val_orin_auc"],
                "val_acc": fold_results["val_acc"],
                "val_f1": fold_results["val_f1"],
                "val_recall": fold_results["val_recall"],
                "val_precision_score": fold_results["val_precision"],
                "val_accuracy": fold_results["val_accuracy"],
                "val_specificity": fold_results["val_specificity"],
                "val_eval_auc": fold_results["val_eval_auc"],
            }
        )

        save_pkl(os.path.join(args.results_dir, f"split_{fold}_results.pkl"), fold_results["patient_results"])

    final_df = pd.DataFrame(summary_rows)
    save_name = "summary.csv" if len(folds) == args.k else f"summary_partial_{start}_{end}.csv"
    final_df.to_csv(os.path.join(args.results_dir, save_name), index=False)


def main():
    parser = build_train_argparser()
    args = normalize_train_args(parser.parse_args())

    if args.feat_mag != REQUIRED_FEAT_MAG:
        raise ValueError(f"{MODEL_TYPE} requires --feat_mag {REQUIRED_FEAT_MAG} because it consumes hierarchical hr tensors.")

    task_config = get_task_config(args.task)
    args.split_dir = resolve_optional_path(args.split_dir, task_config["default_split_dir"])
    if args.split_dir is None:
        raise ValueError(f"No default split_dir is packaged for task {args.task}. Please pass --split_dir explicitly.")
    if not os.path.isdir(args.split_dir):
        raise FileNotFoundError(f"Split directory not found: {args.split_dir}")

    print("\nLoad Dataset")
    dataset, dataset_csv_path = build_dataset(args)
    if dataset.data_dir is None:
        raise ValueError("No feature directory was resolved. Please pass --feat_root_dir.")
    if not os.path.isdir(dataset.data_dir):
        raise FileNotFoundError(f"Feature directory not found: {dataset.data_dir}")

    print(f"feature_dir: {dataset.data_dir}")
    print(f"split_dir: {args.split_dir}")
    print(f"dataset_csv_path: {dataset_csv_path}")

    args.results_dir = os.path.abspath(os.path.join(args.results_dir, f"{args.exp_code}_s{args.seed}"))
    os.makedirs(args.results_dir, exist_ok=True)

    settings = collect_training_settings(args, dataset_csv_path)
    with open(os.path.join(args.results_dir, f"experiment_{args.exp_code}.txt"), "w", encoding="utf-8") as handle:
        print(settings, file=handle)

    print("################# Settings ###################")
    for key, value in settings.items():
        print(f"{key}: {value}")

    run_training(args, dataset)
    print("finished!")
    print("end script")


if __name__ == "__main__":
    main()
