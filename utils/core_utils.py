import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize

try:
    from ..dataset_modules.dataset_generic import save_splits
    from ..Model.RSMIL import build_rsmil_model
    from .utils import calculate_error, get_optim, get_split_loader, print_network
except ImportError:
    from dataset_modules.dataset_generic import save_splits
    from Model.RSMIL import build_rsmil_model
    from utils.utils import calculate_error, get_optim, get_split_loader, print_network


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l1_reg_all(model):
    l1_reg = None
    for weight in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(weight).sum()
        else:
            l1_reg = l1_reg + torch.abs(weight).sum()
    return l1_reg


def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1.0)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


def roc_threshold(labels, predictions):
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if np.unique(labels).size < 2:
        return float("nan"), 0.5

    fpr, tpr, thresholds = roc_curve(labels, predictions, pos_label=1)
    _, _, threshold_optimal = optimal_thresh(fpr, tpr, thresholds)
    auc_value = roc_auc_score(labels, predictions)
    return auc_value, threshold_optimal


def eval_metric(probabilities, labels):
    labels_np = labels.detach().cpu().numpy()
    probs_np = probabilities.detach().cpu().numpy()
    auc_value, threshold = roc_threshold(labels_np, probs_np)

    pred_binary = probs_np > threshold
    label_binary = labels_np > threshold

    tp = np.logical_and(pred_binary, label_binary).sum().astype(np.float32)
    tn = np.logical_and(~pred_binary, ~label_binary).sum().astype(np.float32)
    fp = np.logical_and(pred_binary, ~label_binary).sum().astype(np.float32)
    fn = np.logical_and(~pred_binary, label_binary).sum().astype(np.float32)

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-12)
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    specificity = tn / (tn + fp + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
    return f1, recall, precision, accuracy, specificity, auc_value


def pairwise_attention_diversity(attn_tensor, token_axis):
    if attn_tensor is None:
        return torch.tensor(0.0, device=device)

    num_tokens = attn_tensor.shape[token_axis]
    if num_tokens <= 1:
        return torch.tensor(0.0, device=attn_tensor.device)

    attn_tensor = attn_tensor.movedim(token_axis, 0)
    denom = num_tokens * (num_tokens - 1) / 2.0
    diff_loss = torch.tensor(0.0, device=attn_tensor.device)

    for i in range(num_tokens):
        for j in range(i + 1, num_tokens):
            attn_i = attn_tensor[i].reshape(-1, attn_tensor[i].shape[-1])
            attn_j = attn_tensor[j].reshape(-1, attn_tensor[j].shape[-1])
            diff_loss = diff_loss + torch.cosine_similarity(attn_i, attn_j, dim=-1).mean() / denom

    return diff_loss


def compute_model_loss(logits, label, loss_fn, result_dict=None):
    loss = loss_fn(logits, label)

    if result_dict is None:
        return loss

    sub_preds = result_dict.get("sub_preds")
    if sub_preds is not None and sub_preds.ndim == 2 and sub_preds.size(0) > 1:
        sub_loss = loss_fn(sub_preds, label.repeat_interleave(sub_preds.size(0)))
    else:
        sub_loss = torch.tensor(0.0, device=logits.device)

    diff_loss_1 = pairwise_attention_diversity(result_dict.get("attn_1"), token_axis=1)
    diff_loss_2 = pairwise_attention_diversity(result_dict.get("attn_2"), token_axis=0)
    result_dict["rsmil_sub_loss"] = sub_loss.detach()
    result_dict["rsmil_diff_loss"] = (diff_loss_1 + diff_loss_2).detach()
    return loss + sub_loss + diff_loss_1 + diff_loss_2


class EarlyStopping:
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, epoch, val_loss, model, ckpt_name="checkpoint.pt"):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            return

        if score < self.best_score:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
            return

        self.best_score = score
        self.save_checkpoint(val_loss, model, ckpt_name)
        self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...")
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def build_model(args):
    return build_rsmil_model(
        "rsmil_stage1",
        in_dim=args.in_dim,
        n_classes=args.n_classes,
        inner_dim=args.rsmil_inner_dim,
        attn_dim=args.rsmil_attn_dim,
        dropout=args.drop_out,
        n_token_1=args.rsmil_n_token_1,
        n_token_2=args.rsmil_n_token_2,
        n_masked_patch_2=args.rsmil_n_masked_patch_2,
        mask_drop=args.rsmil_mask_drop,
        dim_reduction=args.rsmil_dim_reduction,
        care_embed_dim=args.rsmil_mixer_embed_dim,
        care_depths=args.rsmil_mixer_depths,
        care_drop_path_rate=args.rsmil_mixer_drop_path_rate,
        care_act=args.rsmil_mixer_act,
        care_dropout=args.rsmil_mixer_dropout,
    )


def train(datasets, cur, args):
    print(f"\nTraining Fold {cur}!")
    writer = None
    writer_dir = os.path.join(args.results_dir, str(cur))
    os.makedirs(writer_dir, exist_ok=True)

    if args.log_data:
        from tensorboardX import SummaryWriter

        writer = SummaryWriter(writer_dir, flush_secs=15)

    train_split, val_split, test_split = datasets
    print("\nInit train/val/test splits...", end=" ")
    save_splits(datasets, ["train", "val", "test"], os.path.join(args.results_dir, f"splits_{cur}.csv"))
    print("Done!")
    print(f"Training on {len(train_split)} samples")
    print(f"Validating on {len(val_split)} samples")
    print(f"Testing on {len(test_split)} samples")

    print("\nInit loss function...", end=" ")
    loss_fn = nn.CrossEntropyLoss()
    print("Done!")

    print("\nInit model...", end=" ")
    model = build_model(args).to(device)
    print("Done!")
    print_network(model)

    print("\nInit optimizer...", end=" ")
    optimizer = get_optim(model, args)
    print("Done!")

    print("\nInit loaders...", end=" ")
    train_loader = get_split_loader(train_split, training=True, testing=args.testing, weighted=args.weighted_sample)
    val_loader = get_split_loader(val_split, testing=args.testing)
    test_loader = get_split_loader(test_split, testing=args.testing)
    print("Done!")

    print("\nSetup early stopping...", end=" ")
    early_stopping = EarlyStopping(patience=20, stop_epoch=50, verbose=True) if args.early_stopping else None
    reg_fn = l1_reg_all if args.reg_type == "TLS" else None
    print("Done!")

    for epoch in range(args.max_epochs):
        train_loop(epoch, model, train_loader, optimizer, args, writer, reg_fn, loss_fn, args.lambda_reg)
        stop = validate(cur, epoch, model, val_loader, args, early_stopping, writer, reg_fn, loss_fn, args.results_dir, args.lambda_reg)
        if stop:
            break

    ckpt_path = os.path.join(args.results_dir, f"s_{cur}_checkpoint.pt")
    if args.early_stopping:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        torch.save(model.state_dict(), ckpt_path)

    val_metrics = summary(model, val_loader, args.n_classes)
    test_metrics = summary(model, test_loader, args.n_classes)

    print(
        "Val_error: {:.4f}, Val_Orin_ROC_AUC: {:.4f}, Val_F1: {:.4f}, Val_Recall: {:.4f}, "
        "Val_Precision: {:.4f}, Val_Accuracy: {:.4f}, Val_Specificity: {:.4f}, Val_Eval_ROC_AUC: {:.4f}".format(
            val_metrics["error"],
            val_metrics["orin_auc"],
            val_metrics["f1"],
            val_metrics["recall"],
            val_metrics["precision"],
            val_metrics["accuracy"],
            val_metrics["specificity"],
            val_metrics["eval_auc"],
        )
    )
    print(
        "Test_error: {:.4f}, Test_Orin_ROC_AUC: {:.4f}, Test_F1: {:.4f}, Test_Recall: {:.4f}, "
        "Test_Precision: {:.4f}, Test_Accuracy: {:.4f}, Test_Specificity: {:.4f}, Test_Eval_ROC_AUC: {:.4f}".format(
            test_metrics["error"],
            test_metrics["orin_auc"],
            test_metrics["f1"],
            test_metrics["recall"],
            test_metrics["precision"],
            test_metrics["accuracy"],
            test_metrics["specificity"],
            test_metrics["eval_auc"],
        )
    )

    if writer:
        writer.add_scalar("final/val_error", val_metrics["error"], 0)
        writer.add_scalar("final/val_auc", val_metrics["orin_auc"], 0)
        writer.add_scalar("final/test_error", test_metrics["error"], 0)
        writer.add_scalar("final/test_auc", test_metrics["orin_auc"], 0)
        writer.close()

    return {
        "patient_results": test_metrics["patient_results"],
        "val_probs": val_metrics["all_probs"],
        "val_labels": val_metrics["all_labels"],
        "test_probs": test_metrics["all_probs"],
        "test_labels": test_metrics["all_labels"],
        "test_orin_auc": test_metrics["orin_auc"],
        "test_f1": test_metrics["f1"],
        "test_recall": test_metrics["recall"],
        "test_precision": test_metrics["precision"],
        "test_accuracy": test_metrics["accuracy"],
        "test_specificity": test_metrics["specificity"],
        "test_eval_auc": test_metrics["eval_auc"],
        "test_acc": 1.0 - test_metrics["error"],
        "val_orin_auc": val_metrics["orin_auc"],
        "val_f1": val_metrics["f1"],
        "val_recall": val_metrics["recall"],
        "val_precision": val_metrics["precision"],
        "val_accuracy": val_metrics["accuracy"],
        "val_specificity": val_metrics["specificity"],
        "val_eval_auc": val_metrics["eval_auc"],
        "val_acc": 1.0 - val_metrics["error"],
    }


def train_loop(epoch, model, loader, optimizer, args, writer=None, reg_fn=None, loss_fn=None, lambda_reg=0.0):
    model.train()
    train_loss_ori = 0.0
    train_loss_reg = 0.0
    train_error = 0.0

    optimizer.zero_grad()
    print("")
    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)

        logits, _, y_hat, _, result_dict = model(data)
        loss = compute_model_loss(logits, label, loss_fn, result_dict)
        loss_value = loss.item()

        loss_reg = reg_fn(model) * lambda_reg if reg_fn is not None else 0.0
        total_loss = loss + loss_reg
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss_ori += loss_value
        train_loss_reg += loss_value + (loss_reg.item() if torch.is_tensor(loss_reg) else loss_reg)
        train_error += calculate_error(y_hat, label)

        if (batch_idx + 1) % 20 == 0:
            print(
                "batch {}, ori_loss: {:.4f}, reg_loss: {:.4f}, label: {}, bag_size: {}".format(
                    batch_idx,
                    loss_value,
                    loss_value + (loss_reg.item() if torch.is_tensor(loss_reg) else loss_reg),
                    label.item(),
                    data.size(0),
                )
            )

    train_loss_ori /= len(loader)
    train_loss_reg /= len(loader)
    train_error /= len(loader)

    print(
        "Epoch: {}, train_loss_ori: {:.4f}, train_loss_reg: {:.4f}, train_error: {:.4f}".format(
            epoch, train_loss_ori, train_loss_reg, train_error
        )
    )

    if writer:
        writer.add_scalar("train/ori_loss", train_loss_ori, epoch)
        writer.add_scalar("train/reg_loss", train_loss_reg, epoch)
        writer.add_scalar("train/error", train_error, epoch)


def validate(cur, epoch, model, loader, args, early_stopping=None, writer=None, reg_fn=None, loss_fn=None, results_dir=None, lambda_reg=0.0):
    model.eval()

    val_loss = 0.0
    val_loss_reg = 0.0
    val_error = 0.0
    probs = np.zeros((len(loader), args.n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(loader):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            logits, y_prob, y_hat, _, result_dict = model(data)
            loss = compute_model_loss(logits, label, loss_fn, result_dict)
            loss_value = loss.item()
            loss_reg = reg_fn(model) * lambda_reg if reg_fn is not None else 0.0

            probs[batch_idx] = y_prob.squeeze(0).cpu().numpy()
            labels[batch_idx] = label.item()
            val_loss += loss_value
            val_loss_reg += loss_value + (loss_reg.item() if torch.is_tensor(loss_reg) else loss_reg)
            val_error += calculate_error(y_hat, label)

    val_loss /= len(loader)
    val_loss_reg /= len(loader)
    val_error /= len(loader)
    auc_value = safe_auc(labels, probs[:, 1] if args.n_classes == 2 else probs, args.n_classes)

    if writer:
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/loss_reg", val_loss_reg, epoch)
        writer.add_scalar("val/auc", auc_value, epoch)
        writer.add_scalar("val/error", val_error, epoch)

    print(
        "\nVal Set, val_loss: {:.4f}, val_loss_reg: {:.4f}, val_error: {:.4f}, auc: {:.4f}".format(
            val_loss, val_loss_reg, val_error, auc_value
        )
    )

    if early_stopping:
        assert results_dir is not None
        early_stopping(epoch, val_loss, model, ckpt_name=os.path.join(results_dir, f"s_{cur}_checkpoint.pt"))
        if early_stopping.early_stop:
            print("Early stopping")
            return True

    return False


def safe_auc(labels, probs, n_classes):
    labels = np.asarray(labels)
    if np.unique(labels).size < 2:
        return float("nan")

    if n_classes == 2:
        return roc_auc_score(labels, probs)
    return roc_auc_score(labels, probs, multi_class="ovr")


def summary(model, loader, n_classes):
    model.eval()
    test_error = 0.0
    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))
    slide_ids = loader.dataset.slide_data["slide_id"]
    patient_results = {}

    for batch_idx, (data, label) in enumerate(loader):
        data = data.to(device)
        label = label.to(device)
        slide_id = slide_ids.iloc[batch_idx]

        with torch.inference_mode():
            logits, y_prob, y_hat, _, _ = model(data)

        probs = y_prob.squeeze(0).cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        patient_results[slide_id] = {"slide_id": np.array(slide_id), "prob": probs, "label": label.item()}
        test_error += calculate_error(y_hat, label)

    test_error /= len(loader)

    if n_classes == 2:
        orin_auc = safe_auc(all_labels, all_probs[:, 1], n_classes)
        f1, recall, precision, accuracy, specificity, eval_auc = eval_metric(
            torch.tensor(all_probs[:, 1]), torch.tensor(all_labels)
        )
        class_aucs = []
    else:
        binary_labels = label_binarize(all_labels, classes=[idx for idx in range(n_classes)])
        class_aucs = []
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                class_aucs.append(calc_auc(fpr, tpr))
            else:
                class_aucs.append(float("nan"))
        orin_auc = float(np.nanmean(np.array(class_aucs)))
        predictions = np.argmax(all_probs, axis=1)
        f1 = f1_score(all_labels, predictions, average="macro")
        recall = recall_score(all_labels, predictions, average="macro", zero_division=0)
        precision = precision_score(all_labels, predictions, average="macro", zero_division=0)
        accuracy = accuracy_score(all_labels, predictions)
        specificity = float("nan")
        eval_auc = orin_auc

    return {
        "patient_results": patient_results,
        "error": float(test_error),
        "orin_auc": float(orin_auc),
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "accuracy": float(accuracy),
        "specificity": float(specificity),
        "eval_auc": float(eval_auc),
        "all_probs": all_probs,
        "all_labels": all_labels,
        "class_aucs": class_aucs,
    }
