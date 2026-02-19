# src/train_kfold_with_val.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix
)

from src.Dataset import PatientBagDataset
from src.MIL import AttentionMIL


def cross_validation_mil_with_val(
    adata,
    input_dim,
    num_classes=2,
    hidden_dim=128,
    num_epochs=50,
    learning_rate=5e-4,
    weight_decay=1e-2,
    save_path="results",
    label_col="Response_3m",
    aggregator="attention",
    topk=0,
    tau=0.0,
    k=5,
    seed=42,
    val_frac=0.2,                 # ✅ NEW
    store_attention=True,
    cache_bags=False,
    dropout=0.25,
    batch_size=1,

    # ✅ selection / stopping now uses validation
    select_metric="val_loss",      # "val_loss" or "val_bal_acc"
    eval_every=1,
    patience=10,
    min_delta=1e-4,
):
    """
    Patient-level stratified K-fold CV with an inner validation split (from training set).
    Output schema is aligned to your original cross_validation_mil().

    - Outer: StratifiedKFold(k) on patient-level labels
    - Inner: StratifiedShuffleSplit(val_frac) on outer-train patients
    - Checkpoint/early-stop: based on validation (val_loss or val_bal_acc)

    NOTE: Only "kfold" style (with val) is implemented here; LOOCV not included.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_path, exist_ok=True)

    # create full dataset (for patient list + stable label_map)
    full_dataset = PatientBagDataset(adata, label_col=label_col)

    # ===== label_map: keep EXACTLY your original logic =====
    raw_labels = adata.obs[label_col].dropna().unique()
    label_map = {}
    for lbl in raw_labels:
        s = str(lbl).strip()
        if s in ['Yes', 'CD19pos', '1', '1.0', 'OR']:
            label_map[s] = 1
        elif s in ['No', 'CD19neg', '0', '0.0', 'NR']:
            label_map[s] = 0
    print(f"[INFO] Final Mapping: {label_map}")

    patients = np.array(full_dataset.patient_list, dtype=object)
    y_pat_raw = np.array([full_dataset.patient_labels[p] for p in patients], dtype=str)

    # sanity: all patient labels must be in label_map (check a subset like your old code)
    missing = [v for v in y_pat_raw[:50] if v not in label_map]
    if len(missing) > 0:
        print("[ERROR] Some patient labels not in label_map:", missing[:10])
        print("[ERROR] unique missing:", sorted(set(missing))[:20])

    y_pat = np.array([label_map[v] for v in y_pat_raw], dtype=int)

    # ===== Debug prints matching your style =====
    print("[DEBUG] raw_labels (from adata.obs):", sorted([str(x).strip() for x in raw_labels])[:20], "...")
    print("[DEBUG] label_map keys:", sorted(label_map.keys()))
    print("[DEBUG] full_dataset._label_to_int (dataset auto):", getattr(full_dataset, "_label_to_int", None))
    for p in patients[:10]:
        print("[DEBUG] example patient label:", p, "->", full_dataset.patient_labels[p])

    pat_tbl = (
        adata.obs.groupby("patient_id")[label_col]
        .first()
        .astype(str).str.strip()
    )
    print("\n[CHECK] patient-level raw label counts:")
    print(pat_tbl.value_counts(dropna=False))
    inv_map = {}
    for lbl, v in label_map.items():
        inv_map.setdefault(v, []).append(lbl)
    print("[CHECK] label_map inverse:", inv_map)

    print(f"[INFO] patients={len(patients)} | class_counts={dict(zip(*np.unique(y_pat, return_counts=True)))}")
    print(f"[INFO] CV={k}-fold stratified + inner val_frac={val_frac}")

    # ---- outer splits ----
    skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)
    splits = list(skf.split(patients, y_pat))

    cv_results = {
        "fold_metrics": [],
        "patient_predictions": {},
        "attention_weights": {} if store_attention else None,
        "overall_metrics": None,
        "cv": "kfold_with_val",
        "k": int(k),
        "val_frac": float(val_frac),
        "label_map": label_map,
        "select_metric": select_metric,
    }

    # accumulators for OOF predictions (test fold only, like your original)
    all_true_labels = []
    all_predicted_labels = []
    all_prediction_probs = []
    all_patient_ids = []

    # parse optional topk/tau once
    _topk = int(topk) if (topk is not None and int(topk) > 0) else None
    _tau  = float(tau) if (tau is not None and float(tau) > 0) else None

    pos_idx = 1
    neg_idx = 0
    print(f"[DEBUG] Using pos_idx={pos_idx} (positive class), neg_idx={neg_idx}")

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        outer_train_pids = patients[tr_idx].tolist()
        test_pids = patients[te_idx].tolist()

        y_outer_tr = y_pat[tr_idx]

        # ✅ inner val split from outer train
        sss = StratifiedShuffleSplit(
            n_splits=1,
            test_size=float(val_frac),
            random_state=int(seed + fold),
        )
        inner_tr_rel, inner_val_rel = next(sss.split(np.array(outer_train_pids, dtype=object), y_outer_tr))
        train_pids = np.array(outer_train_pids, dtype=object)[inner_tr_rel].tolist()
        val_pids   = np.array(outer_train_pids, dtype=object)[inner_val_rel].tolist()

        print(f"\n[Fold {fold}/{len(splits)}] train={len(train_pids)} val={len(val_pids)} test={len(test_pids)} "
              f"test_pos={(y_pat[te_idx]==1).sum() if num_classes==2 else 'NA'}")

        # ---- build fold datasets ----
        train_mask = adata.obs["patient_id"].isin(train_pids).to_numpy()
        val_mask   = adata.obs["patient_id"].isin(val_pids).to_numpy()
        test_mask  = adata.obs["patient_id"].isin(test_pids).to_numpy()

        train_adata = adata[train_mask]
        val_adata   = adata[val_mask]
        test_adata  = adata[test_mask]

        train_dataset = PatientBagDataset(
            train_adata,
            task_type="classification",
            label_col=label_col,
            label_map=label_map,
            drop_missing=False,
            use_sample_source=False,
            cache_bags=cache_bags,
        )
        val_dataset = PatientBagDataset(
            val_adata,
            task_type="classification",
            label_col=label_col,
            label_map=label_map,
            drop_missing=False,
            use_sample_source=False,
            cache_bags=cache_bags,
        )
        test_dataset = PatientBagDataset(
            test_adata,
            task_type="classification",
            label_col=label_col,
            label_map=label_map,
            drop_missing=False,
            use_sample_source=False,
            cache_bags=cache_bags,
        )

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # save dir
        fold_save_path = os.path.join(save_path, f"fold_{fold:02d}")
        os.makedirs(fold_save_path, exist_ok=True)

        # model
        model = AttentionMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            sample_source_dim=None,
            aggregator=aggregator,
            topk=_topk,
            tau=_tau,
        ).to(device)

        # ---- class weights from TRAIN fold (patient-level) ----
        y_tr_raw = np.array([train_dataset.patient_labels[p] for p in train_dataset.patient_list], dtype=str)
        y_tr = np.array([label_map[v] for v in y_tr_raw], dtype=int)

        classes_present = np.unique(y_tr)
        if len(classes_present) < num_classes:
            class_weights = torch.ones(num_classes, device=device)
        else:
            cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y_tr)
            class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # NOTE: no verbose= for older torch compatibility
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5,
            min_lr=1e-6,
        )

        print("[DEBUG] class_weights:", class_weights.detach().cpu().numpy())

        # ---- training with val-based checkpoint selection / early stopping ----
        best_score = None
        best_epoch = -1
        epochs_without_improvement = 0
        best_path = os.path.join(fold_save_path, "best_model.pth")

        for epoch in range(1, num_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_correct, train_total = 0, 0

            for batch in train_loader:
                if len(batch) == 4:
                    bags, batch_labels, _pids, _ = batch
                else:
                    bags, batch_labels, _pids = batch

                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device).long().view(-1)

                out = model(bags)
                logits = out["logits"]
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss_sum += float(loss.item())
                preds = torch.argmax(logits, dim=1)
                train_total += int(batch_labels.size(0))
                train_correct += int((preds == batch_labels).sum().item())

            train_loss = train_loss_sum / max(len(train_loader), 1)
            train_acc = train_correct / train_total if train_total > 0 else 0.0

            # ---- val eval each epoch (cheap: patient-level, batch_size=1) ----
            model.eval()
            val_loss_sum = 0.0
            val_trues, val_preds = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 4:
                        bags, batch_labels, _pids, _ = batch
                    else:
                        bags, batch_labels, _pids = batch

                    bags = [bag.to(device) for bag in bags]
                    batch_labels = batch_labels.to(device).long().view(-1)

                    out = model(bags)
                    logits = out["logits"]
                    loss = criterion(logits, batch_labels)

                    val_loss_sum += float(loss.item())
                    preds = torch.argmax(logits, dim=1)
                    val_trues.extend(batch_labels.cpu().numpy().tolist())
                    val_preds.extend(preds.cpu().numpy().tolist())

            val_loss = val_loss_sum / max(len(val_loader), 1)
            val_bal_acc = balanced_accuracy_score(val_trues, val_preds) if len(val_trues) > 0 else 0.0

            scheduler.step(val_loss)

            if (epoch % 10 == 0) or (epoch == 1):
                print(f"[Fold {fold}] ep={epoch}/{num_epochs} "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                      f"val_loss={val_loss:.4f} val_bal_acc={val_bal_acc:.4f}")

            # ---- checkpoint selection metric ----
            if (epoch % eval_every) == 0:
                if select_metric == "val_loss":
                    score = -val_loss
                elif select_metric == "val_bal_acc":
                    score = float(val_bal_acc)
                else:
                    raise ValueError("select_metric must be 'val_loss' or 'val_bal_acc'")

                improved = (best_score is None) or (score > best_score + float(min_delta))
                if improved:
                    best_score = score
                    best_epoch = epoch
                    epochs_without_improvement = 0
                    torch.save(model.state_dict(), best_path)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(patience):
                        print(f"[Fold {fold}] Early stopping at ep={epoch}, best_ep={best_epoch}")
                        break

        # ---- load best ----
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.eval()

        # ---- evaluate on test patients (OOF) ----
        fold_true, fold_pred, fold_prob = [], [], []

        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 4:
                    bags, batch_labels, patient_ids, _ = batch
                else:
                    bags, batch_labels, patient_ids = batch

                patient_id = patient_ids[0]
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device).long().view(-1)

                want_attn = (aggregator == "attention")
                out = model(bags, return_attention=want_attn)

                logits = out["logits"]
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                # ---- attention store (test only) ----
                if want_attn and ("attn" in out):
                    attn_weights = out["attn"]
                    if store_attention:
                        cv_results["attention_weights"][patient_id] = [
                            (w.detach().cpu().numpy() if w is not None else None) for w in attn_weights
                        ]

                true_label = int(batch_labels.cpu().numpy()[0])
                pred_label = int(preds.cpu().numpy()[0])
                probs_np = probs.detach().cpu().numpy()[0]

                if num_classes == 2:
                    pos_prob = float(probs_np[pos_idx])
                    all_prediction_probs.append(pos_prob)
                    fold_prob.append(pos_prob)

                all_true_labels.append(true_label)
                all_predicted_labels.append(pred_label)
                all_patient_ids.append(patient_id)

                fold_true.append(true_label)
                fold_pred.append(pred_label)

                cv_results["patient_predictions"][patient_id] = {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "probabilities": probs_np.tolist(),
                    "correct": (pred_label == true_label),
                    "fold": fold,
                }

                cv_results["fold_metrics"].append({
                    "patient_id": patient_id,
                    "fold": fold,
                    "accuracy": 1.0 if pred_label == true_label else 0.0,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "prob_positive": float(probs_np[1]) if num_classes == 2 else None,
                })

        # ---- per-fold summary print (like your old code) ----
        fold_true_np = np.array(fold_true, dtype=int)
        fold_pred_np = np.array(fold_pred, dtype=int)

        fold_acc = accuracy_score(fold_true_np, fold_pred_np)
        fold_prec = precision_score(fold_true_np, fold_pred_np, zero_division=0)
        fold_rec  = recall_score(fold_true_np, fold_pred_np, zero_division=0)
        fold_f1   = f1_score(fold_true_np, fold_pred_np, zero_division=0)

        if num_classes == 2 and len(np.unique(fold_true_np)) > 1:
            fold_prob_np = np.array(fold_prob, dtype=float)
            fold_auc = roc_auc_score(fold_true_np, fold_prob_np)
            fold_pr_auc = average_precision_score(fold_true_np, fold_prob_np)
        else:
            fold_auc = float("nan")
            fold_pr_auc = float("nan")

        print(f"\nFold {fold} Results:")
        print(f"Accuracy: {fold_acc:.4f}, Precision: {fold_prec:.4f}, Recall: {fold_rec:.4f}, F1: {fold_f1:.4f}")
        print(f"Best epoch (val): {best_epoch} | select_metric={select_metric}")
        if num_classes == 2:
            print(f"AUC: {fold_auc:.4f}" if np.isfinite(fold_auc) else "AUC: NA")
            print(f"PR-AUC: {fold_pr_auc:.4f}" if np.isfinite(fold_pr_auc) else "PR-AUC: NA")

    # ---- overall metrics (OOF) ----
    all_true_labels = np.array(all_true_labels, dtype=int)
    all_predicted_labels = np.array(all_predicted_labels, dtype=int)
    all_patient_ids = np.array(all_patient_ids, dtype=object)

    overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)

    if num_classes == 2:
        all_prediction_probs = np.array(all_prediction_probs, dtype=float)

        # keep your AUC direction debug
        auc_prob = roc_auc_score(all_true_labels, all_prediction_probs) if len(np.unique(all_true_labels)) > 1 else np.nan
        auc_flip = roc_auc_score(all_true_labels, 1 - all_prediction_probs) if len(np.unique(all_true_labels)) > 1 else np.nan
        if np.isfinite(auc_prob):
            print("\n[DEBUG] AUC(prob) =", auc_prob)
            print("[DEBUG] AUC(1-prob) =", auc_flip)
            print("[DEBUG] sum =", auc_prob + auc_flip)

        m1 = all_prediction_probs[all_true_labels == 1].mean() if (all_true_labels == 1).any() else np.nan
        m0 = all_prediction_probs[all_true_labels == 0].mean() if (all_true_labels == 0).any() else np.nan
        print(f"[CHECK] mean(pos_prob | y=1)={m1:.4f}  mean(pos_prob | y=0)={m0:.4f}")

        overall_precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_auc = roc_auc_score(all_true_labels, all_prediction_probs) if len(np.unique(all_true_labels)) > 1 else float("nan")
        overall_auprc = average_precision_score(all_true_labels, all_prediction_probs) if len(np.unique(all_true_labels)) > 1 else float("nan")
    else:
        overall_precision = precision_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_auc = float("nan")
        overall_auprc = float("nan")

    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    class_metrics = {}
    for c in range(num_classes):
        tp = np.sum((all_true_labels == c) & (all_predicted_labels == c))
        actual = np.sum(all_true_labels == c)
        predp = np.sum(all_predicted_labels == c)
        class_metrics[f"class_{c}"] = {
            "precision": float(tp / predp) if predp > 0 else 0.0,
            "recall": float(tp / actual) if actual > 0 else 0.0,
            "count": int(actual),
        }

    cv_results["overall_metrics"] = {
        "accuracy": float(overall_accuracy),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "f1": float(overall_f1),
        "auc": float(overall_auc) if np.isfinite(overall_auc) else None,
        "auprc": float(overall_auprc) if np.isfinite(overall_auprc) else None,
        "confusion_matrix": conf_matrix.tolist(),
        "class_metrics": class_metrics,
    }

    print("\n===== CV Final Results =====")
    print(f"CV=kfold_with_val k={k} val_frac={val_frac}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({int((all_true_labels==all_predicted_labels).sum())}/{len(all_true_labels)})")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    if num_classes == 2:
        print(f"Overall AUC: {overall_auc:.4f}" if np.isfinite(overall_auc) else "Overall AUC: NA")
        print(f"Overall AUPRC: {overall_auprc:.4f}" if np.isfinite(overall_auprc) else "Overall AUPRC: NA")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # save
    if save_path is not None:
        tag = f"kfold{k}_val{str(val_frac).replace('.','p')}"
        results_path = os.path.join(save_path, f"{tag}_results.pkl")
        with open(results_path, "wb") as f:
            import pickle
            pickle.dump(cv_results, f)
        print(f"[INFO] Saved: {results_path}")

    return cv_results