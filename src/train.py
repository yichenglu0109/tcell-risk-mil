import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os    
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from datetime import datetime
from src.Dataset import PatientBagDataset, preprocess_data, load_and_explore_data
from src.Autoencoder import train_autoencoder, evaluate_autoencoder
from src.MIL import AttentionMIL
from sklearn.model_selection import StratifiedKFold

def cross_validation_mil(
    adata,
    input_dim,
    num_classes=2,
    hidden_dim=128,
    sample_source_dim=4,
    num_epochs=50,
    learning_rate=5e-4,
    weight_decay=1e-2,
    save_path="results",
    label_col="Response_3m",
    aggregator="attention",
    topk=0,
    tau=0.0,
    cv="loocv",     # "loocv" or "kfold"
    k=5,
    seed=42,
):
    """
    Patient-level CV for MIL model.
    - LOOCV: each fold holds out 1 patient.
    - kfold: StratifiedKFold on patient-level labels, out-of-fold predictions.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(save_path, exist_ok=True)

    # create full dataset (for patient list + stable label_map)
    full_dataset = PatientBagDataset(adata, label_col=label_col)

    # ---- 替換原本的 label_map 獲取方式 ----
    # 強制指定映射，不受數據讀取順序影響
    # 這裡的 key 必須對應你 adata.obs[label_col] 裡的原始字串 (如 'No', 'Yes')
    # label_map = {"No": 0, "Yes": 1} 
    # print("[INFO] Fixed Global label_map:", label_map)

    # 修改後的簡潔邏輯
    raw_labels = adata.obs[label_col].dropna().unique()
    label_map = {}

    for lbl in raw_labels:
        s = str(lbl).strip()
        # 支援原有的 Yes/No 以及新的 CD19pos/neg
        if s in ['Yes', 'CD19pos', '1', '1.0', 'OR']:
            label_map[s] = 1
        elif s in ['No', 'CD19neg', '0', '0.0', 'NR']:
            label_map[s] = 0
    
    print(f"[INFO] Final Mapping: {label_map}")

    # ===== DEBUG: check mapping consistency =====
    print("[DEBUG] raw_labels (from adata.obs):", sorted([str(x).strip() for x in raw_labels])[:20], "...")
    print("[DEBUG] label_map keys:", sorted(label_map.keys()))
    print("[DEBUG] full_dataset._label_to_int (dataset auto):", getattr(full_dataset, "_label_to_int", None))

    # show a few patient-level raw labels
    for p in patients[:10]:
        print("[DEBUG] example patient label:", p, "->", full_dataset.patient_labels[p])

    # sanity: all patient labels must be in label_map
    missing = [v for v in y_pat_raw[:50] if v not in label_map]  # check first 50
    if len(missing) > 0:
        print("[ERROR] Some patient labels not in label_map:", missing[:10])
        print("[ERROR] unique missing:", sorted(set(missing))[:20])
    # ===========================================

    # # ---- FIX: freeze label_map globally so folds are consistent ----
    # label_map = full_dataset._label_to_int
    # print("[INFO] Global label_map:", label_map)

    sample_source_dim = (
        full_dataset.sample_source_dim
        if hasattr(full_dataset, "sample_source_dim")
        else None
    )

    patients = np.array(full_dataset.patient_list, dtype=object)
    # patient-level labels as *string* (as stored by Dataset)
    y_pat_raw = np.array([full_dataset.patient_labels[p] for p in patients], dtype=str)
    # map to int labels
    y_pat = np.array([label_map[v] for v in y_pat_raw], dtype=int)

    print(f"[INFO] patients={len(patients)} | class_counts={dict(zip(*np.unique(y_pat, return_counts=True)))}")

    # ---- prepare splits ----
    if cv == "loocv":
        splits = [(
            np.array([j for j in range(len(patients)) if j != i], dtype=int),
            np.array([i], dtype=int)
        ) for i in range(len(patients))]
        print(f"[INFO] CV=LOOCV folds={len(splits)}")
    elif cv == "kfold":
        skf = StratifiedKFold(n_splits=int(k), shuffle=True, random_state=seed)
        splits = list(skf.split(patients, y_pat))
        print(f"[INFO] CV={k}-fold stratified folds={len(splits)}")
    else:
        raise ValueError(f"Unknown cv={cv}, use 'loocv' or 'kfold'")

    cv_results = {
        "fold_metrics": [],
        "patient_predictions": {},
        "attention_weights": {},
        "overall_metrics": None,
        "cv": cv,
        "k": int(k) if cv == "kfold" else None,
        "label_map": label_map,
    }

    # accumulators for OOF predictions
    all_true_labels = []
    all_predicted_labels = []
    all_prediction_probs = []
    all_patient_ids = []

    # parse optional topk/tau once
    _topk = int(topk) if (topk is not None and int(topk) > 0) else None
    _tau  = float(tau) if (tau is not None and float(tau) > 0) else None

    for fold, (tr_idx, te_idx) in enumerate(splits, start=1):
        train_pids = patients[tr_idx].tolist()
        test_pids  = patients[te_idx].tolist()

        print(f"\n[Fold {fold}/{len(splits)}] train={len(train_pids)} test={len(test_pids)} "
              f"test_events/pos={(y_pat[te_idx]==1).sum() if num_classes==2 else 'NA'}")

        # build fold datasets (slice by patient_id)
        train_dataset = PatientBagDataset(
            adata.copy()[adata.obs["patient_id"].isin(train_pids)],
            label_col=label_col,
            label_map=label_map,
        )
        test_dataset = PatientBagDataset(
            adata.copy()[adata.obs["patient_id"].isin(test_pids)],
            label_col=label_col,
            label_map=label_map,
        )

        # loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # save dir
        fold_save_path = os.path.join(save_path, f"fold_{fold:02d}")
        os.makedirs(fold_save_path, exist_ok=True)

        # model
        model = AttentionMIL(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=0.25,
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
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=12,     # loss 5 個 epoch 沒改善就降 LR
            factor=0.5,
            min_lr=1e-6,
            verbose=False,
        )
        print("[DEBUG] class_weights:", class_weights.detach().cpu().numpy())
        
        # ===== DEBUG: define positive class index consistently =====
        # If you define positive label as the one mapped to 1, then pos_idx = 1.
        pos_idx = 1
        neg_idx = 0
        print(f"[DEBUG] Using pos_idx={pos_idx} (positive class), neg_idx={neg_idx}")
        # ===========================================

        # ---- training ----
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        patience = 25          
        min_delta = 1e-5

        for epoch in range(num_epochs):
            model.train()
            train_correct, train_total = 0, 0
            train_loss = 0.0

            for batch in train_loader:
                if len(batch) == 4:
                    bags, batch_labels, _pids, one_hot = batch
                    one_hot = one_hot.to(device)
                else:
                    bags, batch_labels, _pids = batch
                    one_hot = None

                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device).long().view(-1)

                # if one_hot is not None:
                #     out = model(bags, sample_source=one_hot)
                # else:
                #     out = model(bags)
                out = model(bags)

                logits = out["logits"]
                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += float(loss.item())
                preds = torch.argmax(logits, dim=1)
                train_total += int(batch_labels.size(0))
                train_correct += int((preds == batch_labels).sum().item())

            train_loss /= max(len(train_loader), 1)
            train_acc = train_correct / train_total if train_total > 0 else 0.0

            scheduler.step(train_loss)
        
            if (epoch + 1) % 20 == 0:
                print(f"[Fold {fold}] ep={epoch+1}/{num_epochs} train_loss={train_loss:.4f} train_acc={train_acc:.4f}")

            # early stopping on train_loss (since no val set)
            if train_loss < best_train_loss - min_delta:
                best_train_loss = train_loss
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"[Fold {fold}] Early stopping at ep={epoch+1}")
                    break

        # load best
        model.load_state_dict(torch.load(os.path.join(fold_save_path, "best_model.pth"), map_location=device))
        model.eval()

        # ---- evaluate on test patients (OOF) ----
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 4:
                    bags, batch_labels, patient_ids, one_hot = batch
                    one_hot = one_hot.to(device)
                else:
                    bags, batch_labels, patient_ids = batch
                    one_hot = None

                patient_id = patient_ids[0]
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device).long().view(-1)

                want_attn = (aggregator == "attention")
                # if one_hot is not None:
                #     out = model(bags, sample_source=one_hot, return_attention=want_attn)
                # else:
                #     out = model(bags, return_attention=want_attn)
                out = model(bags)

                logits = out["logits"]
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                if want_attn and ("attn" in out):
                    attn_weights = out["attn"]
                    cv_results["attention_weights"][patient_id] = [
                        (w.cpu().numpy() if w is not None else None) for w in attn_weights
                    ]

                true_label = int(batch_labels.cpu().numpy()[0])
                pred_label = int(preds.cpu().numpy()[0])

                # ===== DEBUG: inspect logits/probs per test patient =====
                logits_np = logits.detach().cpu().numpy()[0]
                probs_np  = probs.detach().cpu().numpy()[0]
                print(f"[DEBUG][Fold {fold}] pid={patient_id} true={true_label} pred={pred_label} "
                    f"logits={np.round(logits_np, 4)} probs={np.round(probs_np, 4)} "
                    f"pos_prob(probs[{pos_idx}])={probs_np[pos_idx]:.4f}")
                # ===========================================

                if num_classes == 2:
                    pos_prob = float(probs_np[pos_idx])
                    all_prediction_probs.append(pos_prob)
                else:
                    all_prediction_probs.append(probs_np)

                all_true_labels.append(true_label)
                all_predicted_labels.append(pred_label)
                all_patient_ids.append(patient_id)

                cv_results["patient_predictions"][patient_id] = {
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "probabilities": probs.cpu().numpy()[0].tolist(),
                    "correct": (pred_label == true_label),
                    "fold": fold,
                }

                cv_results["fold_metrics"].append({
                    "patient_id": patient_id,
                    "fold": fold,
                    "accuracy": 1.0 if pred_label == true_label else 0.0,
                    "true_label": true_label,
                    "predicted_label": pred_label,
                    "prob_positive": float(probs.cpu().numpy()[0, 1]) if num_classes == 2 else None,
                })

    # ---- overall metrics (OOF) ----
    all_true_labels = np.array(all_true_labels, dtype=int)
    all_predicted_labels = np.array(all_predicted_labels, dtype=int)
    all_patient_ids = np.array(all_patient_ids, dtype=object)

    overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)

    if num_classes == 2:
        all_prediction_probs = np.array(all_prediction_probs, dtype=float)
        # ===== DEBUG AUC direction check =====
        auc_prob = roc_auc_score(all_true_labels, all_prediction_probs)
        auc_flip = roc_auc_score(all_true_labels, 1 - all_prediction_probs)

        print("\n[DEBUG] AUC(prob) =", auc_prob)
        print("[DEBUG] AUC(1-prob) =", auc_flip)
        print("[DEBUG] sum =", auc_prob + auc_flip)
        # =====================================
        overall_precision = precision_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, zero_division=0)
        overall_auc = roc_auc_score(all_true_labels, all_prediction_probs) if len(np.unique(all_true_labels)) > 1 else float("nan")
    else:
        overall_precision = precision_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_recall = recall_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, average="weighted", zero_division=0)
        overall_auc = float("nan")

    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)

    class_metrics = {}
    for c in range(num_classes):
        tp = np.sum((all_true_labels == c) & (all_predicted_labels == c))
        actual = np.sum(all_true_labels == c)
        predp = np.sum(all_predicted_labels == c)
        class_metrics[f"class_{c}"] = {
            "precision": tp / predp if predp > 0 else 0.0,
            "recall": tp / actual if actual > 0 else 0.0,
            "count": int(actual),
        }

    cv_results["overall_metrics"] = {
        "accuracy": float(overall_accuracy),
        "precision": float(overall_precision),
        "recall": float(overall_recall),
        "f1": float(overall_f1),
        "auc": float(overall_auc) if np.isfinite(overall_auc) else None,
        "confusion_matrix": conf_matrix.tolist(),
        "class_metrics": class_metrics,
    }

    print("\n===== CV Final Results =====")
    print(f"CV={cv} k={k if cv=='kfold' else 'NA'}")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({int((all_true_labels==all_predicted_labels).sum())}/{len(all_true_labels)})")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    if num_classes == 2:
        print(f"Overall AUC: {overall_auc:.4f}" if np.isfinite(overall_auc) else "Overall AUC: NA")
    print("\nConfusion Matrix:")
    print(conf_matrix)

    # save
    tag = "loocv" if cv == "loocv" else f"kfold{k}"
    results_path = os.path.join(save_path, f"{tag}_results.pkl")
    with open(results_path, "wb") as f:
        import pickle
        pickle.dump(cv_results, f)
    print(f"[INFO] Saved: {results_path}")

    return cv_results       

def run_pipeline_loocv(input_file, output_dir='results',
                       latent_dim=64, num_epochs_ae=200,
                       num_epochs=50, num_classes=2,
                       hidden_dim=128, sample_source_dim=None,
                       project_name="car-t-response", label_col="Response_3m", pos_label="R", neg_label="NR",
                       aggregator="attention", topk=0, tau=0.0, cv="loocv", k=5, seed=42):
    """run complete pipeline with leave one out cross validation
    
    Parameters:
    - input_file: path to input file
    - output_dir: directory to save results
    - latent_dim: dimension of latent space
    - num_epochs_ae: number of epochs for autoencoder
    - num_epoch_mil: number of epochs for MIL
    - num_classes: number of classes
    - hidden_dim: dimension of hidden layer

    Returns:
    - dict of results and models
    """

    # config = {
    #     "input_file": input_file,
    #     "output_dir": output_dir,
    #     "latent_dim": latent_dim,
    #     "num_epochs_ae": num_epochs_ae,
    #     "num_epochs_mil": num_epochs,
    #     "num_classes": num_classes,
    #     "hidden_dim": hidden_dim,
    #     "cv_method": "leave-one-out"
    # }

    # wandb.init(project=project_name, config=config)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"run_{timestamp}")
    ae_dir = os.path.join(result_dir, "autoencoder")
    mil_dir = os.path.join(result_dir, "mil")
    
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(mil_dir, exist_ok=True)
    
    
    # Step 1: Load and explore data
    print("\n" + "="*80)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*80)
    adata = load_and_explore_data(input_file)

    # wandb.config.update({
    #     "cells": adata.n_obs,
    #     "TFs": adata.n_vars,
    #     "patients": adata.obs["patient_id"].nunique()
    # })

    # if "Response_3m" in adata.obs.columns:
    #     wandb.config.update({
    #         "Response_distribution": dict(adata.obs["Response_3m"].value_counts())
    #     })

    # step 2: train autoencoder
    print("\n" + "="*80)
    print("STEP 2: TRAINING AUTOENCODER")
    print("="*80)
    train_loader, val_loader, test_loader, input_dim = preprocess_data(adata)

    # Step 3:train autoencoder
    print("\n" + "="*80)
    print("STEP 3: TRAINING AUTOENCODER")
    print("="*80)
    model, train_losses, val_losses = train_autoencoder(
            train_loader, val_loader, input_dim, latent_dim, num_epochs_ae, save_path=ae_dir
        )
    adata_latent, test_loss = evaluate_autoencoder(
        model, test_loader, adata, adata.var_names.tolist(), save_path=ae_dir
    )
    
    # adata_latent = adata.copy()
    # current_input_dim = adata.n_vars
    
    # Save latent representations
    latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
    adata_latent.write(latent_file)

    # ===== DEBUG: label distribution before LOOCV =====
    print("\n[DEBUG] Label distribution BEFORE LOOCV")

    print("[DEBUG] Cell-level label distribution:")
    print(adata_latent.obs[label_col].value_counts(dropna=False))

    print("\n[DEBUG] Patient-level label distribution:")
    print(
        adata_latent.obs
        .groupby("patient_id")[label_col]
        .first()
        .value_counts(dropna=False)
    )

    print("[DEBUG] Number of patients:",
        adata_latent.obs["patient_id"].nunique())
    print("===============================================\n")
    # ================================================

    # ===== DEBUG: CD19 relapse phenotype distribution =====
    # print("\n[DEBUG] CD19 phenotype (cell-level):")
    # print(adata_latent.obs["relapse_phenotype"].value_counts(dropna=False))

    # print("\n[DEBUG] CD19 phenotype (patient-level):")
    # print(
    #     adata_latent.obs
    #     .groupby("patient_id")["relapse_phenotype"]
    #     .first()
    #     .value_counts(dropna=False)
    # )
    # print("===============================================\n")

    # ===== Create binary CD19 relapse label =====
    if "relapse_phenotype" in adata_latent.obs.columns:
        rp = adata_latent.obs["relapse_phenotype"].astype(str).str.strip()

        adata_latent.obs["CD19_neg_relapse"] = rp.map({
            "CD19neg": 1,
            "CD19pos": 0
        })
    # ===========================================

    # Step 4: Run LOOCV
    print("\n" + "="*80)
    print("STEP 4: RUNNING LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*80)
    
    # Check if we have response information
    if label_col not in adata_latent.obs.columns:
        print(f"ERROR: '{label_col}' column not found in the data. Cannot proceed with MIL.")
        return None
    
    # Remove patients with NaN responses
    patients_with_missing = adata_latent.obs[adata_latent.obs[label_col].isna()]['patient_id'].unique()
    if len(patients_with_missing) > 0:
        print(f"Removing {len(patients_with_missing)} patients with missing responses")
        adata_latent = adata_latent[~adata_latent.obs['patient_id'].isin(patients_with_missing)].copy()
        
        # update wandb config
        # wandb.config.update({
        #     "patients_after_filtering": adata_latent.obs['patient_id'].nunique(),
        #     "cells_after_filtering": adata_latent.n_obs,
        #     "patients_removed": len(patients_with_missing)
        #     })
        
    cv_results = cross_validation_mil(
        adata_latent,
        input_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        sample_source_dim=sample_source_dim,
        num_epochs=num_epochs,
        save_path=mil_dir,
        label_col=label_col,
        aggregator=aggregator,
        topk=topk,
        tau=tau,
        cv=cv,
        k=k,
        seed=seed,
    )

    # wandb.finish()

    print(f"Pipeline completed successfully! Results saved to {result_dir}")

    return {
        'adata': adata,
        'autoencoder': model,
        'latent_data': adata_latent,
        'mil_results': cv_results,
        'results_dir': result_dir
    }


