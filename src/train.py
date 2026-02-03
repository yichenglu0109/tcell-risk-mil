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



# define a leave one out cross validation function
def leave_one_out_cross_validation(adata, input_dim, num_classes=2, hidden_dim=128, sample_source_dim=4,
                                  num_epochs=50, learning_rate=5e-4, weight_decay = 1e-2, 
                                  save_path='results', label_col="Response_3m", pos_label="R", neg_label="NR", aggregator="attention", topk=0, tau=0.0):
    """
    Perform leave-one-out cross-validation for the MIL model
    # Parameters:
    - adata: AnnData object with latent representations
    - input_dim: Input dimension (latent space dimension)
    - num_classes: Number of response classes
    - hidden_dim: Dimension of hidden layer
    - num_epochs: Maximum number of training epochs
    - learning_rate: Learning rate for optimizer
    - weight_decay: L2 regularization strength

    Returns:
    - cv_results: Dictionary of cross-validation results
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(save_path, exist_ok=True)

    # initiate wandb
    # wandb.init(project="car-t-IP-MIL", name="loocv-mil",
    #            config={
    #                "input_dim": input_dim,
    #                "num_classes": num_classes,
    #                "hidden_dim": hidden_dim,
    #                "num_epochs": num_epochs,
    #                "learning_rate": learning_rate,
    #                "weight_decay": weight_decay,
    #                "save_path": save_path
    #            })
    
    # create dataset
    full_dataset = PatientBagDataset(adata, label_col=label_col)

    # ---- FIX 1: freeze label_map from full_dataset so every fold uses the same mapping ----
    label_map = full_dataset._label_to_int  # e.g. {"CD19neg":0, "CD19pos":1}
    if num_classes == 2:
        # optional safety: print mapping once
        print("[INFO] Global label_map:", label_map)

    sample_source_dim = (
        full_dataset.sample_source_dim
        if hasattr(full_dataset, "sample_source_dim")
        else None
    )

    # get all patients and their labels
    patients = np.array(full_dataset.patient_list)
    labels = np.array([full_dataset.patient_labels[p] for p in patients])

    print(f"Performing leave-one-out cross-validation for {len(patients)} patients...")

    cv_results = {
        'fold_metrics': [], # store per-fold metrics for reference
        'patient_predictions': {},
        'attention_weights': {}
    }

    # Initialize accumulators for all predictions across folds
    all_true_labels = []
    all_predicted_labels = []
    all_prediction_probs = []
    all_patient_ids = []

    # wandb_patient_table = wandb.Table(columns=["patient_id", "true_label", "predicted_label", "predicted_label", "correct"])

    # all_metrics = []
    # all_confusion_matrices = np.zeros((num_classes, num_classes))


    # LOOCV loop
    for i, test_patient in enumerate(patients):

        print(f"Fold {i+1}/{len(patients)} patients, testing on {test_patient}...")
        train_patients = np.array([p for p in patients if p != test_patient])
        

        # create train and test datasets (Fixed 2: pass label_map to ensure consistent mapping)
        train_dataset = PatientBagDataset(
            adata.copy()[adata.obs['patient_id'].isin(train_patients)],
            label_col=label_col,
            label_map=label_map
        )
        test_dataset = PatientBagDataset(
            adata.copy()[adata.obs['patient_id'] == test_patient],
            label_col=label_col,
            label_map=label_map
        )

        # ===== DEBUG: patient-level labels =====
        labels_debug = [train_dataset.patient_labels[p] for p in train_dataset.patient_list]
        labels_debug = np.array(labels_debug, dtype=str)

        unique, counts = np.unique(labels_debug, return_counts=True)
        print("[DEBUG] patient-level label distribution:")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c}")
        # ======================================

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # create save path for this fold
        fold_save_path = os.path.join(save_path, f'patient_{test_patient}')
        os.makedirs(fold_save_path, exist_ok=True)

        # parse optional topk and tau
        _topk = int(topk) if (topk is not None and int(topk) > 0) else None
        _tau  = float(tau) if (tau is not None and float(tau) > 0) else None

        # train model
        model = AttentionMIL(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=0.25, sample_source_dim=sample_source_dim, aggregator=aggregator, topk=_topk, tau=_tau).to(device)

        # use class weights to address imbalance (compute from TRAIN fold only)
        # ---- FIX 3: compute class weights from TRAIN PATIENTS (patient-level), using frozen label_map ----
        # y: patient-level int labels in TRAIN fold
        y_pat_raw = np.array([train_dataset.patient_labels[p] for p in train_dataset.patient_list], dtype=str)
        y = np.array([label_map[v] for v in y_pat_raw], dtype=int)

        classes_present = np.unique(y)
        if len(classes_present) < num_classes:
            # 這個 fold 訓練集只有單一類：class weighting 沒意義，直接給全 1
            class_weights = torch.ones(num_classes, device=device)
        else:
            cw = compute_class_weight(class_weight="balanced", classes=np.arange(num_classes), y=y)
            class_weights = torch.tensor(cw, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        print("[DEBUG] class_weights (w0,w1):", class_weights.detach().cpu().numpy())

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)
    

        # criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        
        # Training setup
        best_train_acc = -1.0
        epochs_without_improvement = 0
        patience = 20
        min_delta = 1e-4

        history = {
            'train_loss': [],
            'train_acc': []
            
        }

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for batch in train_loader:
                # 支援 Dataset 回傳 (bag, label, patient) 或 (bag, label, patient, one_hot)
                if len(batch) == 4:
                    bags, batch_labels, _, one_hot_sample_source = batch
                    one_hot_sample_source = one_hot_sample_source.to(device)
                else:
                    bags, batch_labels, _ = batch
                    one_hot_sample_source = None

                bags = [bag.to(device) for bag in bags]
                if aggregator == "pseudobulk":
                    bags = [bag.mean(dim=0) for bag in bags]   # [input_dim]

                batch_labels = batch_labels.to(device).long().view(-1)
                
                # ===== DEBUG: inspect MIL input features (only once) =====
                if epoch == 0:
                    x0 = bags[0]  # [num_instances, feature_dim]
                    print("[DEBUG] bag0 feature stats:",
                        "shape=", tuple(x0.shape),
                        "mean=", x0.mean().item(),
                        "std=", x0.std().item(),
                        "min=", x0.min().item(),
                        "max=", x0.max().item())
                # =========================================================

                # Forward pass（有 sample_source 才傳）
                if one_hot_sample_source is not None:
                    out = model(bags, sample_source=one_hot_sample_source)
                else:
                    out = model(bags)

                logits = out["logits"]
                loss = criterion(logits, batch_labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                _, preds = torch.max(logits.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (preds == batch_labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0

            # record history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)


            # use training loss for scheduler and early stopping since we have no validation set
            # scheduler.step(train_loss)

            # print progress every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f'Patient {test_patient} - Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')

            # log metrics
            # wandb.log({
            #     f"patient_{test_patient}/epoch": epoch + 1,
            #     f"patient_{test_patient}/train_loss": train_loss,
            #     f"patient_{test_patient}/train_acc": train_acc,
            #     f"patient_{test_patient}/learning_rate": optimizer.param_groups[0]['lr'],
            # })

            # early stopping + best checkpoint (use train_acc, not train_loss)
            if train_acc > best_train_acc + min_delta:
                best_train_acc = train_acc
                epochs_without_improvement = 0
                torch.save(model.state_dict(), os.path.join(fold_save_path, "best_model.pth"))
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping after {epoch+1} epochs")
                    break

        # load best model
        model.load_state_dict(torch.load(os.path.join(fold_save_path, 'best_model.pth')))

        # evaluate on test patient
        model.eval()
        device = next(model.parameters()).device

        # test_preds = []
        # test_labels = []
        # test_probs = []
        # patient_attentions = {}

        with torch.no_grad():
            for batch in test_loader:
                # 支援 Dataset 回傳 (bag, label, patient) 或 (bag, label, patient, one_hot)
                if len(batch) == 4:
                    bags, batch_labels, patient_ids, one_hot_sample_source = batch
                    one_hot_sample_source = one_hot_sample_source.to(device)
                else:
                    bags, batch_labels, patient_ids = batch
                    one_hot_sample_source = None
                
                patient_id = patient_ids[0]  

                bags = [bag.to(device) for bag in bags]
                if aggregator == "pseudobulk":
                    bags = [bag.mean(dim=0) for bag in bags]   # [input_dim]
                batch_labels = batch_labels.to(device).long().view(-1)

                # Forward pass（有 sample_source 才傳）
                want_attn = (aggregator == "attention")

                if one_hot_sample_source is not None:
                    out = model(bags, sample_source=one_hot_sample_source, return_attention=want_attn)
                else:
                    out = model(bags, return_attention=want_attn)

                logits = out["logits"]
                if want_attn and ("attn" in out):
                    attn_weights = out["attn"]
                    cv_results['attention_weights'][patient_id] = [
                        (w.cpu().numpy() if w is not None else None) for w in attn_weights
                    ]

                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                preds_np = preds.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()
                probs_np = probs.cpu().numpy()

                true_label = labels_np[0]
                pred_label = preds_np[0]
                pos_prob = probs_np[0, 1] if num_classes == 2 else None

                # ===== DEBUG: why always predict class 1 =====
                p1 = probs_np[0, 1]
                l0 = logits[0, 0].item()
                l1 = logits[0, 1].item()
                print(f"[DEBUG] test patient={patient_id} true={true_label} pred={pred_label} "
                    f"p1={p1:.3f} (l1-l0={l1-l0:.3f})")
                # ============================================

                all_true_labels.append(true_label)
                all_predicted_labels.append(pred_label)
                all_prediction_probs.append(pos_prob if num_classes == 2 else probs_np[0])
                all_patient_ids.append(patient_id)

                cv_results['patient_predictions'][patient_id] = {
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'probabilities': probs_np[0].tolist(),
                    'correct': (pred_label == true_label)
                }

                # Add to wandb table
                # wandb_patient_table.add_data(
                #     patient_id, 
                #     int(true_label), 
                #     int(pred_label), 
                #     float(pos_prob) if num_classes == 2 else 'N/A',
                #     bool(pred_label == true_label)
                # )

                
        # Calculate per-fold metrics for individual patient (for monitoring only)
        fold_correct = (preds_np[0] == labels_np[0])
        fold_metrics = {
            'patient_id': patient_id,
            'fold': i,
            'accuracy': 1.0 if fold_correct else 0.0,
            'true_label': int(labels_np[0]),
            'predicted_label': int(preds_np[0]),
            'prob_positive': float(probs_np[0, 1]) if num_classes == 2 else None
        }
        cv_results['fold_metrics'].append(fold_metrics)
        
        # Log patient result to wandb
        # wandb.log({
        #     f"patient_{patient_id}/true_label": labels_np[0],
        #     f"patient_{patient_id}/predicted_label": preds_np[0],
        #     f"patient_{patient_id}/prob_positive": probs_np[0, 1] if num_classes == 2 else None,
        #     f"patient_{patient_id}/correct": fold_correct
        # })

    # Convert accumulators to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    all_prediction_probs = np.array(all_prediction_probs)
    all_patient_ids = np.array(all_patient_ids)
    
    # Calculate overall metrics only once using ALL predictions
    overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
    
    # For binary classification
    if num_classes == 2:
        overall_precision = precision_score(all_true_labels, all_predicted_labels)
        overall_recall = recall_score(all_true_labels, all_predicted_labels)
        overall_f1 = f1_score(all_true_labels, all_predicted_labels)
        overall_auc = roc_auc_score(all_true_labels, all_prediction_probs)
    else:
        # For multiclass classification
        overall_precision = precision_score(all_true_labels, all_predicted_labels, average='weighted')
        overall_recall = recall_score(all_true_labels, all_predicted_labels, average='weighted')
        overall_f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        # For multiclass AUC, we'd need to calculate it differently (beyond scope here)
        overall_auc = 0.5  
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
    
    # Calculate class-specific metrics
    class_metrics = {}
    for c in range(num_classes):
        true_positives = np.sum((all_true_labels == c) & (all_predicted_labels == c))
        actual_positives = np.sum(all_true_labels == c)
        predicted_positives = np.sum(all_predicted_labels == c)
        
        class_metrics[f'class_{c}'] = {
            'precision': true_positives / predicted_positives if predicted_positives > 0 else 0,
            'recall': true_positives / actual_positives if actual_positives > 0 else 0,
            'count': int(actual_positives)
        }
    
    # Store final metrics in results
    cv_results['overall_metrics'] = {
        'accuracy': float(overall_accuracy),
        'precision': float(overall_precision),
        'recall': float(overall_recall),
        'f1': float(overall_f1),
        'auc': float(overall_auc),
        'confusion_matrix': conf_matrix.tolist(),
        'class_metrics': class_metrics
    }
    
    # Log final results to wandb
    # wandb.log({
    #     "overall_accuracy": overall_accuracy,
    #     "overall_precision": overall_precision,
    #     "overall_recall": overall_recall,
    #     "overall_f1": overall_f1,
    #     "overall_auc": overall_auc,
    #     "patient_results": wandb_patient_table
    # })
    
    # Print final results
    print("\n===== LOOCV Final Results =====")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({np.sum(all_predicted_labels == all_true_labels)}/{len(all_true_labels)} patients correct)")
    print(f"Overall Precision: {overall_precision:.4f}")
    print(f"Overall Recall: {overall_recall:.4f}")
    print(f"Overall F1 Score: {overall_f1:.4f}")
    print(f"Overall AUC: {overall_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClass-specific Metrics:")
    for class_name, metrics in class_metrics.items():
        print(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, Count={metrics['count']}")
    
    # Save results to file
    results_path = os.path.join(save_path, 'loocv_results.pkl')
    with open(results_path, 'wb') as f:
        import pickle
        pickle.dump(cv_results, f)
    
    return cv_results
       
   

def run_pipeline_loocv(input_file, output_dir='results',
                       latent_dim=64, num_epochs_ae=200,
                       num_epochs=50, num_classes=2,
                       hidden_dim=128, sample_source_dim=None,
                       project_name="car-t-response", label_col="Response_3m", pos_label="R", neg_label="NR",
                       aggregator="attention", topk=0, tau=0.0):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(output_dir, f"run_{timestamp}")
    ae_dir = os.path.join(result_dir, "autoencoder")
    mil_dir = os.path.join(result_dir, "mil")

    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(ae_dir, exist_ok=True)
    os.makedirs(mil_dir, exist_ok=True)

    # -------------------------
    # Step 1: Load
    # -------------------------
    print("\n" + "="*80)
    print("STEP 1: LOADING AND EXPLORING DATA")
    print("="*80)
    adata = load_and_explore_data(input_file)

    # ✅ CHECKPOINT 1: raw
    debug_patient_labels(
        adata, "C1_raw_loaded",
        label_cols=[label_col, "relapse_gt200", "relapse_gt300", "relapse_phenotype"],
        patient_col="patient_id",
        sample_col="sample_source",   # 若你的欄位叫 Sample_source，改這裡
    )

    # -------------------------
    # Step 2/3: AE train + latent
    # -------------------------
    print("\n" + "="*80)
    print("STEP 2: TRAINING AUTOENCODER (split loaders)")
    print("="*80)
    train_loader, val_loader, test_loader, input_dim = preprocess_data(adata)

    print("\n" + "="*80)
    print("STEP 3: TRAINING AUTOENCODER (fit)")
    print("="*80)
    model, train_losses, val_losses = train_autoencoder(
        train_loader, val_loader, input_dim, latent_dim, num_epochs_ae, save_path=ae_dir
    )

    adata_latent, test_loss = evaluate_autoencoder(
        model, test_loader, adata, adata.var_names.tolist(), save_path=ae_dir
    )

    latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
    adata_latent.write(latent_file)

    # ✅ CHECKPOINT 2: latent
    debug_patient_labels(
        adata_latent, "C2_after_latent",
        label_cols=[label_col, "relapse_gt200", "relapse_gt300", "relapse_phenotype"],
        patient_col="patient_id",
        sample_col="sample_source",
    )

    # -------------------------
    # Your existing debug blocks (keep if you want)
    # -------------------------
    print("\n[DEBUG] Label distribution BEFORE LOOCV")
    print("[DEBUG] Cell-level label distribution:")
    print(adata_latent.obs[label_col].value_counts(dropna=False))
    print("\n[DEBUG] Patient-level label distribution:")
    print(
        adata_latent.obs
        .groupby("patient_id", observed=True)[label_col]
        .first()
        .value_counts(dropna=False)
    )
    print("[DEBUG] Number of patients:", adata_latent.obs["patient_id"].nunique())
    print("===============================================\n")

    # -------------------------
    # Optional: create CD19_neg_relapse
    # -------------------------
    if "relapse_phenotype" in adata_latent.obs.columns:
        rp = adata_latent.obs["relapse_phenotype"].astype(str).str.strip()
        adata_latent.obs["CD19_neg_relapse"] = rp.map({"CD19neg": 1, "CD19pos": 0})

    # ✅ CHECKPOINT 3: after making new label (should not change patient set)
    debug_patient_labels(
        adata_latent, "C3_after_cd19_mapping",
        label_cols=[label_col, "relapse_gt200", "relapse_gt300", "CD19_neg_relapse"],
        patient_col="patient_id",
        sample_col="sample_source",
    )

    # -------------------------
    # Step 4: LOOCV
    # -------------------------
    print("\n" + "="*80)
    print("STEP 4: RUNNING LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*80)

    if label_col not in adata_latent.obs.columns:
        print(f"ERROR: '{label_col}' column not found in the data. Cannot proceed with MIL.")
        return None

    # ✅ 在 filter missing 之前先記住病人集合（用來找誰被砍）
    before_pids = set(adata_latent.obs["patient_id"].astype(str).unique())

    patients_with_missing = adata_latent.obs[adata_latent.obs[label_col].isna()]["patient_id"].unique()
    if len(patients_with_missing) > 0:
        print(f"[INFO] Removing {len(patients_with_missing)} patients with missing {label_col}")
        adata_latent = adata_latent[~adata_latent.obs["patient_id"].isin(patients_with_missing)].copy()

    after_pids = set(adata_latent.obs["patient_id"].astype(str).unique())
    dropped = sorted(list(before_pids - after_pids))

    # ✅ CHECKPOINT 4: after filtering
    debug_patient_labels(
        adata_latent, "C4_after_missing_filter",
        label_cols=[label_col, "relapse_gt200", "relapse_gt300"],
        patient_col="patient_id",
        sample_col="sample_source",
    )
    if len(dropped) > 0:
        print("[DEBUG] dropped patients due to missing label (first 30):")
        print(dropped[:30])

    # ✅ HARD ASSERT (可選)：你預期 diff=10，但如果這裡不是 10 就代表前面某步動到你不想動的欄位
    # 你可以先用 print，不想直接 crash 就把 assert 註解掉
    # （建議先開著，逼你抓到 stage）
    pt = adata_latent.obs.groupby("patient_id", observed=True)[["relapse_gt200","relapse_gt300"]].first()
    a = pt["relapse_gt200"]; b = pt["relapse_gt300"]
    both = a.notna() & b.notna()
    diff = int((a[both] != b[both]).sum())
    print(f"[ASSERT-CHECK] diff among valid (gt200 vs gt300) right before LOOCV = {diff}")
    # assert diff == 10, f"Expected diff=10, but got {diff}. A filtering/subsetting step changed your cohort."

    # ✅ 最重要：把 label_col 換成 relapse_gt300 時，你要在這裡確認 notna 病人數
    # 如果你跑 label_col="relapse_gt300"，這裡應該顯示 notna patients=51
    # 否則就是：某一步讓 relapse_gt300 大量變 NaN 或被移除。

    cv_results = leave_one_out_cross_validation(
        adata_latent,
        input_dim=latent_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        sample_source_dim=sample_source_dim,
        num_epochs=num_epochs,
        save_path=mil_dir,
        label_col=label_col,
        pos_label=pos_label,
        neg_label=neg_label,
        aggregator=aggregator,
        topk=topk,
        tau=tau,
    )

    print(f"Pipeline completed successfully! Results saved to {result_dir}")

    return {
        "adata": adata,
        "autoencoder": model,
        "latent_data": adata_latent,
        "mil_results": cv_results,
        "results_dir": result_dir,
    }


