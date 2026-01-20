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
                                  save_path='results'):
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
    full_dataset = PatientBagDataset(adata)

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
        

        # create train and test datasets
        train_dataset = PatientBagDataset(adata.copy()[adata.obs['patient_id'].isin(train_patients)])
        test_dataset = PatientBagDataset(adata.copy()[adata.obs['patient_id'] == test_patient])

        # create data loaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        # create save path for this fold
        fold_save_path = os.path.join(save_path, f'patient_{test_patient}')
        os.makedirs(fold_save_path, exist_ok=True)

        # train model
        model = AttentionMIL(input_dim, num_classes, hidden_dim, sample_source_dim).to(device)

        # use weight classes to address class imbalance
        y = adata.obs.Response_3m.to_numpy()
        y = np.where(y == "NR", 0,1)
    
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
        class_weights = torch.FloatTensor(class_weights).to(device)
        
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    

        # criterion = torch.nn.CrossEntropyLoss()
        # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        
        # Training setup
        best_train_loss = float("inf")
        epochs_without_improvement = 0
        patience = 8

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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            for bags, batch_labels, _, one_hot_sample_source in train_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)

                # Forward pass
                logits = model(bags, one_hot_sample_source)
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
            scheduler.step(train_loss)

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

            # early stopping
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                epochs_without_improvement = 0

                #save best model
                torch.save(model.state_dict(), os.path.join(fold_save_path, 'best_model.pth'))

            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f'Early stopping after {epoch+1} epochs')
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
            for bags, batch_labels, patient_ids, one_hot_sample_source in test_loader:
                bags = [bag.to(device) for bag in bags]
                batch_labels = batch_labels.to(device)
                one_hot_sample_source = one_hot_sample_source.to(device)

                # Forward pass
                logits, attn_weights = model(bags, one_hot_sample_source, return_attention=True)

                # get predictions
                probs = F.softmax(logits, dim=1)
                _, preds = torch.max(logits, 1)

                # convert to numpy
                preds_np = preds.cpu().numpy()
                labels_np = batch_labels.cpu().numpy()
                probs_np = probs.cpu().numpy()

                # Get the patient_id and store results
                patient_id = patient_ids[0]
                true_label = labels_np[0]
                pred_label = preds_np[0]
                pos_prob = probs_np[0, 1] if num_classes == 2 else None

                # Add to accumulators for global metrics
                all_true_labels.append(true_label)
                all_predicted_labels.append(pred_label)
                all_prediction_probs.append(pos_prob if num_classes ==2 else probs_np[0])
                all_patient_ids.append(patient_id)

                # Store patient-specific results
                cv_results['patient_predictions'][patient_id] = {
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'probabilities': probs_np[0].tolist(),
                    'correct': (pred_label == true_label)
                }
                
                    
                

                # Store attention weights
                cv_results['attention_weights'][patient_id] = [w.cpu().numpy() for w in attn_weights]

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
                       hidden_dim=128, sample_source_dim=4,
                       project_name="car-t-response"):
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
    
    # Save latent representations
    latent_file = os.path.join(ae_dir, "latent_representation.h5ad")
    adata_latent.write(latent_file)

    # Step 4: Run LOOCV
    print("\n" + "="*80)
    print("STEP 4: RUNNING LEAVE-ONE-OUT CROSS-VALIDATION")
    print("="*80)
    
    # Check if we have response information
    if 'Response_3m' not in adata_latent.obs.columns:
        print("ERROR: 'response' column not found in the data. Cannot proceed with MIL.")
        # wandb.finish()
        return None
    
    # Remove patients with NaN responses
    patients_with_missing = adata_latent.obs[adata_latent.obs['Response_3m'].isna()]['patient_id'].unique()
    if len(patients_with_missing) > 0:
        print(f"Removing {len(patients_with_missing)} patients with missing responses")
        adata_latent = adata_latent[~adata_latent.obs['patient_id'].isin(patients_with_missing)].copy()
        
        # update wandb config
        # wandb.config.update({
        #     "patients_after_filtering": adata_latent.obs['patient_id'].nunique(),
        #     "cells_after_filtering": adata_latent.n_obs,
        #     "patients_removed": len(patients_with_missing)
        #     })
        
    cv_results = leave_one_out_cross_validation(
        adata_latent, 
        input_dim = latent_dim,
        num_classes = num_classes, 
        hidden_dim = hidden_dim,
        sample_source_dim = sample_source_dim,
        num_epochs = num_epochs,
        save_path = mil_dir
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