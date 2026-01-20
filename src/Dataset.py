import scanpy as sc
import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load and explore the .h5ad file containing SCENIC AUC matrix
def load_and_explore_data(file_path):
    """
    Load the SCENIC AUC matrix from h5ad file and explore basic statistics as well as transform data range to [-1,1]
    """
    print("Loading SCENIC AUC matrix data...")
    adata = sc.read_h5ad(file_path)

    # subset to harad and deng datasets
    # adata = adata[adata.obs["Sample_source.y"] == "Deng"]
    adata = adata[~adata.obs['Response_3m'].isna()].copy()

    # shift adata.X to -0.5 to 0.5
    adata.X = (adata.X - 0.5) * 2

    # remove "Mono", "DC" and "other"
    
    # adata = adata[~adata.obs['cell_annotation'].isin(['Mono', 'DC', 'other'])].copy()
    # Print basic information
    print(f"Total cells: {adata.n_obs}")
    print(f"Number of TFs: {adata.n_vars}")
    
    # Check if we have patient information
    if 'patient_id' in adata.obs.columns:
        print(f"Number of patients: {adata.obs['patient_id'].nunique()}")
    
    # Check if we have response information
    if 'Response_3m' in adata.obs.columns:
        print("Response distribution:")
        print(adata.obs['Response_3m'].value_counts())
    
    # Basic stats on AUC values
    auc_matrix = adata.X
    if isinstance(auc_matrix, np.ndarray) == False:
        auc_matrix = auc_matrix.toarray()  # Convert sparse matrix if needed
    
    print(f"AUC matrix shape: {auc_matrix.shape}")
    print(f"AUC value range: [{np.min(auc_matrix)}, {np.max(auc_matrix)}]")
    print(f"AUC mean value: {np.mean(auc_matrix)}")
    
    return adata

# Data preprocessing function
def preprocess_data(adata, test_size=0.2, val_size=0.1, random_state=42): #42
    """Patient-level data split for autoencoder training
    
    Parameters:
    - adata: AnnData object with SCENIC results
    - test_size: Portion of data to use for testing
    - val_size: Portion of training data to use for validation
    - random_state: Random seed for reproducibility
    
    Returns:
    - train_loader: DataLoader for training
    - val_loader: DataLoader for validation
    - test_loader: DataLoader for testing
    - input_dim: Input dimension (number of TFs)
    """
    print("Preprocessing data for autoencoder training...")

    # get unique patients
    patients = adata.obs["patient_id"].unique()

    # split patients into train+val and test
    patients_train_val, patients_test = train_test_split(
        patients, test_size=test_size, random_state=random_state
    )

    # Further split train+val into train and val
    patients_train, patients_val = train_test_split(
        patients_train_val, test_size=val_size/(1-test_size), random_state=random_state
    )

    # select cells based on patient assignment
    train_mask = adata.obs["patient_id"].isin(patients_train)
    val_mask = adata.obs["patient_id"].isin(patients_val)
    test_mask = adata.obs["patient_id"].isin(patients_test)
    
    X_train = adata.X[train_mask]
    X_val = adata.X[val_mask]
    X_test = adata.X[test_mask]
    
    # Get AUC matrix and convert to dense if needed
    
    if isinstance(X_train, np.ndarray) == False:
        # X = X.toarray()
        X_train = X_train.toarray()
        X_val = X_val.toarray()
        X_test = X_test.toarray()
    
    print(f"Training set: {X_train.shape[0]} cells")
    print(f"Validation set: {X_val.shape[0]} cells")
    print(f"Test set: {X_test.shape[0]} cells")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_val_tensor = torch.FloatTensor(X_val)
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Create DataLoaders (input and target are the same for autoencoders)
    train_dataset = TensorDataset(X_train_tensor, X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, X_test_tensor)
    
    batch_size = 256  # Adjust based on your GPU memory
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    input_dim = X_train.shape[1]  # Number of TFs (should be 154)
    
    return train_loader, val_loader, test_loader, input_dim

# 1. MIL Dataset class for patient bags
class PatientBagDataset(Dataset):
    """
    Dataset for Multiple Instance Learning where each bag contains cells from one patient
    """
    def __init__(self, adata, patient_col='patient_id', label_col='Response_3m'):
        """
        Initialize the MIL dataset
        
        Parameters:
        - adata: AnnData object with latent representations and patient information
        - patient_col: Column name for patient identifiers
        - label_col: Column name for patient response labels
        """
        self.adata = adata
        self.patient_col = patient_col
        self.label_col = label_col
        
        # Get unique patients
        self.patients = adata.obs[patient_col].unique()
        
        # Create a mapping of patient to label
        self.patient_to_label = dict(zip(
            adata.obs[patient_col], 
            adata.obs[label_col]
        ))

        #############################################
        

        if "Sample_source" in adata.obs.columns:
            # Create one-hot encoding for Sample_source using unique values
            sample_sources = list(adata.obs["Sample_source"].unique())
            self.patient_metadata = {}
            
            # Get Sample_source for each patient
            patient_source_map = adata.obs.groupby(self.patient_col, observed=True)['Sample_source'].first().to_dict()
            
            # Create one-hot encoding for each patient
            for patient in self.patients:
                if patient in patient_source_map:
                    source = patient_source_map[patient]
                    one_hot = [1 if s == source else 0 for s in sample_sources]
                    self.patient_metadata[patient] = one_hot
            


        ##############################################
        
        # Create patient bags
        self.patient_bags = {}
        self.patient_labels = {}
        
        for patient in self.patients:
            # Get indices for this patient
            indices = np.where(adata.obs[patient_col] == patient)[0]
            
            # Get features for this patient's cells
            patient_data = adata.X[indices]
            if isinstance(patient_data, np.ndarray) == False:
                patient_data = patient_data.toarray()
                
            self.patient_bags[patient] = patient_data
            
            # Get label for this patient
            # Assuming all cells from the same patient have the same label
            label = self.patient_to_label[patient]
            self.patient_labels[patient] = label
        
        # Convert patients to a list for indexing
        self.patient_list = list(self.patients)
    
    def __len__(self):
        """Return the number of bags (patients)"""
        return len(self.patient_list)
    
    def __getitem__(self, idx):
        """
        Get a patient bag
        
        Returns:
        - bag: Tensor of shape [num_instances, features] for the patient
        - label: Label for the patient
        - patient: Patient identifier
        """
        patient = self.patient_list[idx]
        bag = self.patient_bags[patient]
        label = self.patient_labels[patient]
        
        # Convert to proper format
        bag = torch.FloatTensor(bag)
        
        # Handle different label types
        if isinstance(label, str):
            # Map string labels to integers
            
            label_map = {"NR":0, "OR":1}
            label = label_map[label]
        
        label = torch.tensor(label, dtype=torch.long)

        if self.patient_metadata:
            one_hot_sample_source = torch.tensor(self.patient_metadata[patient], dtype=torch.float)
            return bag, label, patient, one_hot_sample_source
        else:
            return bag, label, patient
