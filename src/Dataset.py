import scanpy as sc
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

# 1. Load and explore the .h5ad file containing SCENIC AUC matrix
def load_and_explore_data(file_path):
    """
    Load the SCENIC AUC matrix from h5ad file and explore basic statistics as well as transform data range to [-1,1]
    """
    print("Loading SCENIC AUC matrix data...")
    adata = sc.read_h5ad(file_path)

    # shift adata.X to -0.5 to 0.5
    adata.X = (adata.X - 0.5) * 2

    # Print basic information
    print(f"Total cells: {adata.n_obs}")
    print(f"Number of TFs: {adata.n_vars}")
    
    # Check if we have patient information
    if 'patient_id' in adata.obs.columns:
        print(f"Number of patients: {adata.obs['patient_id'].nunique()}")

    # List available metadata columns (for later task selection)
    print("Available obs columns:", list(adata.obs.columns))
    
    # Basic stats on AUC values
    auc_matrix = adata.X
    if not isinstance(auc_matrix, np.ndarray):
        auc_matrix = auc_matrix.toarray()  
    
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

    Supports: 
    - binary / multiclass classification 
    - regression 
    - survival (Cox): returns (time, event)
    """
    def __init__(
        self, 
        adata, 
        patient_col='patient_id', 
        task_type="classification",  # "classification" | "regression" | "survival"
        label_col="Response_3m",  # for classification/regression
        time_col=None,  # for survival
        event_col=None,  # for survival
        label_map=None,  # optional: {"NR":0,"OR":1} or custom
        drop_missing=True,
        use_sample_source=True,  # keep your one-hot covariate
        sample_source_col="Sample_source",
        pooling=None 
    ):
        """
        Initialize the MIL dataset
        
        Parameters:
        - adata: AnnData object with latent representations and patient information
        - patient_col: Column name for patient identifiers
        - label_col: Column name for patient response labels
        """
        self.adata = adata
        self.patient_col = patient_col
        self.task_type = task_type
        self.label_col = label_col
        self.time_col = time_col
        self.event_col = event_col
        self.label_map = label_map
        self.drop_missing = drop_missing
        self.pooling = pooling

        # Always define metadata dict (avoid attribute missing)
        self.patient_metadata = {}
        self.sample_sources = None

        # # ---- sanity checks ----
        # if self.patient_col not in adata.obs.columns:
        #     raise ValueError(f"'{self.patient_col}' not found in adata.obs")

        # if self.task_type not in {"classification", "regression", "survival"}:
        #     raise ValueError("task_type must be one of: 'classification', 'regression', 'survival'")

        # if self.task_type in {"classification", "regression"}:
        #     if self.label_col is None:
        #         raise ValueError("For classification/regression, label_col must be provided.")
        #     if self.label_col not in adata.obs.columns:
        #         raise ValueError(f"'{self.label_col}' not found in adata.obs")

        # if self.task_type == "survival":
        #     if self.time_col is None or self.event_col is None:
        #         raise ValueError("For survival, both time_col and event_col must be provided.")
        #     if self.time_col not in adata.obs.columns:
        #         raise ValueError(f"'{self.time_col}' not found in adata.obs")
        #     if self.event_col not in adata.obs.columns:
        #         raise ValueError(f"'{self.event_col}' not found in adata.obs")

        # ---- determine patient list (and drop missing if requested) ----
        all_patients = adata.obs[self.patient_col].astype(str).unique()

        def patient_has_valid_label(patient_id: str) -> bool:
            rows = adata.obs[self.patient_col].astype(str) == patient_id
            if self.task_type in {"classification", "regression"}:
                vals = adata.obs.loc[rows, self.label_col]
                return vals.notna().any()
            else:
                tvals = adata.obs.loc[rows, self.time_col]
                evals = adata.obs.loc[rows, self.event_col]
                # need at least one non-missing pair
                return (tvals.notna() & evals.notna()).any()

        if drop_missing:
            kept = []
            dropped = []
            for p in all_patients:
                if patient_has_valid_label(p):
                    kept.append(p)
                else:
                    dropped.append(p)
            if len(dropped) > 0:
                warnings.warn(f"Dropping {len(dropped)} patients with missing label/time/event")
            self.patients = np.array(kept, dtype=object)
        else:
            self.patients = np.array(all_patients, dtype=object)

        # ---- sample source one-hot (optional) ----
        if use_sample_source and (sample_source_col in adata.obs.columns):
            sources = list(adata.obs[sample_source_col].astype(str).unique())
            patient_source_map = (
                adata.obs.assign(_pid=adata.obs[self.patient_col].astype(str))
                .groupby("_pid", observed=True)[sample_source_col]
                .first()
                .astype(str)
                .to_dict()
            )
            for p in self.patients:
                src = patient_source_map.get(p, None)
                if src is None:
                    continue
                one_hot = [1.0 if s == src else 0.0 for s in sources]
                self.patient_metadata[p] = one_hot

            self.sample_source_dim = len(sources)
        else:
            self.sample_source_dim = None

        ##############################################
        # ---- create patient to label mapping ----
        ##############################################
        self.patient_bags = {}
        self.patient_labels = {}
        
        for patient in self.patients:
            # Get indices for this patient
            indices = np.where(adata.obs[self.patient_col].astype(str).to_numpy() == patient)[0]
            
            patient_data = adata.X[indices]
            if not isinstance(patient_data, np.ndarray):
                patient_data = patient_data.toarray()
            self.patient_bags[patient] = patient_data

            # patient-level label
            if self.task_type in {"classification", "regression"}:
                # assume patient label is consistent; take first non-missing
                series = adata.obs.iloc[indices][self.label_col]
                label_val = series.dropna().iloc[0] if series.notna().any() else None
                self.patient_labels[patient] = label_val
            else:
                obs_sub = adata.obs.iloc[indices][[self.time_col, self.event_col]]
                obs_sub = obs_sub.dropna()
                if len(obs_sub) == 0:
                    self.patient_labels[patient] = (None, None)
                else:
                    t = float(obs_sub[self.time_col].iloc[0])
                    e = float(obs_sub[self.event_col].iloc[0])
                    self.patient_labels[patient] = (t, e)
        
        # For classification: create mapping if needed (string -> int)
        if self.task_type == "classification":
            raw_labels = [self.patient_labels[p] for p in self.patients]
            # apply explicit map if provided
            if self.label_map is not None:
                self._label_to_int = dict(self.label_map)
            else:
                # auto factorize (stable order)
                uniq = sorted(list({str(x) for x in raw_labels}))
                self._label_to_int = {lab: i for i, lab in enumerate(uniq)}

            self.num_classes = len(set(self._label_to_int.values()))
        else:
            self._label_to_int = None
            self.num_classes = None

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
        bag = torch.FloatTensor(self.patient_bags[patient])
        
        if self.task_type == "classification":
            lab = self.patient_labels[patient]
            if isinstance(lab, str):
                key = lab
            else:
                key = str(lab)
            y = self._label_to_int[key]
            label = torch.tensor(y, dtype=torch.long)

        elif self.task_type == "regression":
            lab = self.patient_labels[patient]
            label = torch.tensor(float(lab), dtype=torch.float32)

        else:  # survival
            t, e = self.patient_labels[patient]
            time = torch.tensor(float(t), dtype=torch.float32)
            event = torch.tensor(float(e), dtype=torch.float32)
            label = (time, event)
        
        # sample source covariate
        if self.sample_source_dim is not None and (patient in self.patient_metadata):
            one_hot = torch.tensor(self.patient_metadata[patient], dtype=torch.float32)
            return bag, label, patient, one_hot
        else:
            return bag, label, patient

DAYS_PER_MONTH = 30.44  # average days per month

def _first_non_null(series):
    s = series.dropna()
    return s.iloc[0] if len(s) > 0 else np.nan

def build_patient_survival_table(
    obs,
    patient_col="patient_id",
    relapse_col="relapse_y_n",
    ttr_col="time_to_relapse_days",
    fu_col="follow_up_duration_months",
    days_per_month=DAYS_PER_MONTH,
    drop_inconsistent=True,
):
    """
    Returns:
      patients: list[str]
      time_days: np.ndarray shape [N]
      event: np.ndarray shape [N] (0/1)
    Does NOT modify adata.
    """
    tmp = obs[[patient_col, relapse_col, ttr_col, fu_col]].copy()
    tmp[patient_col] = tmp[patient_col].astype(str)

    byp = tmp.groupby(patient_col, observed=True).agg({
        relapse_col: _first_non_null,
        ttr_col: _first_non_null,
        fu_col: _first_non_null,
    })

    # normalize relapse to 0/1; anything else (e.g., N/A) -> NaN
    r = byp[relapse_col].astype(str).str.strip().str.upper()

    r = r.map({
        "YES": 1, "Y": 1, "TRUE": 1, "T": 1, "1": 1,
        "NO": 0,  "N": 0, "FALSE": 0, "F": 0, "0": 0,
        "N/A": np.nan, "NA": np.nan, "NAN": np.nan, "NONE": np.nan, "": np.nan,
    })

    r = pd.to_numeric(r, errors="coerce")  # 變成 float (0/1/NaN)

    ttr = byp[ttr_col].astype(float)
    fu_m = byp[fu_col].astype(float)
    fu_days = fu_m * float(days_per_month)

    # ✅ event: r==1 ->1, r==0 ->0, r==NaN -> NaN
    event = np.where(r == 1.0, 1,
            np.where(r == 0.0, 0, np.nan)).astype(float)

    # time_days: event==1 用 ttr；event==0 用 fu_days；event==NaN -> NaN
    time_days = np.where(event == 1.0, ttr.to_numpy(),
                np.where(event == 0.0, fu_days.to_numpy(), np.nan)).astype(float)

    # keep only valid rows (drops N/A automatically)
    keep = np.isfinite(time_days) & np.isfinite(event)

    event = event[keep].astype(int)
    time_days = time_days[keep].astype(float)

    # optional QC: if event==1 and fu exists but fu_days < ttr -> inconsistent
    if drop_inconsistent:
        bad = np.isfinite(fu_days.to_numpy()) & (event == 1) & (fu_days.to_numpy() < ttr.to_numpy())
        if np.any(bad):
            print(f"[WARN] Dropping {bad.sum()} inconsistent patients where follow_up_days < time_to_relapse_days (event=1).")
            keep = keep & (~bad)

    byp2 = byp.loc[keep].copy()
    patients = byp2.index.astype(str).tolist()

    # recompute aligned arrays
    r = byp2[relapse_col].astype(str).str.strip().str.upper()
    r = r.map({
        "YES": 1, "Y": 1, "TRUE": 1, "T": 1, "1": 1,
        "NO": 0,  "N": 0, "FALSE": 0, "F": 0, "0": 0,
    })
    r = pd.to_numeric(r, errors="coerce").to_numpy()

    event = np.where(r == 1.0, 1, 0).astype(int)
    ttr = byp2[ttr_col].astype(float).to_numpy()
    fu_days = (byp2[fu_col].astype(float).to_numpy()) * float(days_per_month)
    time_days = np.where(event == 1, ttr, fu_days).astype(float)

    return patients, time_days, event


class PatientBagSurvivalDataset(Dataset):
    """
    Bags per patient + (time, event) computed from existing obs columns
    Does NOT write anything to adata.obs.
    """
    def __init__(
        self,
        adata,
        patient_col="patient_id",
        relapse_col="relapse_y_n",
        ttr_col="time_to_relapse_days",
        fu_col="follow_up_duration_months",
        days_per_month=DAYS_PER_MONTH,
        drop_inconsistent=True,
        pooling=None,
    ):
        self.adata = adata
        self.patient_col = patient_col
        self.pooling = pooling

        patients, time_days, event = build_patient_survival_table(
            adata.obs,
            patient_col=patient_col,
            relapse_col=relapse_col,
            ttr_col=ttr_col,
            fu_col=fu_col,
            days_per_month=days_per_month,
            drop_inconsistent=drop_inconsistent,
        )

        self.patient_list = patients
        self.time_days = time_days
        self.event = event

        # cache bags
        pid_arr = adata.obs[patient_col].astype(str).to_numpy()
        self.patient_bags = {}
        for pid in self.patient_list:
            idx = np.where(pid_arr == pid)[0]
            X = adata.X[idx]
            if not isinstance(X, np.ndarray):
                X = X.toarray()
            self.patient_bags[pid] = X

        # index mapping
        self._pid_to_idx = {p: i for i, p in enumerate(self.patient_list)}

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, idx):
        pid = self.patient_list[idx]
        bag = torch.tensor(self.patient_bags[pid], dtype=torch.float32)
        t = torch.tensor(float(self.time_days[idx]), dtype=torch.float32)
        e = torch.tensor(int(self.event[idx]), dtype=torch.float32)
        if self.pooling == "mean":
            bag = bag.mean(dim=0, keepdim=True)
        return bag, t, e, pid


def collate_survival(batch):
    bags, times, events, pids = zip(*batch)
    # bags is list[tensor] variable length
    times = torch.stack(times, dim=0)     # [B]
    events = torch.stack(events, dim=0)   # [B]
    return list(bags), times, events, list(pids)