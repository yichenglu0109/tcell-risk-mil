import torch.nn as nn
import torch.optim as optim
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc




# Define the autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=48):
        """
        Autoencoder for SCENIC AUC matrix
        
        Parameters:
        - input_dim: Number of input features (TFs)
        - latent_dim: Size of the bottleneck layer
        """
        super(Autoencoder, self).__init__()
        
        # Encoder with 2 layers (input_dim -> 64 -> latent_dim)
        self.encoder = nn.Sequential(  
            
            nn.Linear(input_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2), #0.3
            
            nn.Linear(64, latent_dim)
        )
        
        # Decoder with 2 layers (latent_dim -> 64 -> input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            # nn.BatchNorm1d(64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.2), # 0.3
            
                   
            nn.Linear(64, input_dim),
            # nn.Sigmoid()  # Sigmoid for AUC values (0-1 range)
            nn.Tanh() # for [-1, 1] range
        )
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation back to original space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Full forward pass: encode then decode"""
        z = self.encode(x)
        return self.decode(z)
    
# Training function for the autoencoder
def train_autoencoder(train_loader, val_loader, input_dim, latent_dim=32, num_epochs=100, 
                      learning_rate=5e-4, weight_decay=1e-4, patience=10, save_path='models'):
    """
    Train the autoencoder model
    
    Parameters:
    - train_loader: DataLoader for training data
    - val_loader: DataLoader for validation data
    - input_dim: Input dimension (number of TFs)
    - latent_dim: Dimension of latent space
    - num_epochs: Maximum number of training epochs
    - learning_rate: Learning rate for optimizer
    - weight_decay: L2 regularization strength
    - patience: Early stopping patience
    - save_path: Directory to save model and plots
    
    Returns:
    - model: Trained autoencoder model
    - train_losses: List of training losses per epoch
    - val_losses: List of validation losses per epoch
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize model
    model = Autoencoder(input_dim, latent_dim)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    model.to(device)
    

    # Initialize WandB
    # wandb.init(project="car-t-IP-MIL", name="autoencoder-training")
    
    # Training setup
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for data, _ in train_loader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, data)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # wandb.log({"train_loss": train_loss})
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(device)
                outputs = model(data)
                loss = criterion(outputs, data)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        # wandb.log({"val_loss": val_loss})
        
        # Learning rate adjustment
        scheduler.step() #val_loss)
        
        # Print progress
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model
            torch.save(model.state_dict(), f'{save_path}/best_autoencoder.pth')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load best model
    model.load_state_dict(torch.load(f'{save_path}/best_autoencoder.pth'))
    
    # Plot training and validation loss
    # plt.figure(figsize=(10, 6))
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Autoencoder Training and Validation Loss')
    # plt.legend()
    # plt.savefig(f'{save_path}/autoencoder_training_loss.png')
    
    return model, train_losses, val_losses

# 5. Evaluation function
def evaluate_autoencoder(model, test_loader, adata, tf_names, save_path='results'):
    """
    Evaluate the trained autoencoder
    
    Parameters:
    - model: Trained autoencoder model
    - test_loader: DataLoader for test data
    - adata: Original AnnData object
    - tf_names: Names of transcription factors
    - save_path: Directory to save evaluation results
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Test set evaluation
    model.eval()
    test_loss = 0
    criterion = nn.MSELoss()
    
    all_inputs = []
    all_reconstructions = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
            
            # Collect inputs and reconstructions for later analysis
            all_inputs.append(data.cpu().numpy())
            all_reconstructions.append(outputs.cpu().numpy())
    
    test_loss /= len(test_loader)
    print(f'Test Loss: {test_loss:.6f}')
    # wandb.log({"test_loss": test_loss})
    
    # Combine batches
    all_inputs = np.vstack(all_inputs)
    all_reconstructions = np.vstack(all_reconstructions)
    
    # Calculate reconstruction error for each TF
    mse_per_tf = np.mean((all_inputs - all_reconstructions)**2, axis=0)
    
    # Plot reconstruction error per TF
    # plt.figure(figsize=(14, 6))
    # plt.bar(range(len(mse_per_tf)), mse_per_tf)
    # plt.xticks(range(len(mse_per_tf)), tf_names, rotation=90)
    # plt.xlabel('Transcription Factors')
    # plt.ylabel('MSE')
    # plt.title('Reconstruction Error per Transcription Factor')
    # plt.tight_layout()
    # plt.savefig(f'{save_path}/tf_reconstruction_error.png')
    
    # Generate latent space representation for all cells
    all_cells = adata.X
    if isinstance(all_cells, np.ndarray) == False:
        all_cells = all_cells.toarray()
    
    all_cells_tensor = torch.FloatTensor(all_cells).to(device)
    
    with torch.no_grad():
        latent_vectors = model.encode(all_cells_tensor).cpu().numpy()
    
    # Create a new AnnData object with latent representations
    adata_latent = sc.AnnData(latent_vectors)
    adata_latent.obs = adata.obs.copy()  # Copy cell annotations
    
    # UMAP visualization of latent space
    sc.pp.neighbors(adata_latent, use_rep='X')
    sc.tl.umap(adata_latent)
    
    # os.makedirs('figures/umapresults', exist_ok=True)
    # if 'patient_id' in adata_latent.obs.columns:
    #     sc.pl.umap(adata_latent, color=['patient_id'], save=f'{save_path}/latent_umap_by_patient.png')
    
    # if 'Response_3m' in adata_latent.obs.columns:
    #     sc.pl.umap(adata_latent, color=['Response_3m'], save=f'{save_path}/latent_umap_by_response.png')

    
    if 'patient_id' in adata_latent.obs.columns:
        fig = sc.pl.umap(adata_latent, color=['patient_id'], show=False, return_fig=True)
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f'{save_path}/latent_umap_by_patient.png', bbox_inches='tight')
        plt.close(fig)
    
    if 'Response_3m' in adata_latent.obs.columns:
        fig = sc.pl.umap(adata_latent, color=['Response_3m'], show=False, return_fig=True)
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f'{save_path}/latent_umap_by_response.png', bbox_inches='tight')
        plt.close(fig)
    
    
    # Save latent representation for MIL
    adata_latent.write(f'{save_path}/latent_representation.h5ad')
    
    return adata_latent, test_loss