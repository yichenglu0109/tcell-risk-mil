import torch
import torch.nn as nn
import torch.nn.functional as F


# 2. Attention-based MIL model
class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model as used in scMILD
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=128, dropout=0.25, sample_source_dim=None):
        """
        Initialize the MIL model
        
        Parameters:
        - input_dim: Dimension of the input features (output of autoencoder)
        - num_classes: Number of response classes
        - hidden_dim: Dimension of hidden layer
        - dropout: Dropout rate
        """
        super(AttentionMIL, self).__init__()

        self.use_sample_source = sample_source_dim is not None
        
        # Feature extractor network
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        if self.use_sample_source:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim + sample_source_dim, hidden_dim//2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim//2, num_classes)
            )
        else:
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ) #nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x, sample_source=None, return_attention=False):
        """
        Forward pass
        
        Parameters:
        - x: Input bag of instances [batch_size, num_instances, features]
        - return_attention: Whether to return attention weights
        
        Returns:
        - logits: Class logits [batch_size, num_classes]
        - attention_weights: Attention weights if return_attention=True
        """
        device = next(self.parameters()).device
        # x shape: [batch_size, num_instances, features]
        batch_size = len(x)
        
        # Process each bag
        all_logits = []
        all_attention_weights = []
        
        for i in range(batch_size):
            instances = x[i]  # [num_instances, features]
            
            # Extract features from each instance
            instance_features = self.feature_extractor(instances)  # [num_instances, hidden_dim]
            
            # Calculate attention scores
            attention_scores = self.attention(instance_features)  # [num_instances, 1]
            attention_weights = F.softmax(attention_scores, dim=0)  # [num_instances, 1]
            
            # Calculate weighted average of instance features
            weighted_features = torch.sum(
                instance_features * attention_weights, dim=0
            )  # [hidden_dim]
            
            

            #########################################################
            ########## Add sample source if provided #################
            #########################################################
            if self.use_sample_source and sample_source is not None:
                
            # get the sample source for this patient
                sample_source_i = sample_source.squeeze(0)
                # print(f"Patient: {patient}")
                # # Print shape and example values for debugging
                # print(f"Sample source shape: {sample_source.shape}")
                # print(f"Example sample source value: {sample_source[0]}")
                # print(f"Sample source shape: {weighted_features.shape}")
                
                combined_features = torch.cat([weighted_features, sample_source_i], dim=0)
                # print(f"combined_features shape: {combined_features.shape}")
                logits = self.classifier(combined_features)
            
            else:
                logits = self.classifier(weighted_features)
                
            all_logits.append(logits)
            all_attention_weights.append(attention_weights)
        
        # Stack results
        logits = torch.stack(all_logits)  # [batch_size, num_classes]
        attention_weights = all_attention_weights  # List of [num_instances, 1]
        
        if return_attention:
            return logits, attention_weights
        else:
            return logits