import torch
import torch.nn as nn
import torch.nn.functional as F


# 2. Attention-based MIL model
class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model as used in scMILD
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=128, dropout=0.25, sample_source_dim=None, topk=None):
        """
        Initialize the MIL model
        
        Parameters:
        - input_dim: Dimension of the input features (output of autoencoder)
        - num_classes: Number of response classes
        - hidden_dim: Dimension of hidden layer
        - dropout: Dropout rate
        """
        super(AttentionMIL, self).__init__()

        self.topk = topk
        self.use_sample_source = sample_source_dim is not None
        self.bag_dim = hidden_dim + (sample_source_dim if self.use_sample_source else 0)
        
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
        
        # --- Task-specific heads ---
        # Classification head (binary or multi-class)
        self.cls_head = nn.Sequential(
            nn.Linear(self.bag_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # Survival head (Cox): single risk score per bag
        self.surv_head = nn.Sequential(
            nn.Linear(self.bag_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
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
        # device = next(self.parameters()).device
        # x shape: [batch_size, num_instances, features]
        batch_size = len(x)
        
        # Process each bag
        all_logits = []
        all_attention_weights = []
        all_risks = []

        for i in range(batch_size):
            instances = x[i]  # [num_instances, features]
            instances = instances.to(next(self.parameters()).device)
            
            # Extract features from each instance
            instance_features = self.feature_extractor(instances)  # [num_instances, hidden_dim]
            
            # Calculate attention scores
            attention_scores = self.attention(instance_features)  # [num_instances, 1]
            # --- TOP-K (新增) ---
            scores = attention_scores.view(-1)  # [N]
            if self.topk is not None and self.topk > 0 and self.topk < scores.numel():
                k = min(self.topk, scores.numel())
                top_idx = torch.topk(scores, k=k, largest=True).indices
                instance_features = instance_features[top_idx]  # [k, hidden_dim]
                scores = scores[top_idx]                        # [k]
            # --------------------
            tau = 0.3
            attention_weights = F.softmax(attention_scores / tau, dim=0)# [num_instances, 1]
            
            # Calculate weighted average of instance features
            weighted_features = torch.sum(
                instance_features * attention_weights, dim=0
            )  # [hidden_dim]

            # bag-level feature
            bag_feat = weighted_features  # [hidden_dim]

            # add sample source if used
            if self.use_sample_source:
                if sample_source is None:
                    raise ValueError("sample_source_dim was set but sample_source is None in forward()")
                sample_source_i = sample_source[i]  # [sample_source_dim]
                sample_source_i = sample_source[i].view(-1) # flatten
                bag_feat = torch.cat([bag_feat, sample_source_i], dim=0)  # [bag_dim]

            # heads
            logits = self.cls_head(bag_feat)            # [num_classes]
            risk = self.surv_head(bag_feat).squeeze(-1) # scalar
                
            all_logits.append(logits)
            all_risks.append(risk)
            all_attention_weights.append(attention_weights)
        
        logits = torch.stack(all_logits, dim=0)  # [B, num_classes]
        risks  = torch.stack(all_risks, dim=0)   # [B]

        out = {"logits": logits, "risk": risks}
        if return_attention:
            out["attn"] = all_attention_weights
        return out
