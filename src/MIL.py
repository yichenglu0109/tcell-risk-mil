import torch
import torch.nn as nn
import torch.nn.functional as F


# 2. Attention-based MIL model
class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model as used in scMILD
    """
    def __init__(self, input_dim, num_classes=2, hidden_dim=128, dropout=0.25, sample_source_dim=None, aggregator="attention", topk=None, tau=None):
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
        self.tau = tau  # None => no temperature scaling
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
        x: list of bags
        - attention mode: each bag is [n_cells, input_dim]
        - pseudobulk mode: each bag is [input_dim]  (already pooled)
        sample_source: optional, shape [B, sample_source_dim]
        """
        batch_size = len(x)

        all_logits = []
        all_attention_weights = []
        all_risks = []

        device = next(self.parameters()).device

        for i in range(batch_size):
            instances = x[i].to(device)

            # =========================
            # Mode switch by shape
            # =========================
            if instances.dim() == 1:
                # -------- pseudobulk --------
                # instances: [input_dim]
                bag_feat = self.feature_extractor(instances)  # [hidden_dim]
                attention_weights = None

            else:
                # -------- attention MIL --------
                # instances: [n_cells, input_dim]
                instance_features = self.feature_extractor(instances)  # [n_cells, hidden_dim]

                # attention scores: [n_cells]
                scores = self.attention(instance_features).view(-1)  # [N]

                # --- optional TOP-K ---
                if self.topk is not None and self.topk > 0 and self.topk < scores.numel():
                    k = min(self.topk, scores.numel())
                    top_idx = torch.topk(scores, k=k, largest=True).indices
                    instance_features = instance_features[top_idx]  # [k, hidden_dim]
                    scores = scores[top_idx]                        # [k]

                # --- optional temperature ---
                if self.tau is None:
                    logits = scores
                else:
                    logits = scores / float(self.tau)

                attention_weights = F.softmax(logits, dim=0).unsqueeze(-1)  # [k,1] or [N,1]

                # pooled bag feature: [hidden_dim]
                bag_feat = torch.sum(instance_features * attention_weights, dim=0)

            # =========================
            # Add sample_source covariate
            # =========================
            if self.use_sample_source:
                if sample_source is None:
                    raise ValueError("sample_source_dim was set but sample_source is None in forward()")
                sample_source_i = sample_source[i].to(device).view(-1)  # [sample_source_dim]
                bag_feat = torch.cat([bag_feat, sample_source_i], dim=0)  # [bag_dim]

            # =========================
            # Heads
            # =========================
            logits = self.cls_head(bag_feat)               # [num_classes]
            risk = self.surv_head(bag_feat).squeeze(-1)    # scalar

            all_logits.append(logits)
            all_risks.append(risk)
            all_attention_weights.append(attention_weights)

        logits = torch.stack(all_logits, dim=0)  # [B, num_classes]
        risks  = torch.stack(all_risks, dim=0)   # [B]

        out = {"logits": logits, "risk": risks}
        if return_attention:
            out["attn"] = all_attention_weights
        return out

