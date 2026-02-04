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

        self.aggregator = aggregator
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

    # Pooling function
    def _pool_instances(self, instances, return_attention=False):
        """
        instances: Tensor [N, input_dim]
        returns:
        bag_feat: Tensor [hidden_dim]
        attention_weights: Tensor [N,1] or None
        """
        # feature extract per cell
        instance_features = self.feature_extractor(instances)  # [N, hidden_dim]
        attention_weights = None

        if self.aggregator in (None, "attention"):
            scores = self.attention(instance_features).view(-1)  # [N]

            # TOP-K on scores (as you already do)
            if self.topk is not None and self.topk > 0 and self.topk < scores.numel():
                k = min(int(self.topk), scores.numel())
                top_idx = torch.topk(scores, k=k, largest=True).indices
                instance_features = instance_features[top_idx]  # [k, hidden_dim]
                scores = scores[top_idx]                        # [k]

            # temperature
            if self.tau is not None and float(self.tau) > 0:
                scores = scores / float(self.tau)

            attention_weights = F.softmax(scores, dim=0).unsqueeze(-1)  # [k,1] or [N,1]
            bag_feat = torch.sum(instance_features * attention_weights, dim=0)  # [hidden_dim]

        elif self.aggregator == "mean":
            # mean pooling after feature extractor
            bag_feat = instance_features.mean(dim=0)  # [hidden_dim]

        elif self.aggregator == "q90":
            # feature-wise 90th percentile after feature extractor
            # (robust "high-signal" pooling baseline)
            bag_feat = torch.quantile(instance_features, q=0.9, dim=0)  # [hidden_dim]

        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")

        if return_attention:
            return bag_feat, attention_weights
        return bag_feat, None
    
    def forward(self, x, sample_source=None, return_attention=False):
        """
        x: list of bags, each bag is Tensor [n_cells, input_dim]
        sample_source: optional, shape [B, sample_source_dim]
        """
        batch_size = len(x)

        all_logits = []
        all_attention_weights = []
        all_risks = []

        device = next(self.parameters()).device

        for i in range(batch_size):
            instances = x[i].to(device)

            # ---- enforce 2D bag ----
            # if user accidentally passes [input_dim], convert to [1, input_dim]
            if instances.dim() == 1:
                instances = instances.unsqueeze(0)

            # ---- pool according to self.aggregator ----
            bag_feat, attention_weights = self._pool_instances(
                instances, return_attention=return_attention
            )

            # ---- add sample_source covariate ----
            if self.use_sample_source:
                if sample_source is None:
                    raise ValueError("sample_source_dim was set but sample_source is None in forward()")
                sample_source_i = sample_source[i].to(device).view(-1)  # [sample_source_dim]
                bag_feat = torch.cat([bag_feat, sample_source_i], dim=0)  # [bag_dim]

            # ---- heads ----
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


