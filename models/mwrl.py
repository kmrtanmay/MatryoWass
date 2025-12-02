"""
Matryoshka-Wasserstein Representation Learning (MWRL)

This module implements MWRL with the following improvements:
1. Fully differentiable Wasserstein regularization using GeomLoss
2. Normalized embeddings for better OT conditioning
3. Stabilized training with proper scaling
4. Updated ResNet API (non-deprecated)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Optional, Tuple, Dict
import numpy as np

try:
    from geomloss import SamplesLoss
    GEOMLOSS_AVAILABLE = True
except ImportError:
    GEOMLOSS_AVAILABLE = False
    print("Warning: geomloss not found. Install with: pip install geomloss")
    print("Falling back to POT-based implementation (less stable)")


class WassersteinRegularizer(nn.Module):
    """
    Differentiable Wasserstein regularizer using GeomLoss.
    Handles custom cost function with padding and reweighting.
    
    The cost function is:
        C(x, y) = ||x - y[:d_k]||^2 + beta * ||y[d_k:]||^2
    
    This is implemented by:
    1. Padding x to match y's dimension
    2. Reweighting the residual dimensions by sqrt(beta)
    3. Using standard L2 Sinkhorn divergence
    """
    
    def __init__(self, beta: float = 1.0, epsilon: float = 0.5):
        """
        Args:
            beta: Weight for remainder penalty (default: 1.0)
            epsilon: Entropic regularization (blur) parameter (default: 0.5)
                    Larger values = more stable but less accurate
        """
        super().__init__()
        self.beta = beta
        
        if GEOMLOSS_AVAILABLE:
            self.sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=epsilon)
            self.use_geomloss = True
        else:
            # Fallback to POT
            self.use_geomloss = False
            self.epsilon = epsilon
            try:
                import ot
                self.ot = ot
            except ImportError:
                raise ImportError("Either geomloss or POT must be installed. "
                                "Install geomloss with: pip install geomloss "
                                "or POT with: pip install pot")
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor, d_k: int) -> torch.Tensor:
        """
        Compute Wasserstein distance with custom cost.
        
        Args:
            X: (B, d_k) - embeddings at dimension k
            Y: (B, d_{k+1}) - embeddings at dimension k+1
            d_k: dimension of X
            
        Returns:
            Scalar Wasserstein distance (differentiable)
        """
        if self.use_geomloss:
            return self._geomloss_forward(X, Y, d_k)
        else:
            return self._pot_forward(X, Y, d_k)
    
    def _geomloss_forward(self, X: torch.Tensor, Y: torch.Tensor, d_k: int) -> torch.Tensor:
        """GeomLoss implementation (fully differentiable)."""
        B, dk = X.shape
        dkk = Y.shape[1]
        
        # Pad X to match Y's dimension
        pad = torch.zeros(B, dkk - dk, device=X.device, dtype=X.dtype)
        Xp = torch.cat([X, pad], dim=1)  # (B, d_{k+1})
        
        # Reweight residual dimensions by sqrt(beta)
        W = X.new_ones(dkk)
        if dkk > dk:
            W[dk:] = self.beta ** 0.5
        
        Xw = Xp * W  # (B, d_{k+1})
        Yw = Y * W   # (B, d_{k+1})
        
        # Compute Sinkhorn divergence (fully differentiable)
        return self.sinkhorn(Xw, Yw)
    
    def _pot_forward(self, X: torch.Tensor, Y: torch.Tensor, d_k: int) -> torch.Tensor:
        """POT fallback implementation (keep everything in torch)."""
        # Compute cost matrix
        Y_shared = Y[:, :d_k]
        Y_res = Y[:, d_k:]
        
        C_shared = torch.cdist(X, Y_shared, p=2) ** 2  # (B, B)
        r = (Y_res ** 2).sum(dim=1)  # (B,)
        C = C_shared + self.beta * r.unsqueeze(0)  # (B, B)
        
        # Scale to improve conditioning
        C = C / (C.median().detach() + 1e-8)
        
        # Uniform distributions
        Bx, By = C.shape
        a = torch.full((Bx,), 1.0/Bx, device=C.device, dtype=C.dtype)
        b = torch.full((By,), 1.0/By, device=C.device, dtype=C.dtype)
        
        # Stabilized Sinkhorn
        T = self.ot.bregman.sinkhorn_stabilized(
            a, b, C, self.epsilon,
            numItermax=1000,
            stopThr=1e-6
        )
        
        return (T * C).sum()


class MatryoshkaClassifiers(nn.Module):
    """
    Multiple linear classifiers for different embedding dimensions.
    Each classifier operates on a truncated version of the full embedding.
    """
    
    def __init__(self, nesting_list: List[int], num_classes: int):
        """
        Args:
            nesting_list: List of embedding dimensions (e.g., [8, 16, 32, ..., 2048])
            num_classes: Number of output classes
        """
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        
        # Create a classifier for each dimension
        self.classifiers = nn.ModuleList([
            nn.Linear(dim, num_classes) for dim in nesting_list
        ])
    
    def forward(self, truncated_embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Args:
            truncated_embeddings: List of tensors, each of shape (B, d_i)
            
        Returns:
            List of logits, each of shape (B, num_classes)
        """
        return [
            classifier(emb) 
            for classifier, emb in zip(self.classifiers, truncated_embeddings)
        ]


class MWRLLoss(nn.Module):
    """
    Combined loss for Matryoshka-Wasserstein Representation Learning.
    
    Loss = sum_i w_i * CE_i + lambda * sum_i W(emb_i, emb_{i+1})
    
    where:
    - CE_i is cross-entropy loss at dimension i
    - W is the Wasserstein distance with custom cost
    - w_i are per-dimension weights
    - lambda is the Wasserstein regularization weight
    """
    
    def __init__(
        self,
        nesting_list: List[int],
        num_classes: int,
        ce_weights: Optional[List[float]] = None,
        wasserstein_weight: float = 0.1,
        beta: float = 1.0,
        epsilon: float = 0.5,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            nesting_list: List of embedding dimensions
            num_classes: Number of classes
            ce_weights: Optional weights for each CE loss (default: uniform)
            wasserstein_weight: Weight for Wasserstein regularization (lambda)
            beta: Beta parameter for Wasserstein cost (remainder penalty)
            epsilon: Entropic regularization for Sinkhorn
            label_smoothing: Label smoothing factor for CE loss
        """
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.wasserstein_weight = wasserstein_weight
        self.label_smoothing = label_smoothing
        
        # CE weights
        if ce_weights is None:
            ce_weights = [1.0] * len(nesting_list)
        self.register_buffer('ce_weights', torch.tensor(ce_weights, dtype=torch.float32))
        
        # Classifiers
        self.classifiers = MatryoshkaClassifiers(nesting_list, num_classes)
        
        # Wasserstein regularizer
        self.wasserstein_regularizer = WassersteinRegularizer(beta=beta, epsilon=epsilon)
        
        # CE loss with label smoothing
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    def get_truncated_embeddings(self, embeddings: torch.Tensor) -> List[torch.Tensor]:
        """
        Truncate embeddings to each nesting dimension.
        
        Args:
            embeddings: (B, full_dim) full-dimensional embeddings
            
        Returns:
            List of truncated embeddings [(B, d_1), (B, d_2), ..., (B, d_n)]
        """
        return [embeddings[:, :dim] for dim in self.nesting_list]
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined MWRL loss.
        
        Args:
            embeddings: (B, full_dim) full-dimensional embeddings
            targets: (B,) class labels
            
        Returns:
            total_loss: Scalar loss tensor
            loss_dict: Dictionary with individual loss components
        """
        # Normalize embeddings for better OT conditioning
        embeddings = F.normalize(embeddings, dim=1)
        
        # Get truncated embeddings
        truncated_embeddings = self.get_truncated_embeddings(embeddings)
        
        # Get logits from classifiers
        logits_list = self.classifiers(truncated_embeddings)
        
        # Compute CE losses
        ce_losses = []
        for i, logits in enumerate(logits_list):
            ce_loss = self.ce_loss(logits, targets)
            ce_losses.append(ce_loss)
        
        # Weighted sum of CE losses
        ce_losses_tensor = torch.stack(ce_losses)
        weighted_ce_loss = (self.ce_weights * ce_losses_tensor).sum()
        
        # Compute Wasserstein regularization
        w_distances = []
        wasserstein_loss = 0.0
        
        for i in range(len(self.nesting_list) - 1):
            emb_k = truncated_embeddings[i]
            emb_k_plus_1 = truncated_embeddings[i + 1]
            d_k = self.nesting_list[i]
            
            # Compute Wasserstein distance
            w_dist = self.wasserstein_regularizer(emb_k, emb_k_plus_1, d_k)
            w_distances.append(w_dist.item())
            wasserstein_loss += w_dist
        
        # Total loss
        total_loss = weighted_ce_loss + self.wasserstein_weight * wasserstein_loss
        
        # Prepare loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': weighted_ce_loss.item(),
            'wasserstein_loss': wasserstein_loss.item() if isinstance(wasserstein_loss, torch.Tensor) else wasserstein_loss,
            'ce_losses': [ce.item() for ce in ce_losses],
            'w_distances': w_distances
        }
        
        return total_loss, loss_dict


class MWRLModel(nn.Module):
    """
    Complete MWRL model: backbone + embedding layer + loss module.
    """
    
    def __init__(
        self,
        nesting_list: List[int],
        num_classes: int = 100,
        pretrained: bool = True,
        ce_weights: Optional[List[float]] = None,
        wasserstein_weight: float = 0.1,
        beta: float = 1.0,
        epsilon: float = 0.5,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            nesting_list: List of embedding dimensions
            num_classes: Number of output classes
            pretrained: Use pretrained ResNet50 weights
            ce_weights: Optional weights for each CE loss
            wasserstein_weight: Weight for Wasserstein regularization
            beta: Beta parameter for Wasserstein cost
            epsilon: Entropic regularization for Sinkhorn
            label_smoothing: Label smoothing for CE loss
        """
        super().__init__()
        self.nesting_list = nesting_list
        self.num_classes = num_classes
        self.full_dim = max(nesting_list)
        
        # Backbone: ResNet50 (updated API, no deprecated warnings)
        if pretrained:
            resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        else:
            resnet = models.resnet50(weights=None)
        
        # Extract features (remove fc layer)
        # Note: resnet.children()[:-1] already includes avgpool, so no need for extra pooling
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Get ResNet output dimension (2048 for ResNet50)
        self.resnet_dim = 2048
        
        # Embedding layer: project to full dimension
        self.embedding_layer = nn.Linear(self.resnet_dim, self.full_dim)
        
        # Loss module (includes classifiers and Wasserstein regularizer)
        self.loss_module = MWRLLoss(
            nesting_list=nesting_list,
            num_classes=num_classes,
            ce_weights=ce_weights,
            wasserstein_weight=wasserstein_weight,
            beta=beta,
            epsilon=epsilon,
            label_smoothing=label_smoothing
        )
    
    def embedding_extractor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract embeddings from input images.
        
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            embeddings: (B, full_dim) embeddings
        """
        # Extract features
        features = self.features(x)  # (B, 2048, 1, 1)
        features = torch.flatten(features, 1)  # (B, 2048)
        
        # Project to embedding space
        embeddings = self.embedding_layer(features)  # (B, full_dim)
        
        return embeddings
    
    def get_truncated_embeddings(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get truncated embeddings at all nesting dimensions.
        
        Args:
            x: (B, 3, H, W) input images
            
        Returns:
            List of truncated embeddings
        """
        embeddings = self.embedding_extractor(x)
        return self.loss_module.get_truncated_embeddings(embeddings)
    
    def classify_at_dimension(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """
        Classify input at a specific embedding dimension.
        
        Args:
            x: (B, 3, H, W) input images
            dim: Embedding dimension to use (must be in nesting_list)
            
        Returns:
            logits: (B, num_classes)
        """
        if dim not in self.nesting_list:
            raise ValueError(f"Dimension {dim} not in nesting_list: {self.nesting_list}")
        
        embeddings = self.embedding_extractor(x)
        truncated_embeddings = self.loss_module.get_truncated_embeddings(embeddings)
        logits_list = self.loss_module.classifiers(truncated_embeddings)
        
        # Find index of requested dimension
        idx = self.nesting_list.index(dim)
        return logits_list[idx]
    
    def forward(
        self, 
        x: torch.Tensor, 
        targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Forward pass.
        
        Args:
            x: (B, 3, H, W) input images
            targets: (B,) class labels (required for training)
            
        Returns:
            If targets provided (training):
                loss: Scalar loss tensor
                loss_dict: Dictionary with loss components
            If targets not provided (inference):
                logits_list: List of logits at each dimension
        """
        embeddings = self.embedding_extractor(x)
        
        if targets is not None:
            # Training mode: compute loss
            return self.loss_module(embeddings, targets)
        else:
            # Inference mode: return logits
            truncated_embeddings = self.loss_module.get_truncated_embeddings(embeddings)
            return self.loss_module.classifiers(truncated_embeddings)


def create_mwrl_model(
    nesting_list: List[int],
    num_classes: int = 100,
    pretrained: bool = True,
    ce_weights: Optional[List[float]] = None,
    wasserstein_weight: float = 0.1,
    beta: float = 1.0,
    epsilon: float = 0.5,
    label_smoothing: float = 0.1
) -> MWRLModel:
    """
    Factory function to create MWRL model.
    
    Args:
        nesting_list: List of embedding dimensions (e.g., [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        num_classes: Number of output classes
        pretrained: Use pretrained ResNet50 weights
        ce_weights: Optional weights for each CE loss
        wasserstein_weight: Weight for Wasserstein regularization (lambda)
        beta: Beta parameter for Wasserstein cost (remainder penalty)
        epsilon: Entropic regularization for Sinkhorn (larger = more stable)
        label_smoothing: Label smoothing for CE loss
        
    Returns:
        MWRLModel instance
    """
    return MWRLModel(
        nesting_list=nesting_list,
        num_classes=num_classes,
        pretrained=pretrained,
        ce_weights=ce_weights,
        wasserstein_weight=wasserstein_weight,
        beta=beta,
        epsilon=epsilon,
        label_smoothing=label_smoothing
    )


# Example usage
if __name__ == "__main__":
    # Create model
    nesting_list = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    model = create_mwrl_model(
        nesting_list=nesting_list,
        num_classes=100,
        pretrained=True,
        wasserstein_weight=0.1,
        beta=1.0,
        epsilon=0.5
    )
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    targets = torch.randint(0, 100, (batch_size,))
    
    # Training mode
    loss, loss_dict = model(x, targets)
    print(f"Total loss: {loss.item():.4f}")
    print(f"CE loss: {loss_dict['ce_loss']:.4f}")
    print(f"Wasserstein loss: {loss_dict['wasserstein_loss']:.4f}")
    
    # Inference mode
    model.eval()
    with torch.no_grad():
        logits_list = model(x)
        print(f"Number of classifiers: {len(logits_list)}")
        print(f"Logits shapes: {[logits.shape for logits in logits_list]}")