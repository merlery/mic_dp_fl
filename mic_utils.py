"""
MIC (Maximum Information Coefficient) utilities for data transformation
Uses mutual information from scikit-learn as an alternative to minepy
"""
import numpy as np
import torch
from typing import Tuple, Optional
from collections import defaultdict

# Try to import minepy, but use scikit-learn as fallback
try:
    from minepy import MINE
    MINE_AVAILABLE = True
except ImportError:
    MINE_AVAILABLE = False
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        SKLEARN_AVAILABLE = True
    except ImportError:
        SKLEARN_AVAILABLE = False


def compute_mic_matrix(X: np.ndarray, y: np.ndarray, alpha: float = 0.6, c: float = 15) -> np.ndarray:
    """
    Compute feature importance scores between each feature and the target label.
    Uses MIC if minepy is available, otherwise uses mutual information from scikit-learn.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        alpha: MIC parameter (default 0.6, ignored if using sklearn)
        c: MIC parameter (default 15, ignored if using sklearn)
    
    Returns:
        scores: Array of importance scores for each feature (n_features,)
    """
    n_features = X.shape[1]
    scores = np.zeros(n_features)
    
    if MINE_AVAILABLE:
        # Use minepy if available
        mine = MINE(alpha=alpha, c=c)
        for i in range(n_features):
            try:
                mine.compute_score(X[:, i], y)
                scores[i] = mine.mic()
            except:
                scores[i] = 0.0
    elif SKLEARN_AVAILABLE:
        # Use scikit-learn mutual information as alternative
        try:
            # Determine if classification or regression
            if len(np.unique(y)) < 20:  # Likely classification
                scores = mutual_info_classif(X, y, random_state=42)
            else:  # Likely regression
                scores = mutual_info_regression(X, y, random_state=42)
        except:
            # Fallback to correlation if sklearn fails
            for i in range(n_features):
                try:
                    corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                    scores[i] = corr if not np.isnan(corr) else 0.0
                except:
                    scores[i] = 0.0
    else:
        # Fallback to simple correlation
        for i in range(n_features):
            try:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                scores[i] = corr if not np.isnan(corr) else 0.0
            except:
                scores[i] = 0.0
    
    return scores


def compute_mic_weights(X: np.ndarray, y: np.ndarray, alpha: float = 0.6, c: float = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute transformation weights based on feature importance scores.
    Uses MIC if available, otherwise uses mutual information or correlation.
    Features with higher importance (stronger relationship with label) get higher weights.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        alpha: MIC parameter (ignored if not using minepy)
        c: MIC parameter (ignored if not using minepy)
    
    Returns:
        gamma: Scale weights based on feature importance (n_features,)
        beta: Shift weights (initialized to zero)
    """
    importance_scores = compute_mic_matrix(X, y, alpha, c)
    
    # Normalize scores to [0, 1] range
    if importance_scores.max() > 0:
        scores_normalized = importance_scores / importance_scores.max()
    else:
        scores_normalized = np.ones_like(importance_scores)
    
    # Use importance scores to determine transformation weights
    # Higher importance -> higher weight (more important feature)
    # Add small epsilon to avoid zero weights
    epsilon = 1e-6
    gamma = scores_normalized + epsilon
    
    # Beta is initialized as zero (can be learned)
    beta = np.zeros_like(gamma)
    
    return gamma, beta


def compute_mic_for_batch(data_batch: torch.Tensor, labels_batch: torch.Tensor, 
                          alpha: float = 0.6, c: float = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute MIC-based weights for a batch of data.
    This is used for per-client personalized transformation.
    
    Args:
        data_batch: Batch of data (batch_size, ...)
        labels_batch: Batch of labels (batch_size,)
        alpha: MIC parameter
        c: MIC parameter
    
    Returns:
        gamma: Scale weights (torch.Tensor)
        beta: Shift weights (torch.Tensor)
    """
    # Flatten data for MIC computation
    if len(data_batch.shape) > 2:
        # For image data, flatten spatial dimensions
        data_flat = data_batch.view(data_batch.size(0), -1).cpu().numpy()
    else:
        data_flat = data_batch.cpu().numpy()
    
    labels_np = labels_batch.cpu().numpy()
    
    # Compute MIC-based weights
    gamma_np, beta_np = compute_mic_weights(data_flat, labels_np, alpha, c)
    
    # Reshape to match original data shape
    if len(data_batch.shape) > 2:
        # For image data, reshape gamma and beta
        gamma = torch.from_numpy(gamma_np).reshape(data_batch.shape[1:]).float()
        beta = torch.from_numpy(beta_np).reshape(data_batch.shape[1:]).float()
    else:
        gamma = torch.from_numpy(gamma_np).float()
        beta = torch.from_numpy(beta_np).float()
    
    return gamma, beta

