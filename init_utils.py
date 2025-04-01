# init_utils.py

import torch
import torch.nn as nn

def apply_weight_initialization(model: nn.Module, method: str = "he"):
    """
    Applies the specified weight initialization method to all linear layers in the model.
    
    Args:
        model (nn.Module): The PyTorch model.
        method (str): Either 'he' or 'orthogonal'.
    """

    for layer in model.modules():
        # We're only interested in initializing nn.Linear layers (fully connected layers)
        if isinstance(layer, nn.Linear):
            if method == "he":
                # He Initialization (also called Kaiming Normal)
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            elif method == "orthogonal":
                # Orthogonal Initialization
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

            else:
                raise ValueError(f"Unsupported initialization method: {method}")
