# models/mlp.py

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super(MLP, self).__init__()

        self.hidden_layers = nn.ModuleList()
        in_features = input_size

        # Define hidden layers
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(in_features, hidden_size))
            in_features = hidden_size

        # Output layer
        self.output = nn.Linear(in_features, output_size)

        # Activation function (ReLU)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input from (batch, 1, 28, 28) → (batch, 784)
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        x = self.output(x)
        return x
