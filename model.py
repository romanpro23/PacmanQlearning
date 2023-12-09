from functools import reduce

import torch.nn
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self, input_size, hidden, output, batch_norm=False, dropout = 0.0):
        super(LinearModel, self).__init__()
        self.layers = nn.ModuleList()

        if hidden is not None:
            if isinstance(hidden, int):
                self.layers.append(nn.Linear(input_size, hidden))
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(hidden))
                self.layers.append(nn.ReLU())
                if dropout > 0.0:
                    self.layers.append(nn.Dropout(dropout))
                self.layers.append(nn.Linear(hidden, output))
            elif isinstance(hidden, list):
                self.layers.append(nn.Linear(input_size, hidden[0]))
                input_size = hidden[0]
                if batch_norm:
                    self.layers.append(nn.BatchNorm1d(input_size))
                self.layers.append(nn.ReLU())
                if dropout > 0.0:
                    self.layers.append(nn.Dropout(dropout))

                for count_neurons in hidden[1:]:
                    self.layers.append(nn.Linear(input_size, count_neurons))
                    input_size = count_neurons
                    if batch_norm:
                        self.layers.append(nn.BatchNorm1d(input_size))
                    self.layers.append(nn.ReLU())
                    if dropout > 0.0:
                        self.layers.append(nn.Dropout(dropout))

                self.layers.append(nn.Linear(input_size, output))
        else:
            self.layers.append(nn.Linear(input_size, output))

    def forward(self, x):
        if x.dtype == torch.int64:
            x = x.to(dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x