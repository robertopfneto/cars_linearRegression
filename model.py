#%% Classe do Modelo
import torch.nn as nn

class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size): 
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)