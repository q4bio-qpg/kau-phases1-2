import torch
from torch import nn
import pennylane as qml


class Classical2QuantumConnector(nn.Module):
    def __init__(self, input_size, n_qubits, n_quantum_layers, depth=1, relu=True):
        super(Classical2QuantumConnector, self).__init__()
        self.input_size = input_size
        self.cirquit_params_shape = qml.StronglyEntanglingLayers.shape(n_layers=n_quantum_layers, n_wires=n_qubits)
        self.falttened_params_shape = len(torch.zeros(self.cirquit_params_shape).flatten())
        self.depth = depth
        if relu:
          self.relu = nn.ReLU()
        self.mlp = self.gen_sequential_model()


    def gen_sequential_model(self):
      if self.depth == 1:
        return nn.Sequential(nn.Linear(self.input_size, self.falttened_params_shape))
      elif self.depth > 1:
        linear_layers = []
        if self.depth % 2 == 0:
          max_lin_layer = self.depth / 2
          increase = True
          curr_size = self.input_size
          for i in range(1, self.depth+1):
            if increase:
              layer = nn.Linear(curr_size, int(curr_size*2))
              curr_size = int(curr_size*2)
            else:
              if i == self.depth:
                layer = nn.Linear(curr_size, self.falttened_params_shape)
              else:
                layer = nn.Linear(curr_size, int(curr_size/2))
                curr_size = int(curr_size/2)
            if i == max_lin_layer:
              increase=False
            linear_layers.append(layer)
            if hasattr(self, 'relu') and i!=self.depth:
               linear_layers.append(self.relu)
        else:
          max_lin_layers = [self.depth // 2, (self.depth // 2) + 1]
          increase1 = True
          increase2 = True
          curr_size = self.input_size
          for i in range(1, self.depth+1):
            if increase1 and increase2:
              layer = nn.Linear(curr_size, int(curr_size*2))
              curr_size = int(curr_size*2)
            elif not increase1 and increase2:
               layer = nn.Linear(curr_size, curr_size)
            else:
              if i == self.depth:
                layer = nn.Linear(curr_size, self.falttened_params_shape)
              else:
                layer = nn.Linear(curr_size, int(curr_size/2))
                curr_size = int(curr_size/2)
            if i in max_lin_layers and increase1:
              increase1=False
            elif i in max_lin_layers and not increase1 and increase2:
              increase2=False 
            linear_layers.append(layer)
            if hasattr(self, 'relu') and i!=self.depth:
               linear_layers.append(self.relu)
        
        return nn.Sequential(*linear_layers)
      else:
         print('depth should be greater then 0')
    
    def forward(self, x):
      _flattened_params = self.mlp(x)
      flattened_params = torch.tanh(_flattened_params) * (torch.pi)

      reshaped_params = flattened_params.view(flattened_params.shape[0], *self.cirquit_params_shape)

      return reshaped_params