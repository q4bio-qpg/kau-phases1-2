import torch
from torch import nn
import pennylane as qml
import torch.nn.functional as F
import numpy as np



class EditDistanceModel(nn.Module):
    def __init__(self, out_dim=1024):
        super(EditDistanceModel, self).__init__()

        self.out_dim = out_dim
        
        self.flatten = nn.Flatten()
        
        # self.conv1_real = nn.Conv1d(1, 2, kernel_size=2, stride=2, padding=0)
        # self.conv1_imag = nn.Conv1d(1, 2, kernel_size=2, stride=2, padding=0)
        self.conv1_real = nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=0)
        self.conv1_imag = nn.Conv1d(1, 1, kernel_size=2, stride=1, padding=0)

        self.conv2_real = nn.Conv1d(1, 4, kernel_size=8, stride=1, padding=0)
        self.conv2_imag = nn.Conv1d(1, 4, kernel_size=8, stride=1, padding=0)

        # self.conv2_real = nn.Conv1d(2, 4, kernel_size=2, stride=2, padding=0)
        # self.conv2_imag = nn.Conv1d(2, 4, kernel_size=2, stride=2, padding=0)
        
        # Calculate the flattened dimension after the conv layers
        # self.flatten_dim = 1552
        # self.flatten_dim = 1551 #k=4, step=1
        # self.flatten_dim = 783 #k=4, step=2
        # self.flatten_dim = 1535 #k=5, step=1
        # self.flatten_dim = 767 #k=4, step=2

        # self.flatten_dim = 63 #k=25, step=25
        # self.flatten_dim = 159 #k=10, step=10
        # self.flatten_dim = 255 #k=25, step=5
        self.flatten_dim = 79 #k=20, step=20
        # self.flatten_dim = 255 #k=25 , step=25, sub=3

        # self.flatten_dim = 64
        # self.flatten_dim = 1295 #step = 1, k = 20
        # self.flatten_dim = 271 #step = 5, k = 20
        # self.flatten_dim = 143 #step = 10
        # self.flatten_dim = 95 #step = 15
        # self.flatten_dim = 79 #step = 20
        # self.flatten_dim = 47 #k = 50, step = 20
        # self.flatten_dim = 31 #k = 50, step = 50
        # self.flatten_dim = 15 #k = 100, step = 100

        self.fc_real = nn.Linear(self.flatten_dim, self.out_dim)
        self.fc_imag = nn.Linear(self.flatten_dim, self.out_dim)

        self.fc1_real = nn.Linear(self.flatten_dim, self.flatten_dim * 2)
        self.fc1_imag = nn.Linear(self.flatten_dim, self.flatten_dim * 2)

        self.fc2_real = nn.Linear(self.flatten_dim * 2, self.out_dim)
        self.fc2_imag = nn.Linear(self.flatten_dim * 2, self.out_dim)

        self.fc3_real = nn.Linear(self.out_dim, self.out_dim)
        self.fc3_imag = nn.Linear(self.out_dim, self.out_dim)
        

    def forward(self, x):
        
        # Flatten the embeddings
        x = self.flatten(x)
        
        # Reshape to (batch_size, 1, embedding_dim * len_dim) for Conv1d
        x = x.unsqueeze(1)
        
        # x_real = F.relu(self.conv1_real(x))
        # x_imag = F.relu(self.conv1_imag(x))

        
        
        x_real = F.relu(self.conv1_real(x))
        x_imag = F.relu(self.conv1_imag(x))
        # x2_real = F.relu(self.conv2_real(x2_real))
        # x2_imag = F.relu(self.conv2_imag(x2_imag))
        
        # Flatten the output of the convolutional layers
        x_real = self.flatten(x_real)
        x_imag = self.flatten(x_imag)

        # x_real = F.max_pool1d(x_real, kernel_size=2, stride=2)
        # x_imag = F.max_pool1d(x_imag, kernel_size=2, stride=2)
        
        # x_real = F.relu(self.fc1_real(x))
        # x_imag = F.relu(self.fc1_real(x))

        # x_real = F.relu(self.fc2_real(x_real))
        # x_imag = F.relu(self.fc2_real(x_imag))

        # x_real = self.fc3_real(x_real)
        # x_imag = self.fc3_real(x_imag)

        x_real = self.fc_real(x_real)
        x_imag = self.fc_real(x_imag)

        x = torch.complex(x_real, x_imag)
        # x = x_real
        
        # Normalize the output vectors to have a norm of 1
        # x = x / torch.norm(x, dim=-1, keepdim=True)
        # x2 = x2 / torch.norm(x2, dim=-1, keepdim=True)
        
        return x
    

class QuantumLayer(nn.Module):
    def __init__(self, wires=10, circuit_layers=4, seed=None, device="cuda"):
        super(QuantumLayer, self).__init__()
        
        # Initialize quantum device on CUDA
        self.wires = wires
        self.device = device  # GPU or CPU device
        self.q_device = qml.device("default.qubit.torch", wires=self.wires, torch_device=device)
        self.layers = circuit_layers
        if seed is None:
            seed = np.random.randint(low=0, high=10e6)
            
        print("Initializing Circuit with random seed", seed)
        
        # Define the quantum circuit
        @qml.qnode(device=self.q_device, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            # print(inputs.shape)
            self.state_preparation(inputs, self.wires)

            for layer_weights in weights:
                self.layer(layer_weights, self.wires)

            state = qml.state()

            return state
        
        weight_shapes = {"weights": [self.layers, self.wires, 3]}
        self.circuit = qml.qnn.TorchLayer(circuit, weight_shapes=weight_shapes).to(self.device)
    
    def layer(self, layer_weights, n_qubits):
        for wire in range(n_qubits):
            qml.Rot(*layer_weights[wire], wires=wire)

        wires_list = [[i, i+1] for i in range(n_qubits - 1)]
        wires_list.append([n_qubits - 1, 0])  # Connect the last qubit back to the first

    def state_preparation(self, x, n_qubits):
        # print(x.shape)
        qml.StatePrep(x, wires=range(n_qubits), normalize=True)
    
    def forward(self, x):
        out = self.circuit(x)
        out = out[:, : (2**self.wires) // 2]
        return out
    

class L2Normalization(torch.nn.Module):
    def forward(self, x):
        # Normalize each vector (along the last dimension)
        return x / torch.norm(x, dim=-1, keepdim=True)
