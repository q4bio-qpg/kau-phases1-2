import torch
from model2 import *
from utils import *
from train import *
from torch.utils.data import DataLoader, Dataset
import pickle
import os

kmer=25
step=25
sub = 2
n_pairs = 10000

output_dir = 'data'
train_filename = f'{output_dir}/train_data_k{kmer}_s{sub}_l100_step{step}_np{n_pairs}.pkl'
val_filename = f'{output_dir}/val_data_k{kmer}_s{sub}_l100_step{step}_np13650.pkl'

with open(train_filename, 'rb') as f:
    train_data_dict = pickle.load(f)

with open(val_filename, 'rb') as f:
    val_data_dict = pickle.load(f)

# model = torch.nn.Sequential(
#         # EditDistanceModel(out_dim=1024),
#         EditDistanceModel(out_dim=2048),
#         QuantumLayer(wires=11),
#         L2Normalization()
#     )

model = torch.nn.Sequential(
        EditDistanceModel(out_dim=33*2),
        QuantumLayer(wires=11, circuit_layers=2),
        L2Normalization()
    )

first_tensors_train = train_data_dict['first_tensors_train']

second_tensors_train = train_data_dict['second_tensors_train']

targets_list_train = []
train_distances = train_data_dict['train_distances']
for distance in train_distances:
    targets_list_train.append(distance / 60)

first_tensors_train, second_tensors_train, targets_list_train = get_data_dict(train_data_dict)
first_tensors_val, second_tensors_val, targets_list_val = get_data_dict(val_data_dict, type='val')

X_train = []
for i in range(len(first_tensors_train)):
    X_train.append((first_tensors_train[i], second_tensors_train[i]))
Y_train = targets_list_train

X_val = []
for i in range(len(first_tensors_val)):
    X_val.append((first_tensors_val[i], second_tensors_val[i]))
Y_val = targets_list_val

class EditDistanceDataset(Dataset):
    def __init__(self, input_pairs, targets):
        self.input_pairs = input_pairs
        self.targets = targets

    def __len__(self):
        return len(self.input_pairs)

    def __getitem__(self, idx):
        return self.input_pairs[idx], self.targets[idx]
    
train_dataset = EditDistanceDataset(X_train, Y_train)
train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)

val_dataset = EditDistanceDataset(X_val, Y_val)
val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False)

epochs = 100
model_type = 'r'

train_losses, train_accuracies, val_losses, val_accuracies, train_outputs, train_targets, val_outputs, val_targets = train(model, train_dataloader, val_dataloader, epochs = epochs, model_type=model_type)

outputs_dict = {
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'val_losses': val_losses,
    'val_accuracies': val_accuracies,
    'train_outputs': train_outputs,
    'train_targets': train_targets,
    'val_outputs': val_outputs,
    'val_targets': val_targets,
}


# Define the filename based on model type and epochs
filename = f"{output_dir}/training_results2_{model_type}_epochs{epochs}_k{kmer}_step{step}_s{sub}_np{n_pairs}_exp.pkl"

# Save the dictionary to a pickle file
with open(filename, 'wb') as f:
    pickle.dump(outputs_dict, f)

print(f"Training results saved to {filename}")

model_filename = f"{output_dir}/model2_{model_type}_epochs{epochs}_k{kmer}_step{step}_s{sub}_np{n_pairs}_exp.pt"

torch.save(model.state_dict(), model_filename)

print(f"Model saved to {model_filename}")
