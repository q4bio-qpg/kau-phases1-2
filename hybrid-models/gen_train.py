from utils import *
import Levenshtein
import torch
import pickle
import os

kmer = 25
subkmer = 2
seq_len = 100
step_size = 25

nucleotide_frequencies = {'A': 0.23929321148185556, 'C': 0.21344851880720792, 'T': 0.3142065126089769, 'G': 0.2330517571019596}

n_pairs = 10000

mutation_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1]

n_mutation_rates = len(mutation_rates)

train_first_sequences = []
train_second_sequences = []
train_distances = []

for mutation_rate in mutation_rates:
    for i in range(n_pairs // n_mutation_rates):
        base_sequence = generate_genome_sequence(length=100, freq_dict=nucleotide_frequencies)
        mutated_sequence = random_mutations(base_sequence, mutation_rate=mutation_rate, freq_dict=nucleotide_frequencies)
        levenshtein_distance = Levenshtein.distance(base_sequence, mutated_sequence) 

        train_first_sequences.append(base_sequence)
        train_second_sequences.append(mutated_sequence)
        train_distances.append(levenshtein_distance)

first_tensors_train = [(torch.from_numpy(frequency_tensor(train_first_sequence, kmer, subkmer, step_size)).type(torch.float) / (kmer - 1)).flatten() for train_first_sequence in train_first_sequences]
second_tensors_train = [(torch.from_numpy(frequency_tensor(train_second_sequence, kmer, subkmer, step_size)).type(torch.float) / (kmer - 1)).flatten() for train_second_sequence in train_second_sequences]

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Generate the filename based on the parameters
filename = f"{data_dir}/train_data_k{str(kmer)}_s{str(subkmer)}_l{str(seq_len)}_step{str(step_size)}_np{n_pairs}.pkl"

# Prepare a dictionary with the variables to save
data_dict = {
    'train_first_sequences': train_first_sequences,
    'train_second_sequences': train_second_sequences,
    'first_tensors_train': first_tensors_train,
    'second_tensors_train': second_tensors_train,
    'train_distances': train_distances
}

# Save the dictionary as a pickle file
with open(filename, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Data saved to {filename}")

