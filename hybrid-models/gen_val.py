from utils import *
import Levenshtein
import torch
import os
import pickle

kmer = 25
subkmer = 2
seq_len = 100
step_size = 25

sequences_dict = fasta_to_dict('data/sequences.fasta')

for key in sequences_dict.keys():
    sequences_dict[key] = skip_N(sequences_dict[key])[:5339]
    print(len(sequences_dict[key]))

reference_dict = fasta_to_dict('data/reference.fasta')
# print(reference_dict.keys())
reference_dict['NC_001422.1'] = reference_dict['NC_001422.1'][:5339]

split_step = 50

splitted_base_val = split_string(reference_dict['NC_001422.1'], 100, split_step)

splitted_seqs_val_prev = [ [] for _ in range(len(splitted_base_val))]
for key in sequences_dict.keys():
    splitted_seq = split_string(sequences_dict[key], 100, split_step)
    
    for i in range(len(splitted_seq)):
        splitted_seqs_val_prev[i].append(splitted_seq[i])
        print(len(splitted_seq[i]))

base_genomes_val_for_pairs = []
for split in splitted_base_val:
    for i in range(len(list(sequences_dict.values()))):
        base_genomes_val_for_pairs.append(split)

seqs_list_val = []
for col in splitted_seqs_val_prev:
    for split in col:
        seqs_list_val.append(split)

lev_dist_val = []
for i in range(len(base_genomes_val_for_pairs)):
    lev_dist_val.append(Levenshtein.distance(base_genomes_val_for_pairs[i], seqs_list_val[i])) 

first_tensors_val = [(torch.from_numpy(frequency_tensor(val_first_sequence, kmer, subkmer, step_size)).type(torch.float) / (kmer - 1)).flatten() for val_first_sequence in base_genomes_val_for_pairs]
second_tensors_val = [(torch.from_numpy(frequency_tensor(val_second_sequence, kmer, subkmer, step_size)).type(torch.float) / (kmer - 1)).flatten() for val_second_sequence in seqs_list_val]

data_dir = './data'
os.makedirs(data_dir, exist_ok=True)

# Generate the filename based on the parameters
filename = f"{data_dir}/val_data_k{str(kmer)}_s{str(subkmer)}_l{str(seq_len)}_step{str(step_size)}_np{len(base_genomes_val_for_pairs)}.pkl"

# Prepare a dictionary with the variables to save
data_dict = {
    'val_first_sequences': base_genomes_val_for_pairs,
    'val_second_sequences': seqs_list_val,
    'first_tensors_val': first_tensors_val,
    'second_tensors_val': second_tensors_val,
    'val_distances': lev_dist_val
}

# Save the dictionary as a pickle file
with open(filename, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Data saved to {filename}")