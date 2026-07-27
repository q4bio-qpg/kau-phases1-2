from pennylane import numpy as np
import random
from collections import Counter


def fasta_to_dict(fasta_file):
    sequences = {}
    with open(fasta_file, 'r') as file:
        header = None
        sequence = []
        for line in file:
            line = line.strip()
            if line.startswith('>'):
                if header:
                    sequences[header] = ''.join(sequence)
                header = line[1:]  # Remove the '>' symbol
                sequence = []  # Reset sequence for the next record
            else:
                sequence.append(line)
        if header:  # Add the last sequence
            sequences[header] = ''.join(sequence)
    return sequences

def compute_frequencies(sequences_dict):
    # Initialize a dictionary to store total counts of A, C, T, G
    total_counts = {'A': 0, 'C': 0, 'T': 0, 'G': 0}
    total_bases = 0  # To track the total number of bases (for normalization)

    # Loop over each sequence in the sequences_dict
    for sequence in sequences_dict.values():
        # Skip 'N' characters in the sequence
        sequence = skip_N(sequence)
        
        # Count occurrences of A, C, T, G in the sequence
        total_counts['A'] += sequence.count('A')
        total_counts['C'] += sequence.count('C')
        total_counts['T'] += sequence.count('T')
        total_counts['G'] += sequence.count('G')
        
        # Update the total number of bases processed
        total_bases += len(sequence)

    # Calculate frequencies as ratios of total bases
    frequencies = {base: count / total_bases for base, count in total_counts.items()}
    return frequencies

def generate_genome(length):
    bases = ['A', 'T', 'C', 'G']
    genome_sequence = ''.join(random.choice(bases) for _ in range(length))
    return genome_sequence

# def random_mutations(genome_sequence, mutation_rate):
#     bases = ['A', 'T', 'C', 'G']
#     mutated_sequence = []

#     for base in genome_sequence:
#         if random.random() < mutation_rate:
#             # Mutate the base to a different random base
#             new_base = random.choice([b for b in bases if b != base])
#             mutated_sequence.append(new_base)
#         else:
#             # Keep the original base
#             mutated_sequence.append(base)
    
#     return ''.join(mutated_sequence)


def random_mutations(genome_sequence, mutation_rate, freq_dict=None):
    bases = ['A', 'T', 'C', 'G']
    mutated_sequence = []
    
    for base in genome_sequence:
        if random.random() < mutation_rate:
            if freq_dict:
                # Exclude the current base from the selection
                new_bases = [b for b in bases if b != base]
                new_weights = [freq_dict[b] for b in new_bases]
                
                # Normalize the weights to sum to 1
                total_new_weights = sum(new_weights)
                new_weights = [w / total_new_weights for w in new_weights]
                
                # Mutate based on normalized weights, ensuring the base is different
                new_base = random.choices(new_bases, weights=new_weights, k=1)[0]
            else:
                # Mutate the base to a different random base
                new_base = random.choice([b for b in bases if b != base])
            
            mutated_sequence.append(new_base)
        else:
            # Keep the original base
            mutated_sequence.append(base)
    
    return ''.join(mutated_sequence)

def generate_genome_seed(length, seed):
    random.seed(seed)
    bases = ['A', 'T', 'C', 'G']
    genome_sequence = ''.join(random.choice(bases) for _ in range(length))
    return genome_sequence

random.seed(42)

def replace_N_with_random(sequence):
    symbols = ['A', 'C', 'T', 'G']
    return ''.join(random.choice(symbols) if base == 'N' else base for base in sequence)

def replace_N_in_sequences(sequences):
    return [replace_N_with_random(seq) for seq in sequences]

def kmer_to_index(kmer):
    """Converts a kmer (string) to an index."""
    base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    index = 0
    for char in kmer:
        index = 4 * index + base_to_index[char]
    return index

def subkmer_frequencies_in_kmer(kmer, subkmer_length):
    """Calculates the frequency of each subkmer in a kmer."""
    subkmer_counts = Counter(kmer[i:i+subkmer_length] for i in range(len(kmer) - subkmer_length + 1))
    frequencies = np.zeros(4**subkmer_length)
    for subkmer, count in subkmer_counts.items():
        index = kmer_to_index(subkmer)
        frequencies[index] = count
    return frequencies

def frequency_tensor(sequence, kmer_length, subkmer_length, step_size=1):
    """Creates a 2D tensor of subkmer frequencies for each kmer in the sequence with a specified step size."""
    num_kmers = (len(sequence) - kmer_length) // step_size + 1
    tensor = np.zeros((num_kmers, 4**subkmer_length))
    
    for i in range(0, len(sequence) - kmer_length + 1, step_size):
        kmer = sequence[i:i+kmer_length]
        frequencies = subkmer_frequencies_in_kmer(kmer, subkmer_length)
        tensor[i // step_size, :] = frequencies
        
    return tensor

def fidelity(vector1, vector2):
    fidelity = np.abs(np.sum(vector1 * np.conj(vector2)))
    # Compute the dot product
    # dot_product = np.dot(vector1, vector2)
    # Compute the absolute value of the dot product
    # abs_dot_product = np.abs(dot_product)
    # return abs_dot_product
    return fidelity

# def split_string(s, chunk_size):
#     splitted = [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
#     if len(splitted[-1]) != 100:
#         return splitted[:-1]
    
#     return splitted

def split_string(s, chunk_size, step=None):
    # If step is not provided, default to chunk_size
    if step is None:
        step = chunk_size
    
    # Split the string into chunks with the given chunk_size and step
    splitted = [s[i:i+chunk_size] for i in range(0, len(s), step)]
    
    # If the last chunk is not of the specified chunk_size, remove it
    for i in range(1, len(splitted)):
        if len(splitted[-i]) == chunk_size:
            if i == 1:
                return splitted[:-i]
            return splitted[:-i + 1]
    # if len(splitted[-1]) != chunk_size:
    #     return splitted[:-1]
    
    return splitted


def skip_N(sequence):
    return ''.join('' if base == 'N' else base for base in sequence)

def generate_genome_sequence(length, freq_dict):
    # Create a list of nucleotide bases based on their frequencies
    bases = list(freq_dict.keys())
    weights = list(freq_dict.values())
    
    # Generate the genome sequence
    genome_sequence = ''.join(random.choices(bases, weights=weights, k=length))
    return genome_sequence

def get_data_dict(data_dict, type='train'):
    first_tensors = data_dict[f'first_tensors_{type}']

    second_tensors = data_dict[f'second_tensors_{type}']

    targets_list = []
    distances = data_dict[f'{type}_distances']
    for distance in distances:
        distance = distance / 60
        # distance = max(0, min(distance, 1))
        # distance = np.exp(3*distance - 3)
        targets_list.append(distance)

    return first_tensors, second_tensors, targets_list