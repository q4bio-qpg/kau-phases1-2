from ml.HybridModel import NewHybridModel
from ml.EmbeddingModel import EditDistanceModel
import torch
import json
import numpy as np
from collections import Counter, defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_weights=None, model_config=None):
    
    if model_config:
        model_config_path = model_config
    else:
        model_config_path = "models/config/model_config.json"

    with open(model_config_path, "r") as f:
        model_config = json.load(f)

    n_qubits = int(model_config['architecture']['n_qubits'])
    n_lin_layers = int(model_config['architecture']['n_lin_layers'])
    n_quant_layers = int(model_config['architecture']['n_quant_layers'])
    
    if 'training_params' in model_config.keys():
        extraseq = model_config['training_params']['extraseq']
        n_muts = model_config['training_params']['n_muts']

    if model_weights:
        model_weights_path = model_weights
    else:
        model_weights_path = f"models/weights/model_dct_{int(2**n_qubits)}_multistate_{extraseq}extaseq_{n_muts}mut_{n_quant_layers}ql_{n_lin_layers}ll.pth"

    embedding_model = EditDistanceModel()
    hybrid_model = NewHybridModel(embedding_model, n_qubits=n_qubits, n_layers=n_quant_layers, depth=n_lin_layers, freeze_embed_model=False)

    state_dict = torch.load(model_weights_path, weights_only=True)

    try:
        for key in ["embedding_model.conv2.weight", "embedding_model.conv2.bias", "embedding_model.bn2.weight", "embedding_model.bn2.bias", "embedding_model.bn2.running_mean", "embedding_model.bn2.running_var", "embedding_model.bn2.num_batches_tracked", "embedding_model.bn3.weight", "embedding_model.bn3.bias", "embedding_model.bn3.running_mean", "embedding_model.bn3.running_var", "embedding_model.bn3.num_batches_tracked", "embedding_model.bn4.weight", "embedding_model.bn4.bias", "embedding_model.bn4.running_mean", "embedding_model.bn4.running_var", "embedding_model.bn4.num_batches_tracked", "embedding_model.fc2.weight", "embedding_model.fc2.bias", "embedding_model.fc.weight", "embedding_model.fc.bias"]:
            state_dict.pop(key)
    except:
        None

    hybrid_model.load_state_dict(state_dict)

    hybrid_model.eval()
    hybrid_model.to(DEVICE)

    return hybrid_model

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

def kmer_to_index(kmer):
      """Converts a kmer (string) to an index."""
      base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':3}
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
    
def frequency_tensor(sequence, kmer_length=25, subkmer_length=2, step_size=25):
      """Creates a 2D tensor of subkmer frequencies for each kmer in the sequence with a specified step size."""
      num_kmers = (len(sequence) - kmer_length) // step_size + 1
      tensor = np.zeros((num_kmers, 4**subkmer_length))
      
      for i in range(0, len(sequence) - kmer_length + 1, step_size):
          kmer = sequence[i:i+kmer_length]
          frequencies = subkmer_frequencies_in_kmer(kmer, subkmer_length)
          tensor[i // step_size, :] = frequencies
        
      tensor = (torch.from_numpy(tensor).type(torch.float) / (kmer_length - 1)).flatten() 
      return tensor

def process_input(seq_file):
    seq_dict = fasta_to_dict(seq_file)
    embeddings = {}
    for seq_id, seq in seq_dict.items():
        embeddings[seq_id] = frequency_tensor(seq)
    return embeddings

def main(seq_file, out_file, model_weights=None, model_config=None):
    embeddings = process_input(seq_file)

    ids = list(embeddings.keys())
    kmers = torch.stack(list(embeddings.values())).to(DEVICE)

    hybrid_model = load_model(model_weights=model_weights, model_config=model_config)

    with torch.no_grad():
        state_vecs = hybrid_model(kmers)
    
    probs = (state_vecs*100).int()
    windows_counts = probs[:,:-1] + probs[:, 1:]
    indexes = (windows_counts == torch.max(windows_counts, dim=1).values.unsqueeze(1))
    probs, indexes = probs.cpu().numpy(), indexes.cpu().numpy()

    out = defaultdict(dict)

    for i, id in enumerate(ids):
        out[id]['probs'] = probs[i].tolist()
        windows = indexes[i].nonzero()[0].tolist()
        positions = set()
        exact_positions = []
        for w in windows:
            positions.add(w)
            positions.add(w+1)
            ratios = probs[i][np.array([w,w+1])]
            ratios = ratios / np.sum(ratios)
            exact_positions.append(int((ratios[0]*w + ratios[1]*(w+1))*100))
        positions = list(sorted(positions))
        out[id]['matched_100mer_indexes'] = positions
        out[id]['approximate_exact_position(s)'] = exact_positions

    out = dict(out)

    with open(out_file, 'w') as out_json:
        json.dump(out, out_json, indent=4)
    
    return out

import argparse

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the model on a sequence file and output results.")
    
    # Add arguments for the sequence file and output file
    parser.add_argument(
        '--seq_file', type=str, required=True, help="Path to the input sequence file (FASTA format)"
    )
    parser.add_argument(
        '--out_file', type=str, required=True, help="Path to save the output JSON file"
    )
    
    # Optionally add arguments for model weights and config file
    parser.add_argument(
        '--model_weights', type=str, help="Path to model weights file", default=None
    )
    parser.add_argument(
        '--model_config', type=str, help="Path to model config file", default=None
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Call main with arguments from command-line
    main(args.seq_file, args.out_file, model_weights=args.model_weights, model_config=args.model_config)
