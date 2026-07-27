from .ConnectorModel import Classical2QuantumConnector
import torch
from torch import nn
import pennylane as qml
from collections import Counter
import numpy as np

class NewHybridModel(nn.Module):
    def __init__(self, embedding_model, n_qubits, n_layers, depth, freeze_embed_model=False):
        super(NewHybridModel, self).__init__()
        self.embedding_model = embedding_model
        self.emb_out = self.embedding_model.out_dim
        self.connector_model = Classical2QuantumConnector(self.emb_out, n_qubits, n_layers, depth=depth)
        self.n_qubits = n_qubits
        self.qnode = qml.QNode(self.quantum_circuit, qml.device("default.qubit", wires=range(n_qubits)), interface='torch')
        self.qnode_inference = qml.QNode(self.quantum_circuit_inference, qml.device("default.qubit", wires=range(n_qubits)), interface='torch')
        if freeze_embed_model:
            for param in self.embedding_model.parameters():
                param.requires_grad = False
        else:
            for param in self.embedding_model.parameters():
                param.requires_grad = True

    def forward(self, x):
        # Get parameters from the classical model
        #print(x.shape)
        embeddings = self.embedding_model.forward(x)
        #print(embeddings.shape)
        params = self.connector_model(embeddings)
        #print(params.shape)
        # Convert to numpy and run the quantum circuit
        #output = torch.tensor(self.qnode(params,self.n_qubits),dtype=torch.float32, requires_grad=True)
        output = self.qnode(params,self.n_qubits) #dtype=torch.float32, requires_grad=True)

        #output = self.qnode(params,self.n_qubits),dtype=torch.float32, requires_grad=True)
        
        #print(output)

        return output
    

    def quantum_circuit(self, params, n_qubits):
    # Example: Apply rotation gates based on the parameters
      qml.StronglyEntanglingLayers(params, range(n_qubits))
      probs = qml.probs()
      #print(probs.return_type)
      return probs  # Measure the first qubit
    
    def quantum_circuit_inference(self, params, n_qubits):
        # Example: Apply rotation gates based on the parameters
        qml.StronglyEntanglingLayers(params, range(n_qubits))
        return qml.counts()
      # Measure the first qubit

    def binary_embedding(self, seq):
          bin_dict = {
              'A':[0,0],
              'T':[0,1],
              'G':[1,0],
              'C':[1,1]
          }
          embedding = []
          for i in range(len(seq)):
            embedding+= bin_dict[seq[i]]
          embedding = torch.tensor(embedding, dtype=torch.float32)
          return embedding
    
    def kmer_to_index(self, kmer):
      """Converts a kmer (string) to an index."""
      base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
      index = 0
      for char in kmer:
          index = 4 * index + base_to_index[char]
      return index
    
    def subkmer_frequencies_in_kmer(self, kmer, subkmer_length):
      """Calculates the frequency of each subkmer in a kmer."""
      subkmer_counts = Counter(kmer[i:i+subkmer_length] for i in range(len(kmer) - subkmer_length + 1))
      frequencies = np.zeros(4**subkmer_length)
      for subkmer, count in subkmer_counts.items():
          index = self.kmer_to_index(subkmer)
          frequencies[index] = count
      return frequencies
    
    def frequency_tensor(self, sequence, kmer_length, subkmer_length, step_size=1, **kwargs):
      """Creates a 2D tensor of subkmer frequencies for each kmer in the sequence with a specified step size."""
      num_kmers = (len(sequence) - kmer_length) // step_size + 1
      tensor = np.zeros((num_kmers, 4**subkmer_length))
      
      for i in range(0, len(sequence) - kmer_length + 1, step_size):
          kmer = sequence[i:i+kmer_length]
          frequencies = self.subkmer_frequencies_in_kmer(kmer, subkmer_length)
          tensor[i // step_size, :] = frequencies
        
      tensor = (torch.from_numpy(tensor).type(torch.float) / (kmer_length - 1)).flatten() 
      return tensor


    def inference(self, input_embedding, nshots):
      embeddings = self.embedding_model.forward(input_embedding)
      params = self.connector_model(embeddings)
      quantum_params_double = params.double()
      #qnode = qml.QNode(self.quantum_circuit_inference, qml.device("default.qubit"))
      counts = self.qnode_inference(quantum_params_double, self.n_qubits, shots=nshots)
      #counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
      return counts