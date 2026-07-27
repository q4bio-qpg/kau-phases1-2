from torch.utils.data import Dataset
import random
import torch
from collections import Counter
import numpy as np
from copy import deepcopy
from collections import defaultdict
from sklearn.cluster import KMeans

class GenomeDataset(Dataset):
    def __init__(self, kmers, targets=None, mutants=True, embedding_func_mode = 'binary', quantum_mode='multistate', n_clusters=None, clusters=None, extra_seq=64, minibatch_size=None, **kwargs):
        
        self.input_kmers = kmers
        self.input_targets = targets
        self.mutants = mutants
        self.embedding_func_mode = embedding_func_mode
        self.quantum_mode = quantum_mode
        self.n_clusters = n_clusters
        self.input_clusters = clusters
        self.init_clusters = defaultdict(list)
        self.set_emb_func()
        if not minibatch_size:
          self.minibatch_size = extra_seq
        else:
           self.minibatch_size=minibatch_size
        self.extra_seq = extra_seq

        if self.quantum_mode == 'multistate':
           self.prepare_multistate_db()
        elif self.quantum_mode == 'singlestate':
           self.prepare_singlestate_db()
        
        self.pick_extra_seq()

        if self.mutants:
           self.gen_mut_dataset(**kwargs)

        self.embedd_clusters(**kwargs)


    def set_emb_func(self):
        if self.embedding_func_mode == 'binary':
           self.embedding_func = self.binary_embedding
        else:
           self.embedding_func = self.frequency_tensor

    def embedd_clusters(self, **kwargs):
        self.embedded_clusters = defaultdict(list)
        for clus, data in self.clusters.items():
           for kmer, target in data:
              self.embedded_clusters[clus].append((self.embedding_func(kmer, **kwargs), target))
    
    def kmeans_clustering(self, features, kmers, n_clusters):
        if len(features) != len(kmers):
          raise ValueError(f'frovided features list (n={len(features)}) do not match provided kmers (n={len(kmers)})')
        feature_vectors = np.array(features)

      # Perform K-Means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        print(f'Started calculating {n_clusters} clusters for {len(self.input_targets)} kmers')
        labels = kmeans.fit_predict(feature_vectors)
        return labels
    
    def pick_extra_seq(self):
        filtered_clusters = {}
        for clus, data in self.init_clusters.items():
          new_data = random.choices(data, k=self.extra_seq)
          filtered_clusters[clus] = new_data
        self.clusters = filtered_clusters

    def prepare_multistate_db(self):
        if self.quantum_mode != 'multistate':
          raise AttributeError('quantum_mode is incompatible with this function')
        
        self.database_size = int(np.log2(len(self.input_targets[0])))
        labels = self.kmeans_clustering(self.input_targets, self.input_kmers, int(2**self.database_size))

        for l, s, t in zip(labels, self.input_kmers, self.input_targets):
          self.init_clusters[l].append((s,t))
        
        self.clusters = deepcopy(self.init_clusters)
    
    def prepare_singlestate_db(self):
        if self.quantum_mode != 'singlestate':
            raise AttributeError('quantum_mode is incompatible with this function')
        
        if self.n_clusters and self.input_targets and not self.input_clusters:
          self.database_size = int(np.ceil(np.log2(self.n_clusters)))
          print('Calculating clusters since no input clusters provided and n_clusters and multistate target are provided')
          labels = self.kmeans_clustering(self.input_targets, self.input_kmers, int(2**self.database_size))
          
          #renumber cluster indexes according to position in ref seq
          _targets_clusters = defaultdict(list)
          for l, t in zip(labels, self.input_targets):
            _targets_clusters[l].append(t)
          temp = {}
          for i, t in _targets_clusters.items():
            temp[i] = (torch.stack(t).mean(dim=0) * torch.arange(len(t[0]))).mean().item()
          cl2pos = {i:j for j,(i,v) in enumerate(sorted(temp.items(), key=lambda x: x[1]))}
          
          for l, s in zip(labels, self.input_kmers):
            self.init_clusters[cl2pos[l]].append((s, self.number_to_base2_tensor(cl2pos[l], int(2**self.database_size))))
        
        elif self.input_clusters:
          print('using provided clusters')

          self.database_size = int(np.ceil(np.log2(len(self.input_clusters))))

          for index, kmers in self.input_clusters.items():
            target = self.number_to_base2_tensor(index, int(2**self.database_size))
            for kmer in kmers:
              self.init_clusters[index].append((kmer, target))
        else:
           raise ValueError('insufficient information to prepare db for singlestate mode')
        
        self.clusters = deepcopy(self.init_clusters)


    def number_to_base2_tensor(self, number, num_bits):
        # Convert number to binary and pad with leading zeros
        target = torch.zeros(num_bits)
        target[number] = 1
        return target 

    def random_mutations(self, genome_sequence, mutation_rate):
      bases = ['A', 'T', 'C', 'G']
      mutated_sequence = []

      for base in genome_sequence:
          if random.random() < mutation_rate:
              # Mutate the base to a different random base
              new_base = random.choice([b for b in bases if b != base])
              mutated_sequence.append(new_base)
          else:
              # Keep the original base
              mutated_sequence.append(base)

      return ''.join(mutated_sequence)
    
    def split_string(self, s, chunk_size):
      splitted = [s[i:i+chunk_size] for i in range(0, len(s), chunk_size)]
      if len(splitted[-1]) != 100:
          return splitted[:-1]
      return splitted
    
    def generate_genome_seed(self, length, seed):
      random.seed(seed)
      bases = ['A', 'T', 'C', 'G']
      genome_sequence = ''.join(random.choice(bases) for _ in range(length))
      return genome_sequence

    def binary_embedding(self, seq, **kwargs):
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

    def gen_mut_dataset(self, num_mutants=1, mutation_ratio=0.05, **kwargs):
      print('generating mutants')
      for clus, data in self.clusters.items():
         mutants = []
         for kmer, target in data:
            for i in range(num_mutants):
              mutant_mer = self.random_mutations(kmer, mutation_ratio)
              mutants.append((mutant_mer, target))
         self.clusters[clus] += mutants
    
    def kmer_to_index(self, kmer):
      """Converts a kmer (string) to an index."""
      base_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N':3}
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

    def __len__(self):
        return len(self.embedded_clusters)

    def __getitem__(self, idx):
        kmers, targets = [],[]
        for i in range(self.minibatch_size):
          kmer, target = random.choice(self.embedded_clusters[idx])
          kmers.append(kmer)
          targets.append(target)
        kmers = torch.stack(kmers)
        targets = torch.stack(targets)
        return kmers, targets