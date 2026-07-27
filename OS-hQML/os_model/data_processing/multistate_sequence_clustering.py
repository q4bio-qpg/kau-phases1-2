import torch
import copy
from utils import fasta_to_dict
import math
import Levenshtein
import pickle

def gen_kmers(seq, kmer_length=100, step=1):
    kmers = []
    i = 0
    for j in range(((len(seq)-kmer_length) // step) + 1):
        kmers.append(seq[i:i+kmer_length])
        i += step
    return kmers

def _prep_data(sequence_file, kmer_size):
    seqs = list(fasta_to_dict(sequence_file).values())
    ref_seq = max(seqs, key=len)
    ref_double_mers = [ref_seq[i: i+kmer_size*2] for i in range(0,len(ref_seq), kmer_size) if len(ref_seq[i: i+kmer_size*2])>=kmer_size]
    ref_double_mers[-1] = ref_double_mers[-1] + '-'*(kmer_size*2 - len(ref_double_mers[-1]))
    n_qubits = math.ceil(math.log2(len(ref_double_mers)))
    zeros = torch.zeros(int(2**n_qubits), dtype=torch.float32)
    print(f"chosen kmer size on reference (longest) sequence produced {len(ref_double_mers)} doublemers resulting in {n_qubits} qubits")
    return seqs, ref_double_mers, zeros

def process_kmers(seqs, ref_double_mers, kmer_size, zeros):
    kmers = []
    targets = []

    all_kmers = set(sum([gen_kmers(seq, kmer_length=kmer_size, step=1) for seq in seqs], []))

    for kmer in all_kmers:
        dmers = {i: Levenshtein.distance(dmer, kmer, weights=(100,100,1)) for i, dmer in enumerate(ref_double_mers)}
        top2 = sorted(dmers, key=dmers.get)[:2]

        precise_alignment = {}
        for topd in top2:
            d = ref_double_mers[topd]
            dkmers = gen_kmers(d, kmer_length=kmer_size, step=1)
            for i, dkmer in enumerate(dkmers):
                precise_alignment[f"{topd}_{i}"] = Levenshtein.distance(dkmer, kmer, weights=(100,100,1))
        
        top1 = sorted(precise_alignment, key=precise_alignment.get)[0]
        d, pos = map(int, top1.split('_'))

        target = copy.deepcopy(zeros)
        target[d] = 1 - (pos / kmer_size)
        target[d+1] = pos / kmer_size

        kmers.append(kmer)
        targets.append(target)

    return kmers, targets


def create_data(fasta_file, kmer_size, save_data=True):
    
    seqs, ref_double_mers, zeros = _prep_data(fasta_file, kmer_size)

    kmers, targets = process_kmers(seqs, ref_double_mers, kmer_size, zeros)

    if save_data:
        data = (kmers, targets)
        with open('kmer_data.pkl', 'wb') as o:
            pickle.dump(data, o)
    
    return kmers, targets


