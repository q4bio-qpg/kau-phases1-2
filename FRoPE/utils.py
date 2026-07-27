import Levenshtein
from random import choices, randint
from itertools import product
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import ortho_group
from scipy.fftpack import dct
from sklearn.metrics import mean_absolute_error
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor
from functools import partial

base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
kmer_to_index = {}
for kmer in product('ATCG', repeat=5):
    index = 0
    for j, base in enumerate(kmer):
        index += base_to_int[base] * (4 ** (4 - j))
    kmer_to_index[''.join(kmer)] = index

subkmer_to_index = {}
for kmer in product('ATCG', repeat=2):
    index = 0
    for j, base in enumerate(kmer):
        index += base_to_int[base] * (4 ** (1 - j))
    subkmer_to_index[''.join(kmer)] = index

def create_kmer_index_dict(k):
    bases = ['A', 'T', 'C', 'G']
    base_to_int = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    
    kmer_to_index = {}
    for kmer in product(bases, repeat=k):
        index = 0
        for j, base in enumerate(kmer):
            index += base_to_int[base] * (4 ** (k - 1 - j))
        kmer_to_index[''.join(kmer)] = index
    
    return kmer_to_index

def create_pairs(sequences, reference, n=1000):
    n_bases = len(min(sequences)) // 100
    pairs = []
    for i in range(n_bases):
        base = reference[i:i+100]
        for j in range(n):
            seq_idx = randint(0, len(sequences) - 1)
            start = randint(0, len(sequences[seq_idx]) - 100)
            seq = sequences[seq_idx][start:start+100]
            pairs.append((base, seq))
    return pairs


def read_fasta_to_list(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        sequence = ''
        for line in file:
            if line.startswith('>'):
                if sequence:
                    sequence = sequence.replace('N', '')
                    sequences.append(sequence)
                sequence = ''
            else:
                sequence += line.strip()
        if sequence:
            sequence = sequence.replace('N', '')
            sequences.append(sequence)
    return sequences


def weighted_levenshtein(seq1, seq2, match=1, mismatch=0, gap=0):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i * gap
    for j in range(n + 1):
        dp[0][j] = j * gap

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + match
            else:
                dp[i][j] = max(
                    dp[i-1][j-1] + mismatch, 
                    dp[i-1][j] + gap,
                    dp[i][j-1] + gap
                )
    return dp[m][n]

def generate_sequence(length=100):
    return ''.join(choices(['A', 'T', 'C', 'G'], k=length))

def introduce_mutations(sequence, mutation_rate=0.1):
    mutated = list(sequence)
    num_mutations = int(len(mutated) * mutation_rate)
    
    for _ in range(num_mutations):
        pos = randint(0, len(mutated) - 1)
        mutation_type = randint(0, 2)
        
        if mutation_type == 0: 
            mutated[pos] = choices([x for x in 'ATCG' if x != mutated[pos]])[0]
            
        elif mutation_type == 1:
            mutated.insert(pos, choices('ATCG')[0])
            del_pos = randint(0, len(mutated) - 1)
            while del_pos == pos:
                del_pos = randint(0, len(mutated) - 1)
            mutated.pop(del_pos)
            
        elif mutation_type == 2 and len(mutated) > 1: 
            deleted = mutated.pop(pos)
            ins_pos = randint(0, len(mutated) - 1)
            mutated.insert(ins_pos, choices('ATCG')[0])
    
    return ''.join(mutated)

def one_hot_encode(sequence):
    num_kmers = len(sequence) - 5 + 1
    one_hot = np.zeros((num_kmers, 1024), dtype=np.float32)
    
    for i in range(num_kmers):
        kmer = sequence[i:i+5]
        one_hot[i, kmer_to_index[kmer]] = 1
    
    return one_hot

def two_hot_encode(sequence, sukmer_dict, sub_kmer_len=2, kmer_weight=0.5):
    num_kmers = len(sequence) - 5 + 1
    num_subkmers = 5-sub_kmer_len+1
    two_hot = np.zeros((int(num_kmers*num_subkmers), int(1024 + 4**sub_kmer_len)), dtype=np.float32)
    
    for i in range(num_kmers):
        kmer = sequence[i:i+5]
        for j in range(num_subkmers):
            subkmer = kmer[j:j+sub_kmer_len]
            kmer_one_hot = kmer_to_index[kmer]
            subkmer_one_hot = 1024+sukmer_dict[subkmer]
            two_hot[i+j, kmer_one_hot] = np.sqrt(kmer_weight)
            two_hot[i+j, subkmer_one_hot] = np.sqrt(1-kmer_weight)
    
    return two_hot

def get_rotation_matrix(position):
    if position not in rotation_matrices:
        angles = position * theta
        sin_vals = torch.sin(angles)
        cos_vals = torch.cos(angles)
        
        rotation_matrix = torch.eye(dim, dtype=torch.float32)
        
        for i in range(0, dim, 2):
            if i + 1 < dim:  
                rotation_matrix[i, i] = cos_vals[i//2]
                rotation_matrix[i, i+1] = -sin_vals[i//2]
                rotation_matrix[i+1, i] = sin_vals[i//2]
                rotation_matrix[i+1, i+1] = cos_vals[i//2]
        
        rotation_matrices[position] = rotation_matrix.numpy()
    return rotation_matrices[position]

def get_rotation_vectors(position, theta):
    # if position not in rotation_vectors:
    angles = position * theta.repeat_interleave(2)
    sin_vals = torch.sin(angles)
    cos_vals = torch.cos(angles)
        # rotation_vectors[position] = (cos_vals.numpy(), sin_vals.numpy())
    # return rotation_vectors[position]
    return cos_vals.numpy(), sin_vals.numpy()

def encode_hierarchical_sequences_rope_fast(sequences, theta, subkmer_dict, sub_kmer_len=2, kmer_weight=0.5, random_pos=None, absolute=None, random_rotation=None, dct_type=None):
    
    X = np.zeros((len(sequences), int(1024+4**sub_kmer_len)), dtype=np.float32)
    if absolute is None:
        absolute = 1
    for i, sequence in enumerate(sequences):
        two_hot = two_hot_encode(sequence, subkmer_dict, sub_kmer_len, kmer_weight)
        if dct_type is not None:
            two_hot = dct(two_hot, type=dct_type, norm='ortho')
        L = two_hot.shape[0]

        if isinstance(random_pos, float) and random_pos > 1:
            pos_pool_len = int(random_pos * L)
            pos_pool = np.arange(pos_pool_len)
            pos_indices = np.sort(np.random.choice(pos_pool, size=L, replace=False))
        else:
            pos_indices = np.arange(1, L + 1)

        rotated = np.zeros_like(two_hot)
        for pos in range(L):
            cos_vals, sin_vals = get_rotation_vectors(pos_indices[pos]**absolute, theta)
            x = two_hot[pos]
            if random_rotation is not None:
                x = np.dot(x, random_rotation)
            y = np.empty_like(x)
            y[::2] = -x[1::2]
            y[1::2] = x[::2]
            rotated[pos] = x * cos_vals + y * sin_vals
        
        sum_two_hot = np.sum(rotated, axis=0)
        X[i] = sum_two_hot / np.linalg.norm(sum_two_hot)
        
    return X

def encode_sequences_rope_fast(sequences, theta, random_pos=None, absolute=None, random_rotation=None, dct_type=None):
    X = np.zeros((len(sequences), 1024), dtype=np.float32)
    if absolute is None:
        absolute = 1
    for i, sequence in enumerate(sequences):
        one_hot = one_hot_encode(sequence)
        if dct_type is not None:
            one_hot = dct(one_hot, type=dct_type, norm='ortho')
        L = one_hot.shape[0]

        if isinstance(random_pos, float) and random_pos > 1:
            pos_pool_len = int(random_pos * L)
            pos_pool = np.arange(pos_pool_len)
            pos_indices = np.sort(np.random.choice(pos_pool, size=L, replace=False))
        else:
            pos_indices = np.arange(1, L + 1)

        rotated = np.zeros_like(one_hot)
        for pos in range(L):
            cos_vals, sin_vals = get_rotation_vectors(pos_indices[pos]**absolute, theta)
            x = one_hot[pos]
            if random_rotation is not None:
                x = np.dot(x, random_rotation)
            y = np.empty_like(x)
            y[::2] = -x[1::2]
            y[1::2] = x[::2]
            rotated[pos] = x * cos_vals + y * sin_vals
        
        sum_one_hot = np.sum(rotated, axis=0)
        X[i] = sum_one_hot / np.linalg.norm(sum_one_hot)
        
    return X

def encode_sequences_rope(sequences):
    X = np.zeros((len(sequences), 1024), dtype=np.float32)
    
    for i, sequence in enumerate(sequences):
        one_hot = one_hot_encode(sequence)
        rotated = np.zeros_like(one_hot)
        for pos in range(one_hot.shape[0]):
            rot_mat = get_rotation_matrix(pos+1)
            rotated[pos] = one_hot[pos] @ rot_mat
        
        sum_one_hot = np.sum(rotated, axis=0)
        X[i] = sum_one_hot / np.linalg.norm(sum_one_hot)
        
    return X


# def test_encoding_ropefrec_fast(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None, theta=None, random_pos=None, absolute=None, random_rotation=None, dct_type=None, caption=None):
#     lev_distances = []
#     cos_similarities = []

#     if pairs is None:
#         for _ in tqdm(range(num_pairs)):
#             mutation_rate = np.random.uniform(0, max_mutation_rate)
#             original = generate_sequence(100)
#             mutated = introduce_mutations(original, mutation_rate)

#             encoded = encode_sequences_rope_fast([original, mutated], theta, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
#             # print('aaa')
#             cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

#             lev_dist = Levenshtein.distance(original, mutated)
            
#             lev_distances.append(lev_dist)
#             cos_similarities.append(cos_sim)
#     else:
#         for original, mutated in tqdm(pairs):
#             encoded = encode_sequences([original, mutated])
#             cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

#             lev_dist = Levenshtein.distance(original, mutated)
            
#             lev_distances.append(lev_dist)
#             cos_similarities.append(cos_sim)
    
#     plt.figure(figsize=(10, 6))
#     plt.scatter(cos_similarities, lev_distances, alpha=0.6)
#     if caption is not None:
#         plt.title(f"Cosine Similarity vs. Levenshtein Distance (k={k}) (RoPE + Frequencies) ({caption})")
#     else:
#         plt.title(f"Cosine Similarity vs. Levenshtein Distance (k={k}) (RoPE + Frequencies)")
#     plt.xlabel("Cosine Similarity")
#     plt.ylabel("Levenshtein Distance")
#     plt.grid(True)
#     plt.show()

def test_encoding_ropefrec_fast(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None, theta=None, random_pos=None, absolute=None, random_rotation=None, dct_type=None, caption=None):
    lev_distances = []
    cos_similarities = []

    if pairs is None:
        for _ in tqdm(range(num_pairs)):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_sequences_rope_fast([original, mutated], theta, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)

    # Convert to numpy arrays
    x = np.array(cos_similarities)
    y = np.array(lev_distances)

    # Define exponential function for fitting
    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        # Fit the curve
        popt, _ = curve_fit(exp_func, x, y, p0=(1, -4, 10), maxfev=10000)
        y_pred = exp_func(x, *popt)
        mae = mean_absolute_error(y, y_pred)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = None
        mae = None

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label='Data Points')

    if popt is not None:
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = exp_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Exponential Fit')
        plot_title = f"Exponential Fit with MAE = {mae:.2f}"
    else:
        plot_title = "Cosine Similarity vs. Levenshtein Distance (Fit Failed)"

    if caption is not None:
        plot_title += f" ({caption})"

    plt.title(plot_title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Levenshtein Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_encoding_ropefrec_fast_noplot(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None, theta=None, random_pos=None, absolute=None, random_rotation=None, dct_type=None, caption=None):
    lev_distances = []
    cos_similarities = []

    if pairs is None:
        for _ in tqdm(range(num_pairs)):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_sequences_rope_fast([original, mutated], theta, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)

    x = np.array(cos_similarities)
    y = np.array(lev_distances)

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        popt, _ = curve_fit(exp_func, x, y, p0=(1, -4, 10), maxfev=10000)
        y_pred = exp_func(x, *popt)
        mae = mean_absolute_error(y, y_pred)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = None
        mae = None

    return mae

def worker(original, mutated, theta, subkmer_dict, sub_kmer_len, kmer_weight, random_pos, absolute, random_rotation, dct_type):
    encoded = encode_hierarchical_sequences_rope_fast(
        [original, mutated],
        theta,
        subkmer_dict,
        sub_kmer_len=sub_kmer_len,
        kmer_weight=kmer_weight,
        random_pos=random_pos,
        absolute=absolute,
        random_rotation=random_rotation,
        dct_type=dct_type
    )
    cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
    lev_dist = Levenshtein.distance(original, mutated)
    return lev_dist, cos_sim

def unpack_and_call(pair, func):
    return func(*pair)

def test_hierarchical_encoding_ropefrec_fast_noplot_parallel(k=5, sub_kmer_len=2, kmer_weight=0.5, num_pairs=1000,
                                                    max_mutation_rate=0.9, pairs=None, random_pos=None,
                                                    absolute=None, random_rotation=None, dct_type=None,
                                                    caption=None, executor=None): # invalid

    lev_distances = []
    cos_similarities = []

    dim = 1024 + 4**sub_kmer_len
    theta = 1.0 / (10 ** (torch.arange(0, dim, 2).float() / dim))
    subkmer_dict = create_kmer_index_dict(sub_kmer_len)


    if pairs is None:
        pairs = []
        for _ in range(num_pairs):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)
            pairs.append((original, mutated))

    func = partial(worker, theta=theta, subkmer_dict=subkmer_dict,
                    sub_kmer_len=sub_kmer_len, kmer_weight=kmer_weight,
                    random_pos=random_pos, absolute=absolute,
                    random_rotation=random_rotation, dct_type=dct_type)

    results = list(tqdm(executor.map(partial(unpack_and_call, func=func), pairs), total=len(pairs)))

    for lev_dist, cos_sim in results:
        lev_distances.append(lev_dist)
        cos_similarities.append(cos_sim)

    x = np.array(cos_similarities)
    y = np.array(lev_distances)

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        popt, _ = curve_fit(exp_func, x, y, p0=(1, -4, 10), maxfev=10000)
        y_pred = exp_func(x, *popt)
        mae = mean_absolute_error(y, y_pred)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = None
        mae = None

    return mae

def test_hierarchical_encoding_ropefrec_fast_noplot(k=5, sub_kmer_len=2, kmer_weight=0.5, num_pairs=1000, max_mutation_rate=0.9, pairs=None, random_pos=None, absolute=None, random_rotation=None, dct_type=None, caption=None):
    lev_distances = []
    cos_similarities = []
    dim = 1024 + 4**sub_kmer_len
    theta = 1.0 / (10 ** (torch.arange(0, dim, 2).float() / dim))
    if sub_kmer_len != 0:
        subkmer_dict = create_kmer_index_dict(sub_kmer_len)
    if pairs is None:
        for _ in tqdm(range(num_pairs)):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            if sub_kmer_len != 0:
                encoded = encode_hierarchical_sequences_rope_fast([original, mutated], theta, subkmer_dict, sub_kmer_len=sub_kmer_len, kmer_weight=kmer_weight, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
            else: 
                encoded = encode_sequences_rope_fast([original, mutated], theta, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)

    x = np.array(cos_similarities)
    y = np.array(lev_distances)

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        popt, _ = curve_fit(exp_func, x, y, p0=(1, -4, 10), maxfev=10000)
        y_pred = exp_func(x, *popt)
        mae = mean_absolute_error(y, y_pred)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = None
        mae = None

    return mae


def test_hierarchical_encoding_ropefrec_fast(k=5, sub_kmer_len=2, kmer_weight=0.5, num_pairs=1000, max_mutation_rate=0.9, pairs=None, random_pos=None, absolute=None, random_rotation=None, dct_type=None, caption=None):
    lev_distances = []
    cos_similarities = []
    dim = 1024 + 4**sub_kmer_len
    theta = 1.0 / (10 ** (torch.arange(0, dim, 2).float() / dim))
    subkmer_dict = create_kmer_index_dict(sub_kmer_len)
    if pairs is None:
        for _ in tqdm(range(num_pairs)):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_hierarchical_sequences_rope_fast([original, mutated], theta, subkmer_dict, sub_kmer_len=sub_kmer_len, kmer_weight=kmer_weight, random_pos=random_pos, absolute=absolute, random_rotation=random_rotation, dct_type=dct_type)
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]
            lev_dist = Levenshtein.distance(original, mutated)

            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)

    x = np.array(cos_similarities)
    y = np.array(lev_distances)

    def exp_func(x, a, b, c):
        return a * np.exp(b * x) + c

    try:
        popt, _ = curve_fit(exp_func, x, y, p0=(1, -4, 10), maxfev=10000)
        y_pred = exp_func(x, *popt)
        mae = mean_absolute_error(y, y_pred)
    except RuntimeError as e:
        print("Curve fitting failed:", e)
        popt = None
        mae = None

    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.6, label='Data Points')

    if popt is not None:
        x_fit = np.linspace(x.min(), x.max(), 300)
        y_fit = exp_func(x_fit, *popt)
        plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Exponential Fit')
        plot_title = f"Exponential Fit with MAE = {mae:.2f}"
    else:
        plot_title = "Cosine Similarity vs. Levenshtein Distance (Fit Failed)"

    if caption is not None:
        plot_title += f" ({caption})"

    plt.title(plot_title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Levenshtein Distance")
    plt.legend()
    plt.grid(True)
    plt.show()

def test_encoding_ropefrec(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None):
    lev_distances = []
    cos_similarities = []

    if pairs is None:
        for _ in tqdm(range(num_pairs)):
            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_sequences_rope([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = Levenshtein.distance(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = Levenshtein.distance(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(cos_similarities, lev_distances, alpha=0.6)
    plt.title(f"Cosine Similarity vs. Levenshtein Distance (k={k}) (RoPE + Frequencies)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Levenshtein Distance")
    plt.grid(True)
    plt.show()

def get_all_kmers(k=5):
    return [''.join(p) for p in product('ATCG', repeat=k)]
    
def test_encoding_frec(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None):
    
    lev_distances = []
    cos_similarities = []

    def encode_sequences(sequences):
        X = np.zeros((len(sequences), 1024), dtype=np.float32)
        
        for i, sequence in enumerate(sequences):
            one_hot = one_hot_encode(sequence)    
            sum_one_hot = np.sum(one_hot, axis=0)
            X[i] = sum_one_hot / np.linalg.norm(sum_one_hot)
            # X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    if pairs is None:
        for _ in range(num_pairs):

            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = Levenshtein.distance(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in pairs:
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = Levenshtein.distance(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(cos_similarities,lev_distances, alpha=0.6)
    plt.title(f"Cosine Similarity vs. Levenshtein Distance (k={k}) (Frequencies only)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Levenshtein Distance")
    plt.grid(True)
    plt.show()

def test_encoding_frec_overlap(pairs):
    overlap_area = [100-i for i in range(100)]
    cos_similarities = []

    def encode_sequences(sequences):
        X = np.zeros((len(sequences), 1024), dtype=np.float32)
        
        for i, sequence in enumerate(sequences):
            one_hot = one_hot_encode(sequence)    
            sum_one_hot = np.sum(one_hot, axis=0)
            X[i] = sum_one_hot / np.linalg.norm(sum_one_hot)
            # X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    for original, mutated in pairs:
        encoded = encode_sequences([original, mutated])
        cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

        cos_similarities.append(cos_sim)

    plt.figure(figsize=(10, 6))
    plt.scatter(cos_similarities,overlap_area, alpha=0.6)
    plt.title("Cosine Similarity vs. Overlap area (k=5) (Frequencies only)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Overlap area")
    plt.grid(True)
    plt.show()

def create_overlapping_pairs(reference):
    base = reference[:100]
    pairs = []
    for i in range(100):
        pairs.append((base,reference[i:i+100]))

    return pairs

def test_encoding_frec_weighted(k=5, num_pairs=1000, max_mutation_rate=0.9, pairs=None):
    
    lev_distances = []
    cos_similarities = []

    def encode_sequences(sequences):
        X = np.zeros((len(sequences), 1024), dtype=np.float32)
        
        for i, sequence in enumerate(sequences):
            one_hot = one_hot_encode(sequence)    
            sum_one_hot = np.sum(one_hot, axis=0)
            X[i] = sum_one_hot / np.linalg.norm(sum_one_hot)
            # X = X / np.linalg.norm(X, axis=1, keepdims=True)
        return X

    if pairs is None:
        for _ in tqdm(range(num_pairs)):

            mutation_rate = np.random.uniform(0, max_mutation_rate)
            original = generate_sequence(100)
            mutated = introduce_mutations(original, mutation_rate)

            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = 100 - weighted_levenshtein(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    else:
        for original, mutated in tqdm(pairs):
            encoded = encode_sequences([original, mutated])
            cos_sim = cosine_similarity([encoded[0]], [encoded[1]])[0][0]

            lev_dist = 100 - weighted_levenshtein(original, mutated)
            
            lev_distances.append(lev_dist)
            cos_similarities.append(cos_sim)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(cos_similarities,lev_distances, alpha=0.6)
    plt.title(f"Cosine Similarity vs. Weighted Levenshtein Distance (k={k}) (Frequencies only)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Weighted Levenshtein Distance")
    plt.grid(True)
    plt.show()
