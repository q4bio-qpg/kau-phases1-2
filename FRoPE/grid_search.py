from tqdm import tqdm
from utils import *
import random
import itertools
import pickle

kmer_weights = [i/10 for i in range(3,11)]
subkmer_lens = [0, 2, 3]
random_poss = [1, 1.5, 2, 5]
absolutes = [0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
dct_types = [None, 1, 2, 3, 4]

all_combinations = list(itertools.product(kmer_weights, subkmer_lens, random_poss, absolutes, dct_types))

filtered_combinations = [
    combo for combo in all_combinations
    if not (combo[1] == 0 and combo[0] != kmer_weights[0]) 
]

random.seed(42)
# sampled_combinations = random.sample(all_combinations, n_samples)

results = []

for idx, (kmer_weight, subkmer_len, random_pos, absolute, dct_type) in tqdm(enumerate(filtered_combinations)):
    try:
        maes = []
        for i in range(3):
            mae = test_hierarchical_encoding_ropefrec_fast_noplot(
            k=5,
            kmer_weight=kmer_weight,
            sub_kmer_len=subkmer_len,
            random_pos=random_pos,
            absolute=absolute,
            dct_type=dct_type,
            num_pairs=1000,
            max_mutation_rate=0.9
            )
            maes.append(mae)

        results.append({
            "kmer_weight": kmer_weight,
            "subkmer_len": subkmer_len,
            "random_pos": random_pos,
            "absolute": absolute,
            "dct_type": dct_type,
            "mae_list": maes,
            "mae_mean": (maes[0] + maes[1] + maes[2]) / 3
        })
    except Exception as e:
        print(f"Error with combination {idx + 1}: {e}")

with open("random_grid_search_results.pkl", "wb") as f:
    pickle.dump(results, f)
