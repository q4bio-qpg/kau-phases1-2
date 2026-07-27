from copy import deepcopy
import statistics

def eval_model(hybrid_model, dataset, n_eval, n_counts, mode='SR-TS'):
    '''
    mode possible parameters:
    SR-TS - Success rate on Training set
    SR-M - Success rate on mutants from Training set
    SR-IS - success rate on random samples from Initial set
    '''

    hybrid_model.cpu()

    if mode=='SR-TS':
        n_eval = 1
        clusters=dataset.embedded_clusters
    elif mode=='SR-IS':
        dataset1 = deepcopy(dataset)
        dataset1.extra_seq=n_eval
        dataset1.minibatch_size=dataset.extra_seq
        dataset1.pick_extra_seq()
        dataset1.embedd_clusters(kmer_length=25, subkmer_length=2, step_size=25)
        clusters = dataset.embedded_clusters
        n_eval=1
    elif mode=='SR-M':
        clusters = dataset.clusters

    succ_rates=[]

    for e in range(n_eval):
        for _, data in clusters:
            for sample in data:
                kmer, target = sample
                if 'M' in mode:
                    mut = dataset.random_mutations(kmer, mutation_rate=0.1)
                    kmer = dataset.frequency_tensor(mut,kmer_length=25, subkmer_length=2, step_size=25)
                kmer = kmer.unsqueeze(0)
                counts = hybrid_model.inference(kmer, n_counts)
                succ = 0
                for i in [bin(i)[2:].zfill(dataset.datbase_size) for i in target.nonzero().squeeze(1)]:
                    succ+=counts.get(i,0)
                succ = int(succ*n_counts / 100)
                succ_rates.append(succ)

    return statistics.mean(succ_rates)
