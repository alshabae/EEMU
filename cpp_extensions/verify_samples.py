import sys
sys.path.append("build/")
from Sampler import CompactAdjacency
import numpy as np
from scipy.special import softmax
import torch
import random
from tqdm import tqdm

print("Imported")

probs = []
for _ in range(1):
    num_neighbors = random.randint(20, 500)
    p = softmax(np.random.random(num_neighbors).tolist())
    probs.append(p)

nodes = np.arange(1)
comp_adj = CompactAdjacency(probs, 0)

# samples = np.concatenate([comp_adj.SampleNaive(nodes, 15).flatten() for _ in range(5000000)])
# unique_counts1 = np.unique(samples, return_counts=True)[1]

samples = np.concatenate([comp_adj.SampleOptimized(nodes, 15).flatten() for _ in range(5000000)])
unique_counts1 = np.unique(samples, return_counts=True)[1]

# start = timer()
# samples = np.concatenate([comp_adj.SampleOptimizedOMP(nodes, 15).flatten() for _ in range(5000000)])
# end = timer()
# print(f"C++ Optimized = {1000 * (end - start)}, Samples shape = {samples.shape}")
# unique_counts1 = np.unique(samples, return_counts=True)[1]

samples = []
for i in tqdm(range(5000000)):
    res = []
    for node_id in nodes:
        prob = probs[node_id]
        sampled = torch.multinomial(torch.tensor(prob), 15).numpy()
        res.append(sampled)
    
    samples.append(np.concatenate(res, axis=0).reshape(-1, 15).flatten())
samples = np.concatenate(samples)
unique_counts2 = np.unique(samples, return_counts=True)[1]

avg_freq_diff = (np.abs(unique_counts2 - unique_counts1) / unique_counts2).mean()
print(100 * avg_freq_diff)