import sys
sys.path.append("build/")
from CustomSampler import Sampler
import numpy as np
from scipy.special import softmax
import torch
from timeit import default_timer as timer
import random

print("Imported")

probs = []
for _ in range(10000):
    num_neighbors = random.randint(20, 500)
    p = softmax(np.random.random(num_neighbors).tolist())
    probs.append(p)

nodes = np.random.randint(low=0, high=10000, size=(50000,))
sampler = Sampler(probs, 0)

start = timer()
samples = sampler.SampleNaive(nodes, 15)
end = timer()
print(f"C++ Naive Sampling = {1000 * (end - start)}, Samples shape = {samples.shape}")

start = timer()
samples = sampler.SampleOptimized(nodes, 15)
end = timer()
print(f"C++ Optimized = {1000 * (end - start)}, Samples shape = {samples.shape}")

start = timer()
samples = sampler.SampleOptimizedOMP(nodes, 15)
end = timer()
print(f"C++ Optimized OMP = {1000 * (end - start)}, Samples shape = {samples.shape}")



start = timer()
res = []
for node_id in nodes:
    prob = probs[node_id]
    sampled = torch.multinomial(torch.tensor(prob), 15).numpy()
    res.append(sampled)

res = np.concatenate(res, axis=0).reshape(-1, 15)
end = timer()
print(f"Pytorch / Python Loop = {1000 * (end - start)}, Samples shape = {samples.shape}")