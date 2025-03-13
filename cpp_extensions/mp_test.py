import sys
sys.path.append("build/")
from Sampler import CompactAdjacency
import numpy as np
from scipy.special import softmax
from timeit import default_timer as timer
import multiprocessing as mp
import random
import torch

def run(rank):
    probs = []
    for _ in range(10000):
        num_neighbors = random.randint(20, 500)
        p = softmax(np.random.random(num_neighbors).tolist())
        probs.append(p)

    nodes = np.random.randint(low=0, high=10000, size=(50000,))
    comp_adj = CompactAdjacency(probs, 0)

    start = timer()
    samples = comp_adj.SampleOptimizedOMP(nodes, 15)
    end = timer()
    print(f"RANK {rank} : C++ Sampling Optimized (OMP) = {1000 * (end - start)}, Samples shape = {samples.shape}")

    # start = timer()
    # res = []
    # for node_id in nodes:
    #     prob = probs[node_id]
    #     sampled = torch.multinomial(torch.tensor(prob), 15).numpy()
    #     res.append(sampled)

    # res = np.concatenate(res, axis=0).reshape(-1, 15)
    # end = timer()
    # print(f"RANK {rank} : Pytorch / Python Loop = {1000 * (end - start)}, Samples shape = {res.shape}")


if __name__ == '__main__':
    processes = []
    num_processes = 32
    for rank in range(num_processes):
        p = mp.Process(target=run, args=(rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()