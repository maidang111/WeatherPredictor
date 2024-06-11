import argparse
import pandas as pd
import numpy as np
import os
import re

def k_NN(predictions):
    count = {}
    freq = [[] for i in range(len(predictions) + 1)]

    for n in predictions:
        count[n] = 1 + count.get(n, 0)
    for n, c in count.items():
        freq[c].append(n)

    res = []
    for i in range(len(freq) -1, 0 ,-1):
        for n in freq[i]:
            res.append(n)
        if len(res) > 1:
            break
    if len(res) > 1:
        n = set(res)
        for i in predictions:
            if i in n:
                return i

    return res

def main():
    print(k_NN([1,1,3,2,2]))
    return 1


if __name__ == "__main__":
    main()
