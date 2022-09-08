#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on 2020/9/6
@author zy

"""

import click
import numpy as np
from collections import defaultdict
from tqdm import tqdm


@click.command()
@click.option('-p', '--prefix', help='Prefix of results.', multiple=True)
@click.option('-t', '--trees', type=click.INT, help='The number of results using for ensemble for each classifier.')
def main(prefix, trees):
    ensemble_name = []
    labels, scores = [], []
    for p in prefix:
        ensemble_name.extend(p.split('_')[1:])    
        for i in range(trees):
            labels.append(np.load(F'{p}-Tree-{i}-labels.npy', allow_pickle=True))
            scores.append(np.load(F'{p}-Tree-{i}-scores.npy', allow_pickle=True))

    ensemble_labels, ensemble_scores = [], []
    for i in tqdm(range(len(labels[0]))):
        s = defaultdict(float)
        for j in range(len(labels[0][i])):
            for k in range(len(labels)): 
                s[labels[k][i][j]] += scores[k][i][j]  
        s = sorted(s.items(), key=lambda x: x[1], reverse=True)
        ensemble_labels.append([x[0] for x in s[:len(labels[0][i])]])
        ensemble_scores.append([x[1] for x in s[:len(labels[0][i])]])

    np.save(F'{"_".join(p.split("_")[:1] + ensemble_name)}-Ensemble-labels', np.asarray(ensemble_labels))
    np.save(F'{"_".join(p.split("_")[:1] + ensemble_name)}-Ensemble-scores', np.asarray(ensemble_scores))

if __name__ == '__main__':
    main()
