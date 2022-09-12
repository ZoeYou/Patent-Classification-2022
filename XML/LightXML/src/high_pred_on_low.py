"""
Created on 2020/9/7
@author zy

This script is created for testing ( LightXML trained on deeper level of IPC ) on ( patent data with lower level of IPC ). 
It is proved that model trained on deeper levels of IPC sometimes knows better predicting on shallower levels.
"""


import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from tqdm import tqdm
from collections import defaultdict

from model import LightXML

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_train', type=str, required=False, default='INPI_title_desc_2020_6')     # high level IPC model
parser.add_argument('--dataset_test', type=str, required=False, default='INPI_title_desc_2020_4')   # low level IPC data

parser.add_argument('--model_level', type=int, required=False, default=6)
parser.add_argument('--dataset_level', type=int, required=False, default=4)

parser.add_argument('--debugging', action='store_true')
parser.add_argument('--mode', required=False, default='sum', choices=['sum', 'mean', 'set'], help='Function used to calculate ensemble score.')
#parser.add_argument('--train_label_file', type=str, required=False, default='../../data/ipc-sections/20210101/labels_group_id_8.tsv')
#parser.add_argument('--test_label_file', type=str, required=False, default='../../data/ipc-sections/20210101/labels_group_id_6.tsv')

args = parser.parse_args()

def inverse_label_map(label_map):
    inv_map = {}
    for k,v in label_map.items():
        inv_map[v] = k
    return inv_map


if __name__ == '__main__':
    # load label dict
    labels_train = [l.split("\t")[0] for l in open(f'../../data/ipc-sections/20210101/labels_group_id_{args.model_level}.tsv', "r").read().splitlines()[1:]]
    label_dict_train = dict(zip(range(len(labels_train)), labels_train))

    labels_test = [l.split("\t")[0] for l in open(f'../../data/ipc-sections/20210101/labels_group_id_{args.dataset_level}.tsv', "r").read().splitlines()[1:]]
    label_dict_test = dict(zip(range(len(labels_test)), labels_test))

    # levels of train and test
    train_level = args.model_level 
    test_level = args.dataset_level

    # use original label map (label_file) instead of label map created by createDataCSV
    label_map_train = {}
    for i, label in enumerate(labels_train):
        label_map_train[str(i)] = i  

    """
    label_map_test = {}
    for i, label in enumerate(labels_test):
        label_map_test[str(i)] = i  
    """
 

    df_test, label_map_test = createDataCSV(args.dataset_test)

    print(f'load testing dataset {args.dataset_test} with '
          f'{len(df_test[df_test.dataType =="train"])} train {len(df_test[df_test.dataType =="test"])} test with {len(label_map_test)} labels done')
  
    inv_label_map_train = inverse_label_map(label_map_train)

    predicts = []
    if "-fr-" in args.model_train or "INPI" in args.model_train:
        berts = ['camembert', 'xlm-roberta', 'mbert']
    else:
        berts = ['bert-base', 'roberta', 'xlnet'] 
    
    for index in range(len(berts)):
        model_name = [args.model_train, '' if berts[index] == 'bert-base' else berts[index]]
        model_name = '_'.join([i for i in model_name if i != ''])

        model = LightXML(n_labels=len(label_map_train), bert=berts[index])

        print(f'models/model-{model_name}.bin')
        model.load_state_dict(torch.load(f'models/model-{model_name}.bin'))

        tokenizer = model.get_tokenizer()
        
        test_d = MDataset(df_test, 'test', tokenizer, label_map_train, 512)
        testloader = DataLoader(test_d, batch_size=8, num_workers=0,
                                shuffle=False)

        model.cuda()
        predicts.append(torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0]))

    df = df_test[df_test.dataType == 'test']

    total = len(df)
    acc1 = [0 for i in range(len(berts) + 1)]       # shape (1, 4)
    acc3 = [0 for i in range(len(berts) + 1)]
    acc5 = [0 for i in range(len(berts) + 1)]


    # save prediction results for error analysis
    preds = []

    for index, true_labels in tqdm(enumerate(df.label.values)):
        true_labels = set([label_dict_test[int(i)] for i in true_labels.split()])

        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(berts))]
        logits.append(sum(logits))
  
        scores = [i.sort(descending=True)[0][:1000].cpu().numpy() for i in logits]
        logits = [(-i).argsort()[:1000].cpu().numpy() for i in logits]   # shape (4, 1000)

        for i, (logit, score) in enumerate(zip(logits, scores)):
            logit_code = defaultdict(list)

            for k,s in zip(logit, score):
                # inverse logits code by inv_label_map
                if test_level == 4:
                    logit_code[label_dict_train[int(inv_label_map_train[k])][:4]].append(s)
                elif test_level == 6:
                    logit_code[label_dict_train[int(inv_label_map_train[k])].split('/')[0]].append(s)

            if args.mode == 'sum':
                logit_code = {k:sum(v) for k,v in logit_code.items()}
                logit_code = sorted(logit_code, key=logit_code.get, reverse=True)
            elif args.mode == 'mean':
                logit_code = {k:sum(v) for k,v in logit_code.items()}
                logit_code = sorted(logit_code, key=logit_code.get, reverse=True)
            else:   # mode == 'set'
                logit_code = list(logit_code.keys())
           
            if args.debugging: 
                print('true labels:', true_labels)
                print('predictions:', logit_code[:10])
                print('==========================')
 
            acc1[i] += len(set([logit_code[0]]) & true_labels)
            acc3[i] += len(set(logit_code[:3]) & true_labels)
            acc5[i] += len(set(logit_code[:5]) & true_labels)

        preds.append(logit[0])

    with open(f'./results/{args.model_train}_on_{args.dataset_test}.out', 'w') as f:
        for i, name in enumerate(berts + ['all']):
            p1 = acc1[i] / total
            p3 = acc3[i] / total / 3
            p5 = acc5[i] / total / 5

            print(f'{name} {p1} {p3} {p5}', file=f)
            print(f'{name} {p1} {p3} {p5}')


    with open(f'./results/{args.model_train}_on_{args.dataset_test}_pred.txt', 'w') as out_f:
        lines = [str(l) for l in preds]
        out_f.write("\n".join(lines))
