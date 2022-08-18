import sys
import random
import tqdm
import math

import numpy as np
import pandas as pd
from apex import amp
from model import LightXML

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch

from torch.utils.data import DataLoader
from dataset import MDataset
from log import Logger

def load_group(dataset, group_tree=0):
    return np.load(f'./data/{dataset}/label_group{group_tree}.npy', allow_pickle=True)

def train(model, df, label_map, curr_period):
    tokenizer = model.get_tokenizer()

    if args.group_y_group > 0:
        group_y = load_group(args.dataset, args.group_y_group)
        train_d = MDataset(df, f'train{str(curr_period)}', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, 
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)

        train_d.tokenizer = model.get_fast_tokenizer()
        test_d.tokenizer = model.get_fast_tokenizer()

        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=False)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)
        if args.valid:
            valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                               candidates_num=args.group_y_candidate_num)
            validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0, 
                                     shuffle=False)
    else:
        train_d = MDataset(df, f'train{str(curr_period)}', tokenizer, label_map, args.max_len)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)
        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,
                                 shuffle=False)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,
                                shuffle=False)

    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)#, eps=1e-8)
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    max_only_p5 = 0
    for epoch in range(0, args.epoch+5):
        train_loss, running_loss = model.one_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        if args.valid:
            ev_result = model.one_epoch(epoch, validloader, optimizer, mode='eval')
        else:
            ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')

        g_p1, g_p3, g_p5, p1, p3, p5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
        if args.group_y_group > 0:
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        # save loss for each batch
        running_loss = np.array(running_loss)
        np.save(f'loss/{get_exp_name(curr_period)}-{str(epoch)}.npy', running_loss)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{get_exp_name(curr_period)}.bin')

        if epoch >= args.epoch + 5 and max_only_p5 != p5:
            break


def get_exp_name(curr_period):
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.group_y_group > 0:
        name.append('t'+str(args.group_y_group))

    name.append('p'+str(curr_period))

    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def createDataCSV(dataset, nb_periods):
    labels = []
    texts = []
    dataType = []

    fext = '_raw_texts.txt'

    # count size of training set
    with open(f'./data/{dataset}/train{fext}', 'r') as f:
        size = math.ceil(len(f.read().splitlines()) * 1.0 / nb_periods)
   
    with open(f'./data/{dataset}/train{fext}') as f:
        cnt = 0
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))

            period = cnt // size
            dataType.append(f'train{str(period)}')
            cnt += 1

    with open(f'./data/{dataset}/test{fext}') as f:
        for i in tqdm.tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')


    with open(f'./data/{dataset}/train_labels.txt') as f:
        for i in tqdm.tqdm(f):
            labels.append(i.replace('\n', ''))

    with open(f'./data/{dataset}/test_labels.txt') as f:
         for i in tqdm.tqdm(f):
            labels.append(i.replace('\n', ''))

    assert len(texts) == len(labels) == len(dataType)

    df_row = {'text': texts, 'label': labels, 'dataType': dataType}
    df = pd.DataFrame(df_row)

    return df



import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

parser.add_argument('--label_file', type=str, required=True)
parser.add_argument('--nb_periods', type=int, default=5)

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)

    curr_period = 0

    print(get_exp_name(curr_period))

    LOG = Logger('log_'+get_exp_name(curr_period))
    
    print(f'load {args.dataset} dataset...')
    ### org: df, label_map = createDataCSV(args.dataset)
    df = createDataCSV(args.dataset, args.nb_periods)

    ### use original label map (label_file) instead of label map created by createDataCSV
    labels = [l.split("\t")[0] for l in open(args.label_file).read().splitlines()[1:]]
    label_map = {}
    for i, label in enumerate(labels):
        label_map[label] = i

    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train0'],
                                              test_size=4000 // args.nb_periods,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train0"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    if args.group_y_group > 0:
        group_y = load_group(args.dataset, args.group_y_group)
        _group_y = []
        for idx, labels in enumerate(group_y):
            _group_y.append([])
            for label in labels:
                _group_y[-1].append(label_map[label])
            _group_y[-1] = np.array(_group_y[-1])
        group_y = np.array(_group_y)

        model = LightXML(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                          update_count=args.update_count,
                          use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                          candidates_topk=args.group_y_candidate_topk,
                          hidden_dim=args.hidden_dim)
    else:
        model = LightXML(n_labels=len(label_map), bert=args.bert,
                         update_count=args.update_count,
                         use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)

    if args.eval_model and args.group_y_group > 0:  # test the last model and exit
        print(f'load models/model-{get_exp_name(args.nb_periods-1)}.bin')
        testloader = DataLoader(MDataset(df, 'test', model.get_fast_tokenizer(), label_map, args.max_len, 
                                         candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)

        group_y = load_group(args.dataset, args.group_y_group)
        validloader = DataLoader(MDataset(df, 'valid', model.get_fast_tokenizer(), label_map, args.max_len, group_y=group_y,
                                          candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)
        model.load_state_dict(torch.load(f'models/model-{get_exp_name(args.nb_periods-1)}.bin'))
        model = model.cuda()

        print(len(df[df.dataType == 'test']))
        model.one_epoch(0, validloader, None, mode='eval')

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{get_exp_name(args.nb_periods-1)}-labels.npy', np.array(pred_labels))
        np.save(f'results/{get_exp_name(args.nb_periods-1)}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    train(model, df, label_map, curr_period)

    curr_period += 1

    while curr_period < args.nb_periods:
        # repeate the process of train

        if args.valid:
            train_df, valid_df = train_test_split(df[df['dataType'] == f'train{str(curr_period)}'],
                                                test_size=4000 // args.nb_periods,
                                                random_state=1240)
            df.iloc[valid_df.index.values, 2] = 'valid'
            print('valid size', len(df[df['dataType'] == 'valid']))


        print(f'load {args.dataset} dataset with '
            f'{len(df[df.dataType =="train{str(curr_period)}"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

        if args.group_y_group > 0:
            group_y = load_group(args.dataset, args.group_y_group)
            _group_y = []
            for idx, labels in enumerate(group_y):
                _group_y.append([])
                for label in labels:
                    _group_y[-1].append(label_map[label])
                _group_y[-1] = np.array(_group_y[-1])
            group_y = np.array(_group_y)

            model = LightXML(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                            update_count=args.update_count,
                            use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                            candidates_topk=args.group_y_candidate_topk,
                            hidden_dim=args.hidden_dim)
        else:
            model = LightXML(n_labels=len(label_map), bert=args.bert,
                            update_count=args.update_count,
                            use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)

        print(f'load models/model-{get_exp_name(curr_period-1)}.bin')
        model.load_state_dict(torch.load(f'models/model-{get_exp_name(curr_period-1)}.bin'))

        train(model, df, label_map, curr_period)

        curr_period += 1
