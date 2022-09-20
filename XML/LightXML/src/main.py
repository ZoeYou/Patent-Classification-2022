import sys, os
import random
import numpy as np
import apex
from apex import amp
from model import LightXML

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch
import torch.nn as nn

torch.cuda.empty_cache()

from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from log import Logger

def load_group(dataset, group_tree=0):
    return np.load(f'./data/{dataset}/label_group{group_tree}.npy', allow_pickle=True)

def train(model, df, label_map, data_parallel):
    tokenizer = model.get_tokenizer()

    if args.group_y_group > 0:
        group_y = load_group(args.dataset, args.group_y_group)
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, 
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)

        train_d.tokenizer = model.get_fast_tokenizer()
        test_d.tokenizer = model.get_fast_tokenizer()

        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=True)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)
        if args.valid:
            valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                               candidates_num=args.group_y_candidate_num)
            validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0, 
                                     shuffle=False)
    else:
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)

        if data_parallel:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_d, num_replicas=torch.cuda.device_count())
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_d, num_replicas=torch.cuda.device_count())

            trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,
                                 pin_memory=True,
                                 sampler=train_sampler
                                 )
            testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,
                                pin_memory=True,
                                sampler=test_sampler
                                )

        else:
            trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,
                                 shuffle=True)
            testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,
                                shuffle=False)

    model.cuda()

    ###if torch.cuda.device_count() > 1:
    ###device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
    #else:
    ###device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    ###print('device:', torch.cuda.current_device())
    ###model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)#, eps=1e-8)
        
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.mode_amp)
    if data_parallel:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        """
        print('__Python VERSION:', sys.version)
        print('__pyTorch VERSION:', torch.__version__)
        print('__CUDA VERSION')
        from subprocess import call
        # call(["nvcc", "--version"]) does not work
        #! nvcc --version
        print('__CUDNN VERSION:', torch.backends.cudnn.version())
        print('__Number CUDA Devices:', torch.cuda.device_count())
        print('__Devices')
        call(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
        print('Active CUDA Device: GPU', torch.cuda.current_device())
        print ('Available devices ', list(range(0,torch.cuda.device_count())))
        print ('Current cuda device ', torch.cuda.current_device())
        """
        ###model = MyDataParallel(model, device_ids = list(range(0,torch.cuda.device_count())))
        model = MyDataParallel(model)
        print('check if it is doing data parallel: ', isinstance(model, nn.DataParallel))

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
        np.save(f'loss/{get_exp_name()}-{str(epoch)}.npy', running_loss)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{get_exp_name()}.bin')

        if epoch >= args.epoch + 5 and max_only_p5 != p5:
            break

def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.group_y_group > 0:
        name.append('t'+str(args.group_y_group))
    if args.checkpoint:
        name = [args.checkpoint, '+'] + name
    return '_'.join([i for i in name if i != ''])

def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


#class MyDataParallel(nn.parallel.DistributedDataParallel):
class MyDataParallel(apex.parallel.DistributedDataParallel):
    def __getattr__(self, name):
        if name == 'module':
            return super().__getattr__('module')
        else:
            return getattr(self.module, name)


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
parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)
parser.add_argument('--eval_model', action='store_true')

# added by zy
parser.add_argument('--checkpoint', type=str, required=False, default=None, help='Name of the previous saved checkpoint. (will continue to train on this checkpoint)')
parser.add_argument('--label_file', type=str, required=False, default='../../data/ipc-sections/20210101/labels_group_id_6.tsv')
parser.add_argument('--data_parallel', action='store_true', help='Whether to train the model with dataparallel. If yes then use two gpus.')
parser.add_argument('--mode_amp', type=str, required=False, default='O1', help='Mode of mixed precision.')
parser.add_argument('--local_rank', type=int, default=-1, help="DDP parameter, do not modify")

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('log_'+get_exp_name())

    ### orig: df, label_map = createDataCSV(args.dataset) 
    print(f'load {args.dataset} dataset...')
    df, _ = createDataCSV(args.dataset)

    ### use original label map (label_file) instead of label map created by createDataCSV
    labels = [l.split("\t")[0] for l in open(args.label_file).read().splitlines()[1:]]
    label_map = {}
    for i, label in enumerate(labels):
        label_map[str(i)] = i

    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

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

    if args.checkpoint:
        model.load_state_dict(torch.load(f'models/model-{args.checkpoint}_{args.bert}.bin'))
    
    elif args.eval_model and args.group_y_group > 0:
        print(f'load models/model-{get_exp_name()}.bin')
        testloader = DataLoader(MDataset(df, 'test', model.get_fast_tokenizer(), label_map, args.max_len, 
                                         candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)

        group_y = load_group(args.dataset, args.group_y_group)
        validloader = DataLoader(MDataset(df, 'valid', model.get_fast_tokenizer(), label_map, args.max_len, group_y=group_y,
                                          candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)
        model.load_state_dict(torch.load(f'models/model-{get_exp_name()}.bin'), strict=False)
        model = model.cuda()

        print(len(df[df.dataType == 'test']))
        model.one_epoch(0, validloader, None, mode='eval')

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{get_exp_name()}-labels.npy', np.array(pred_labels))
        np.save(f'results/{get_exp_name()}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    if args.data_parallel:
        n_cudas = ','.join([str(e) for e in list(range(torch.cuda.device_count()))])
        os.environ["CUDA_VISIBLE_DEVICES"] = n_cudas
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        torch.distributed.init_process_group(backend="nccl")

    train(model, df, label_map, data_parallel = args.data_parallel)
