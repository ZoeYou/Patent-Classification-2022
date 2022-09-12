import torch, csv, sys, pickle, os
from pathlib import Path
torch.cuda.empty_cache()
csv.field_size_limit(sys.maxsize)

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'LightXML', 'src'))
from dataset import MDataset, createDataCSV
from model import LightXML

sys.path.append(os.path.join(os.path.dirname(__file__), 'AttentionXML'))
from deepxml.evaluation import *


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=False, default='../data/INPI/new_extraction/output/inpi_new_final.csv')
parser.add_argument('--pred_level', type=int, required=False, default=6)

parser.add_argument('--lightxml', type=str, required=False, action='append')
parser.add_argument('--attentionxml', type=str, required=False, action='append')

args = parser.parse_args()

def label_encoding(line, label_list, label_map):
    labels = line.split(",")
    labels = [l for l in labels if l in label_list]

    idx = [str(label_map[l]) for l in labels if l in label_list]
    if idx: 
        return " ".join(idx)
    
def get_datatype(dataname, sec_names=['title','abs','claims','desc']):
    splits = dataname.split('_')
    ret = '_'.join([sec for sec in splits if sec in sec_names])
    return ret

if __name__ == '__main__':

    labels = [l.split("\t")[0] for l in open(f"../data/ipc-sections/20210101/labels_group_id_{str(args.pred_level)}.tsv", "r").read().splitlines()[1:]]
    label_dict = dict(zip(labels, range(len(labels))))
       
    predicts = []

    classifiers = {'lightxml': [], 'attentionxml': []}
    if args.lightxml: classifiers['lightxml'] = args.lightxml
    if args.attentionxml: classifiers['attentionxml'] = args.attentionxml

    # load test data of INPI with all sections (abstract, description, claims)
    df = pd.read_csv(args.dataset_path, dtype=str, engine="python")
    df = df[df['date'].apply(lambda x: int(x[:4]) >= 2020)].dropna()
    df['dataType'] = ['test'] * len(df)

    for sec in ['title', 'abs', 'claims', 'desc']:
        df[sec] = df[sec].apply(str)

    list_sections = [['title', 'abs'], ['title', 'desc'], ['claims']]   # TODO

    for secs in list_sections:
        df["_".join(secs)] = df[secs].apply(' /SEP/ '.join, axis=1)
    df = df[['title_abs', 'title_desc', 'claims', f'IPC{str(args.pred_level)}', 'dataType']]
     
    df['label'] = df[f'IPC{str(args.pred_level)}'].apply(lambda line: label_encoding(line, labels, label_dict))
    df = df.dropna(subset = ['label'])


    ######################### LightXML ###############################
    berts = ['camembert', 'xlm-roberta', 'mbert']
    if not predicts:
        cnt = 0
        for sections in ['title_abs', 'title_desc', 'claims'][:len(classifiers['lightxml'])]:
            dataset_name = f"INPI_{sections}_2020_{str(args.pred_level)}"

            df['text'] = df[sections]
            df_test = df[['text', 'label', 'dataType']]

            for index in range(len(berts)):
                model_name = [dataset_name, '' if berts[index] == 'bert-base' else berts[index]]
                model_name = '_'.join([i for i in model_name if i != ''])
                try:
                    with open(f'LightXML/predictions/INPI_IPC{str(args.pred_level)}_{sections}_{model_name}_ensemble.pkl', 'rb') as in_f:
                        single_predictions = pickle.load(in_f)

                except FileNotFoundError:
                    model = LightXML(n_labels=len(label_dict), bert=berts[index])

                    print(f'LightXML/models/model-{model_name}.bin')
            
                    model.load_state_dict(torch.load(f'models/model-{model_name}.bin'))

                    tokenizer = model.get_tokenizer()

                    test_d = MDataset(df_test, 'test', tokenizer, label_dict, 512)

                    if args.pred_level == 4:
                        testloader = DataLoader(test_d, batch_size=16, num_workers=0,
                                        shuffle=False)
                    elif args.pred_level > 4:
                        testloader = DataLoader(test_d, batch_size=4, num_workers=0,
                                        shuffle=False)    

                    torch.cuda.empty_cache()
                    model.cuda()
                    single_predictions = torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0])
                
                    # save predictions
                    with open(f'LightXML/predictions/INPI_IPC{str(args.pred_level)}_{sections}_{model_name}_ensemble.pkl', 'wb') as out_f:
                        pickle.dump(single_predictions, out_f)

                predicts.append(single_predictions) 
                print(single_predictions.shape) # torch.Size([25943, 7523])
            cnt += 1

    ######################### AttentionXML ###############################
    model_names = classifiers['attentionxml']
    trees = 3

    prefix = [f'AttentionXML/results/FastAttentionXML-{mn}' for mn in model_names]

    labels_att, scores_att = [], []
    attentionxml_predictions = torch.zeros(len(df), len(labels))
    for p in prefix: 
        for i in range(trees):
            labels_att.append(np.load(F'{p}-Tree-{i}-labels.npy', allow_pickle=True))   # 3 * 25943 * 100
            scores_att.append(np.load(F'{p}-Tree-{i}-scores.npy', allow_pickle=True))

    for i in tqdm(range(len(labels_att[0]))):   # for each data
        s = defaultdict(float)
        for j in range(len(labels_att[0][i])):  # for each attentionxml classifier
            for k in range(len(labels_att)):
                s[labels_att[k][i][j]] += scores_att[k][i][j]  

        # pad to complete vector (to adapt to LightXML predictions)
        for s_k, s_v in s.items():
            attentionxml_predictions[i, int(s_k)] = s_v 
    predicts.append(attentionxml_predictions)
    ######################################################################

    total = len(df)
    acc1 = [0 for i in range(len(predicts) + 1)]       # shape (1, nb_ensembles * nb_of_models_in_each_ensemble + 1)
    acc3 = [0 for i in range(len(predicts) + 1)]
    acc5 = [0 for i in range(len(predicts) + 1)]

    # save prediction results for error analysis
    preds = []

    for index, true_labels in enumerate(df.label.values):
        try:
            true_labels = set([i for i in true_labels.split()])
        except AttributeError:
            continue

        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(predicts))] 

        logits.append(sum(logits))  # not voting, it's calculating the sum of logits of all models
        logits = [(-i).argsort()[:100].cpu().numpy() for i in logits]

        for i, logit in enumerate(logits):
            logit_code = [] 
             
            for k in range(len(logit)):
                logit_code.append(str(logit[k]))
    
            acc1[i] += len(set([logit_code[0]]) & true_labels)
            acc3[i] += len(set(logit_code[:3]) & true_labels)
            acc5[i] += len(set(logit_code[:5]) & true_labels)

        preds.append(logit[0])

    # create directory results if dir does not exist
    if not Path('./results').exists():
        try:
            Path('./results').mkdir(parents=True)
            print(f"Created output directory results")
        except FileExistsError:
            print(f"Directory 'results' already exists!")

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble.out', 'w') as f:
        for i, name in enumerate(berts * len(classifiers['lightxml']) + ['attentionxml'] + ['all']):
            p1 = acc1[i] / total
            p3 = acc3[i] / total / 3
            p5 = acc5[i] / total / 5

            print(f'{name} {p1} {p3} {p5}', file=f)
            print(f'{name} {p1} {p3} {p5}')

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble_pred.txt', 'w') as out_f:
        lines = [str(l) for l in preds]
        out_f.write("\n".join(lines))
