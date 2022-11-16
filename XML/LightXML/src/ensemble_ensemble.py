import torch, csv, sys, pickle
torch.cuda.empty_cache()

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
csv.field_size_limit(sys.maxsize)

from model import LightXML

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, required=False, default="../../data/INPI/new_extraction/output/inpi_new_final.csv")
parser.add_argument('--pred_level', type=int, required=False, default=6)
parser.add_argument('--label_file', type=str, required=False, default='../../data/ipc-sections/20210101/labels_group_id_6.tsv')

parser.add_argument('--model1', type=str, required=True)    # title+abstract
parser.add_argument('--model2', type=str, required=False)   # title+description
parser.add_argument('--model3', type=str, required=False)   # claims

args = parser.parse_args()

def label_encoding(line, label_list, label_map, plus1 = False):
    labels = line.split(",")
    labels = [l for l in labels if l in label_list]

    if plus1:
        idx = [str(label_map[l]-1) for l in labels if l in label_list]
    else:
        idx = [str(label_map[l]) for l in labels if l in label_list]
    if idx: 
        return " ".join(idx)


        
if __name__ == '__main__':

    labels = [l.split("\t")[0] for l in open(f"../../data/ipc-sections/20210101/labels_group_id_{str(args.pred_level)}.tsv", "r").read().splitlines()[1:]]
    label_dict = dict(zip(labels, range(len(labels))))
    label_map = {}
    for i, label in enumerate(labels):
        label_map[str(i)] = i
      
    predicts = []

    classifiers = [args.model1]
    if args.model2:
        classifiers.append(args.model2)
    if args.model3:
        classifiers.append(args.model3)

    # load test data of INPI with all sections (abstract, description, claims)
    df = pd.read_csv(args.dataset_path, dtype=str, engine="python")
    df = df[df['date'].apply(lambda x: int(x[:4]) >= 2020)].dropna()
    df['dataType'] = ['test'] * len(df)

    for sec in ['title', 'abs', 'claims', 'desc']:
        df[sec] = df[sec].apply(str)

    list_sections = [['title', 'abs'], ['title', 'desc'], ['claims']][:len(classifiers)]
    for secs in list_sections:
        df["_".join(secs)] = df[secs].apply(' /SEP/ '.join, axis=1)
    df = df[['title_abs', 'title_desc', 'claims', f'IPC{str(args.pred_level)}', 'dataType']]
     
    # fill in na with empty string
    df['label'] = df[f'IPC{str(args.pred_level)}'].apply(lambda line: label_encoding(line, labels, label_dict, plus1 = False))
    df = df.dropna(subset = ['label'])

    berts = ['camembert', 'xlm-roberta', 'mbert']
    cnt = 0
    
    for sections in ['title_abs', 'title_desc', 'claims'][:len(classifiers)]:
        dataset_name = f"INPI_{sections}_2020_{str(args.pred_level)}"

        df['text'] = df[sections]
        df_test = df[['text', 'label', 'dataType']]

        for index in range(len(berts)):
            model_name = [dataset_name, '' if berts[index] == 'bert-base' else berts[index]]
            model_name = '_'.join([i for i in model_name if i != ''])
            try:
                with open(f'predictions/INPI_IPC{str(args.pred_level)}_{sections}_{model_name}_ensemble.pkl', 'rb') as in_f:
                    single_predictions = pickle.load(in_f)

            except FileNotFoundError:
                model = LightXML(n_labels=len(label_dict), bert=berts[index])

                print(f'models/model-{model_name}.bin')
        
                model.load_state_dict(torch.load(f'models/model-{model_name}.bin'), strict=False)

                tokenizer = model.get_tokenizer()

                test_d = MDataset(df_test, 'test', tokenizer, label_map, 512)

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
                with open(f'predictions/INPI_IPC{str(args.pred_level)}_{sections}_{model_name}_ensemble.pkl', 'wb') as out_f:
                    pickle.dump(single_predictions, out_f)

            predicts.append(single_predictions)

        cnt += 1

    total = len(df)
    acc1 = [0 for i in range(len(predicts) + 1)]       # shape (1, 3 * nb_of_models + 1)
    acc3 = [0 for i in range(len(predicts) + 1)]
    acc5 = [0 for i in range(len(predicts) + 1)]

    # save prediction results for error analysis
    preds = []
    nb_trueLabels = 0
    for index, true_labels in enumerate(df.label.values):
        try:
            true_labels = set([i for i in true_labels.split()])
            nb_trueLabels += len(true_labels)
        except AttributeError:
            continue

        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(predicts))] 

        logits.append(sum(logits))  # not voting, it's calculating the sum of logits of all models
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        for i, logit in enumerate(logits):
            logit_code = [] 
             
            for k in range(len(logit)):
                logit_code.append(str(logit[k]))
    
            acc1[i] += len(set([logit_code[0]]) & true_labels)
            acc3[i] += len(set(logit_code[:3]) & true_labels)
            acc5[i] += len(set(logit_code[:5]) & true_labels)

        preds.append(logit[0])

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble.out', 'w') as f:
        for i, name in enumerate(berts * len(classifiers) + ['all']):
            p1 = acc1[i] / total
            p3 = acc3[i] / total / 3
            p5 = acc5[i] / total / 5

            r1 = acc1 / nb_trueLabels
            r3 = acc3 / nb_trueLabels
            r5 = acc5 / nb_trueLabels

            print(f'{name} P@1:{p1}, P@3:{p3}, P@5:{p5}, R@1:{r1}, R@3:{r3}, R@5:{r5}', file=f)
            print(f'{name} P@1:{p1}, P@3:{p3}, P@5:{p5}, R@1:{r1}, R@3:{r3}, R@5:{r5}')

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble_pred.txt', 'w') as out_f:
        lines = [str(l) for l in preds]
        out_f.write("\n".join(lines))
