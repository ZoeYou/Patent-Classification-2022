import torch, csv, sys, pickle, os, argparse
from pathlib import Path
# torch.cuda.empty_cache()
csv.field_size_limit(sys.maxsize)

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
from logzero import logger

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", default='../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input file of INPI-CLS data.")
parser.add_argument('--pred_level', type=int, required=False, default=6)
# parser.add_argument('--weighted_average', action='store_true', help='Whether to use Precision@1 to ensemble the weighted average of single classifiers.')
parser.add_argument('--lightxml', type=str, required=False, action='append')
parser.add_argument('--attentionxml', type=str, required=False, action='append')
parser.add_argument('--lightxml_highOnLow', type=str, required=False, action='append')  # TODO

args = parser.parse_args()

def label_encoding(line, label_list, label_map):
    labels = line.split(",")
    labels = [l for l in labels if l in label_list]
    idx = [str(label_map[l]) for l in labels if l in label_list]
    if idx: 
        return " ".join(idx)

def uniq(lst):
    last = object()
    for item in lst:
        if item == last: continue
        yield item
        last = item

def sort_and_deduplicate(l):
    return list(uniq(sorted(l, key=l.index)))

def get_datatype(datanames, sec_names=['title','abs','claims','desc']):
    dict_sec = defaultdict(lambda: None)
    dict_sec["ABSTR"] = "abs"
    dict_sec["abstract"] = "abs"
    dict_sec["TITLE"] = "title"
    dict_sec["CLAIM1"] = "claims"
    dict_sec["DESCR"] = "desc"
    ret = []
    for name in datanames:
        splits = name.split('_')
        ret.append([sec for sec in splits if (sec in sec_names) or (dict_sec[sec] in sec_names)])
    return sort_and_deduplicate(ret)

def softmax(vector):
    e = np.exp(vector)
    return e / e.sum()


if __name__ == '__main__':
    labels = [l.split("\t")[0] for l in open(f"../data/ipc-sections/20220101/labels_group_id_{str(args.pred_level)}.tsv", "r").read().splitlines()[1:]]
    label_dict = dict(zip(labels, range(len(labels))))
    label_decode_dict = dict(zip(range(len(labels)), labels))
    label_map = {}
    for i, label in enumerate(labels):
        label_map[str(i)] = i

    classifiers = {'lightxml': [], 'attentionxml': [], 'lightxml_highOnLow':[]}
    if args.lightxml: classifiers['lightxml'] = args.lightxml
    if args.attentionxml: classifiers['attentionxml'] = args.attentionxml
    if args.lightxml_highOnLow: classifiers['lightxml_highOnLow'] = args.lightxml_highOnLow

    # load test data of INPI with all sections (abstract, description, claims)
    df = pd.read_csv(args.in_file, dtype=str, engine="python")
    df = df[df['date'].apply(lambda x: int(x[:4]) >= 2020)].dropna()
    df['dataType'] = ['test'] * len(df)
    for sec in ['title', 'abs', 'claims', 'desc']:
        df[sec] = df[sec].apply(lambda x: str(x).replace("\n", " ").replace("  ", " "))

    list_sections = get_datatype(classifiers['lightxml'] + classifiers['attentionxml'] + classifiers['lightxml_highOnLow'])
    for secs in list_sections:
        df['_'.join(secs)] = df[secs].apply(' /SEP/ '.join, axis=1)
    df = df[['_'.join(l) for l in list_sections] + [f'IPC{str(args.pred_level)}', 'dataType']]
    df['label'] = df[f'IPC{str(args.pred_level)}'].apply(lambda line: label_encoding(line, labels, label_dict))
    df = df.dropna(subset = ['label'])[-100:]

    # create directory results if dir does not exist
    Path('./results/predictions').mkdir(parents=True, exist_ok=True)
    predicts = []
    ######################### LightXML ###############################
    berts = ['camembert', 'xlm-roberta', 'mbert']
    if classifiers['lightxml']:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'LightXML', 'src'))
        from dataset import MDataset
        from model_ensemble import LightXML 

        for lightxml_name in classifiers['lightxml']:
            df['text'] = df['_'.join(get_datatype([lightxml_name])[0])]
            df_test = df[['text', 'label', 'dataType']]
            for index in range(len(berts)):
                model_name = '_'.join([lightxml_name, berts[index]])
                logger.info(f'LightXML/models/model-{model_name}.bin')

                try:
                    single_predictions= torch.load(f'./results/predictions/lightxml_{model_name}_ensemble.pt')
                except FileNotFoundError:
                    model = LightXML(n_labels=len(label_dict), bert=berts[index])
                    if torch.cuda.is_available():
                        model.load_state_dict(torch.load(f'LightXML/models/model-{model_name}.bin'), strict=False)
                    else:
                        model.load_state_dict(torch.load(f'LightXML/models/model-{model_name}.bin', map_location='cpu'), strict=False)

                    tokenizer = model.get_tokenizer()

                    test_d = MDataset(df_test, 'test', tokenizer, label_map, 512)

                    if args.pred_level <= 4:
                        testloader = DataLoader(test_d, batch_size=16, num_workers=2,
                                        shuffle=False)
                    elif args.pred_level > 4:
                        testloader = DataLoader(test_d, batch_size=4, num_workers=1,
                                        shuffle=False)    

                    # torch.cuda.empty_cache()
                    # model.cuda() 
                    single_predictions = torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0]) # torch.Size([#data, #labels])
                
                    # save predictions
                    torch.save(single_predictions, f'./results/predictions/lightxml_{model_name}_ensemble.pt')
                predicts.append(single_predictions) 

            # for high on low prediction

    ######################### AttentionXML ###############################
    if classifiers['attentionxml']:
        sys.path.append(os.path.join(os.path.dirname(__file__), 'AttentionXML'))
        from deepxml.evaluation import *
        from deepxml.data_utils import get_data, get_mlb
        from ruamel.yaml import YAML
        from deepxml.tree import FastAttentionXML

        yaml = YAML(typ='safe')
        nb_trees = 1
        for attentionxml_name in classifiers['attentionxml']:
            try:
                with open(f'./results/predictions/attentionxml_{attentionxml_name}_ensemble.pkl', 'rb') as in_f:
                    attentionxml_predictions = pickle.load(in_f)
            except FileNotFoundError:
                attentionxml_predictions = torch.zeros(len(df), len(label_map))
                df['text'] = df['_'.join(get_datatype([attentionxml_name])[0])]
                data_cnf, model_cnf = yaml.load(Path(f"./AttentionXML/configure/datasets/{attentionxml_name}.yaml")), yaml.load(Path(f"AttentionXML/configure/models/FastAttentionXML-{attentionxml_name}.yaml"))
                mlb = get_mlb(data_cnf['labels_binarizer'], [list(label_map.keys())])

                try:
                    test_x, _ = get_data(f"tmp_{attentionxml_name}/test_texts.npy", None)
                except:
                    os.makedirs(f"tmp_{attentionxml_name}", exist_ok=True)
                    df['text'].to_csv(f"tmp_{attentionxml_name}/test_raw_texts.txt", index=False, header=False)
                    os.system(f"python ./AttentionXML/preprocess.py --text-path ./tmp_{attentionxml_name}/test_raw_texts.txt --tokenized-path ./tmp_{attentionxml_name}/test_texts.txt --vocab-path ./AttentionXML/data/{attentionxml_name}/vocab.npy")
                    test_x, _ = get_data(f"tmp_{attentionxml_name}/test_texts.npy", None)
                logger.info(F'Size of Test Set: {len(test_x)}')

                labels_att, scores_att = [], []
                for tree_id in range(nb_trees):
                    tree_id = f"-Tree-{tree_id}"
                    model = FastAttentionXML(len(label_map), data_cnf, model_cnf, tree_id)
                    scores, labels = model.predict(test_x)
                    labels = mlb.classes_[labels]
                    scores_att.append(scores), labels_att.append(labels)
                
                for i in range(len(labels_att[0])): # for each dataset
                    s = defaultdict(float)
                    for j in range(len(labels_att[0][i])): # for each label
                        for k in range(len(labels_att)):    # for tree
                            s[labels_att[k][i][j]] += scores_att[k][i][j]  

                    # pad to complete vector (to adapt to LightXML predictions)
                    for s_k, s_v in s.items():
                        attentionxml_predictions[i, int(s_k)] = s_v

                # save predictions
                with open(f'./results/predictions/attentionxml_{attentionxml_name}_ensemble.pkl', 'wb') as out_f:
                    pickle.dump(attentionxml_predictions, out_f)
            predicts.append(attentionxml_predictions)
    ######################################################################
    total = len(df)
    # print(len(predicts))

    acc1 = [0 for i in range(len(predicts) + 1)]       # shape (1, nb_ensembles * nb_of_models_in_each_ensemble + 1)
    acc3 = [0 for i in range(len(predicts) + 1)]
    acc5 = [0 for i in range(len(predicts) + 1)]

    # save prediction results for error analysis
    preds = []
    nb_true_labels = 0
    for index, true_labels in enumerate(df.label.values):
        try:
            true_labels = set([i for i in true_labels.split()])
            nb_true_labels += len(true_labels)
        except AttributeError:
            continue

        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(predicts))] 
        logits.append(sum(logits)) 
        logits = [(-i).argsort()[:100].cpu().numpy() for i in logits]

        for i, logit in enumerate(logits):
            logit_code = [str(logit[k]) for k in range(len(logit))] 
            acc1[i] += len(set([logit_code[0]]) & true_labels)
            acc3[i] += len(set(logit_code[:3]) & true_labels)
            acc5[i] += len(set(logit_code[:5]) & true_labels)
        preds.append([label_decode_dict[e] for e in logit[:10]])

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble.out', 'w') as f:
        for i, name in enumerate(berts * len(classifiers['lightxml']) + ['attentionxml'] * len(classifiers['attentionxml']) + ['all']):
            p1 = acc1[i] / total
            p3 = acc3[i] / total / 3
            p5 = acc5[i] / total / 5
            r1 = acc1[i] / nb_true_labels
            r3 = acc3[i] / nb_true_labels
            r5 = acc5[i] / nb_true_labels
            print(f'{name} P@1:{p1}, P@3:{p3}, P@5:{p5}, R@1:{r1}, R@3:{r3}, R@5:{r5}', file=f)
            logger.info(f'{name} P@1:{p1}, P@3:{p3}, P@5:{p5}, R@1:{r1}, R@3:{r3}, R@5:{r5}')

    # if args.weighted_average:  ###TODO add eval set
    #     p1s = torch.zeros(len(berts) * len(classifiers['lightxml']) + len(classifiers['attentionxml']), 1)
    #     for i in range(len(p1s)):
    #         p1s[i] = acc1[i] / total
        
    #     ### scale precision@1 weights with exponential function
    #     #p1s = np.exp(p1s)
    #     ### sigmoid function
    #     p1s = 1/(1 + np.exp(-p1s))

    #     acc1, acc3, acc5 = 0, 0, 0
    #     nb_true_labels = 0
    #     for index, true_labels in enumerate(df.label.values):
    #         try:
    #             true_labels = set([i for i in true_labels.split()])
    #         except AttributeError:
    #             continue

    #         logits = [torch.sigmoid(predicts[i][index]) for i in range(len(predicts))]  # (nb_classifiers * nb_labels)
    #         logits = torch.stack(logits)
    #         logits = torch.squeeze(sum(p1s * logits))
    #         logit_code = [str(code) for code in logits.argsort(descending=True)[:100].cpu().numpy()]

    #         acc1 += len(set([logit_code[0]]) & true_labels)
    #         acc3 += len(set(logit_code[:3]) & true_labels)
    #         acc5 += len(set(logit_code[:5]) & true_labels)
    #         nb_true_labels += len(true_labels)
            
    #     p1 = acc1 / total
    #     p3 = acc3 / total / 3
    #     p5 = acc5 / total / 5

    #     r1 = acc1 / nb_true_labels
    #     r3 = acc3 / nb_true_labels
    #     r5 = acc5 / nb_true_labels

    #     logger.info(f'all (weighted) P@1:{p1}, P@3:{p3}, P@5:{p5}, R@1:{r1}, R@3:{r3}, R@5:{r5}')

    with open(f'./results/INPI_{str(args.pred_level)}_ensemble_pred.txt', 'w') as out_f:
        lines = [str(l) for l in preds]
        out_f.write("\n".join(lines))