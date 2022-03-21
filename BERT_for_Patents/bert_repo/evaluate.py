import pandas as pd
import sys, csv, json
from collections import OrderedDict
from tqdm import tqdm
from operator import itemgetter

csv.field_size_limit(sys.maxsize)

def precision(actual, predicted, k):
    """
    example-based precision (for one example)
    """
    act_set = set(actual)

    if len(predicted) < k:
        pred_set = set(predicted)
    else:
        pred_set = set(predicted[:k])

    if len(act_set) ==0 or k == 0: 
        result = 0.0
    else:
        result = len(act_set & pred_set) / float(k)
    return result

def recall(actual, predicted, k):
    """
    example-based recall (for one example)
    """
    act_set = set(actual)

    if len(predicted) < k:
        pred_set = set(predicted)
    else:
        pred_set = set(predicted[:k])

    if len(act_set) == 0:
        result = 1.0
    else:
        result = len(act_set & pred_set) / float(len(act_set))
    return result

def eval(predictions, labels, k):
    """
    label-based micro percision and recall (for all examples)
    """
    precision = 0.0
    nexamples = 0
    nlabels = 0
    for prediction, labels in zip(predictions, labels):
        for p in prediction:
            if p in labels:
                precision += 1
        nexamples += 1
        nlabels += len(labels)
    return (precision / (k * nexamples), precision / nlabels)

def get_labels(label_file=None):
    """See base class."""
    lst = []
    #TODO
    #cpc_id = 'section_id'      # A~H, Y
    #cpc_id = 'subsection_id'   # A01
    #cpc_id = 'group_id'        # A01B
    #cpc_id = 'subgroup_id'     # C40B40/14

    if label_file:
        f = open(label_file)
        reader = csv.reader(f, delimiter='\t')
        lines = []
        for i, line in enumerate(reader):
            if len(line[0]) == 0:
                continue
            lst.append(line[0])

        f.close() 
        return lst[1:]
    else:
        lst = [chr(i) for i in range(ord('A'), ord('H')+1) ]
        return lst




TEST_FILE = sys.argv[1]  #'content/PatentBERT/test_data.tsv'
PRED_FILE = sys.argv[2]  #'results/predict_result.txt'
label_file = sys.argv[3] #../../data/ipc-sections/20210101/labels_group_id_4.tsv
K = int(sys.argv[4]) # 5
level_ipc = int(label_file[-5])


print("Reading test file...")
true_labels = []

with open(TEST_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader):
        true_labels.append(row['group_ids'])

if level_ipc == 4:
    true_labels = [list(OrderedDict.fromkeys([label[:4] for label in l.split(',')])) for l in true_labels]
elif level_ipc == 6:
    true_labels = [list(OrderedDict.fromkeys([label.split("/")[0] for label in l.split(',')])) for l in true_labels] 
elif level_ipc == 8:
    true_labels = [list(label for label in l.split(',')) for l in true_labels] 



with open(PRED_FILE, 'r') as in_f:
    pred_scores = []
    for l in in_f:
    	pred_scores.append([float(e) for e in l.split('\t')])

    predictions = []
            
    if level_ipc == 1:
        labels = get_labels()
    else:
        labels = get_labels(label_file)
        print("Using label file: ", label_file)
        print(len(labels))

    for line in pred_scores:
        dict_tmp = dict(zip(labels, line))
        dict_sorted = {k: v for k,v in sorted(dict_tmp.items(), reverse=True, key=itemgetter(1))[:K]}
        print(dict_sorted)
        predictions.append(list(dict_sorted.keys()))

assert len(true_labels) == len(predictions)

precisions = []
recalls = []
f1_at_ks = []

# example-based metrics calculation
for i in tqdm(range(len(predictions))):
    true = [str(l) for l in true_labels[i]]
    pred = [str(l) for l in predictions[i]]

    pre = precision(true, pred, K)
    rec = recall(true, pred, K)
    if pre != 0 and rec != 0:
        f1_at_k = 2 * pre * rec / (pre + rec)
    else:
        f1_at_k = 0.0

    precisions.append(pre)
    recalls.append(rec)
    f1_at_ks.append(f1_at_k)


res_df = pd.DataFrame({'true_labels': true_labels, 'predict_labels': predictions, 'precision': precisions, 'recall': recalls, 'F1_at_k': f1_at_ks})
print(res_df)

# save prediction results
file_name = PRED_FILE.split('/')[-2]
res_df.to_csv(f'res_eval/{file_name}_res.csv', index=False)

# label-based metrics calculation
eval_micro = eval(predictions, true_labels, k=K)
precision_micro, recall_micro = eval_micro
f1atk_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
print('Precision: ', precision_micro, 'Recall: ', recall_micro, 'F1 at k: ', f1atk_micro)
