import pandas as pd
import sys
import csv
import json
from collections import OrderedDict
from tqdm import tqdm
from operator import itemgetter

csv.field_size_limit(sys.maxsize)

def precision(actual, predicted, k):
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

def get_labels(label_file=None):
    import csv
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
        lst = [ chr(i) for i in range(ord('A'), ord('H')+1) ]
        return lst




TEST_FILE = sys.argv[1]  #'content/PatentBERT/test_data.tsv'
PRED_FILE = sys.argv[2]  #'results/predict_result.txt'
K = int(sys.argv[3]) # 5
label_file = "labels_group_id_4.tsv"
threshold = 0
level_ipc = 4


"""
df = pd.read_csv(TEST_FILE, sep='\t')
df = df[df['group_ids'].apply(lambda x: True if len(str(x)) > 3 else False)]
"""
print("Read test file...")
with open(TEST_FILE, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    true_labels = []
    for row in tqdm(reader):
        true_labels.append(row['group_ids'])


#true_labels = df['group_ids'].apply(lambda x: str(x)).to_list()
if level_ipc == 4:
    true_labels = [list(OrderedDict.fromkeys([label[:4] for label in l.split(',')])) for l in true_labels]

with open(PRED_FILE, 'r') as in_f:
    pred_scores = in_f.read().splitlines()
    pred_scores = [[float(e) for e in l.split('\t')] for l in pred_scores]

    predictions = []
            
    if level_ipc == 1:
        labels = get_labels()
    elif level_ipc == 4:
        labels = get_labels(label_file)
        print(len(labels))

    for line in pred_scores:
        #if line == ['']:
        #    predictions.append(line)

        dict_tmp = dict(zip(labels, line))
            
        #print(dict_tmp)
        #print(sum([v for k, v in dict_tmp.items()]))
        dict_sorted = {k: v for k,v in sorted(dict_tmp.items(), reverse=True, key=itemgetter(1))}
        #{k: v for k, v in sorted(dict_tmp.items(), reverse=True, key=lambda item: item[1])} # if v >= threshold}
        predictions.append(list(dict_sorted.keys()))

precisions = []
recalls = []
f1_at_ks = []


print(len(true_labels))
print(len(predictions))

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

file_name = PRED_FILE.split('/')[0]
res_df.to_csv(f'res_eval/{file_name}_res.csv', index=False)

print('Precision: ', res_df['precision'].mean(), 'Recall: ', res_df['recall'].mean(), 'F1 at k: ', res_df['F1_at_k'].mean())
