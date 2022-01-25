from fast_bert.prediction import BertClassificationPredictor
import pandas as pd
from tqdm import tqdm
import sys

OUTPUT_DIR = sys.argv[1]
LABEL_PATH = './'
MODEL_PATH = OUTPUT_DIR +'/model_out'
test_file = OUTPUT_DIR + '/test.csv'
K = int(sys.argv[2])


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

predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='camembert-base',
				do_lower_case=False)


# read test texts
df_test = pd.read_csv(test_file)
text_list = df_test['text'].to_list()

if 'epo' in OUTPUT_DIR:
    import csv
    from collections import OrderedDict
    csv.field_size_limit(sys.maxsize)

    file_name = '../data/EPO/epo_data/epo_fr_CLAIM1_from_2015/test.tsv' # TODO
    true_labels = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader):
            true_labels.append(row['group_ids'])
    true_labels = [list(OrderedDict.fromkeys([label[:4] for label in l.split(',')])) for l in true_labels]
else:
    # read labels from original data file
    test_df0 = pd.read_csv('orig_test.csv')
    true_labels = [l.split(',') for l in test_df0['IPC'+OUTPUT_DIR.strip('/ ').split('_')[-1]].to_list()]
    print(len(true_labels))

multilabel_predictions = predictor.predict_batch(text_list)
predictions = [[l[0] for l in line[:10]] for line in multilabel_predictions]
print(len(predictions))

precisions = []
recalls = []
f1_at_ks = []

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

print('Precision: ', res_df['precision'].mean(), 'Recall: ', res_df['recall'].mean(), 'F1 at k: ', res_df['F1_at_k'].mean())


