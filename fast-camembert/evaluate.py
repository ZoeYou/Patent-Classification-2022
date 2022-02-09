from fast_bert.prediction import BertClassificationPredictor
import pandas as pd
from tqdm import tqdm
import sys

OUTPUT_DIR = sys.argv[1]    # model directory
DATA_DIR = sys.argv[2]  # data directory
K = 1 #int(sys.argv[3])
LABEL_PATH = './'
MODEL_PATH = OUTPUT_DIR +'/model_out'
test_file = DATA_DIR + '/test.csv'
cross_test = True
IPC_level = 4


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

def eval(predictions, labels, k=1):
    """
    Return precision and recall modeled after fasttext's test
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



predictor = BertClassificationPredictor(
				model_path=MODEL_PATH,
				label_path=LABEL_PATH, # location for labels.csv file
				multi_label=True,
				model_type='camembert-base',
				do_lower_case=False)


# read test texts
df_test = pd.read_csv(test_file)
text_list = df_test['text'].to_list()

global cols
cols = list(df_test.columns[:-1])

def get_labels(line):
    res = []

    for l in cols:
        if line[l] == 1:
            res.append(l)
    return res
true_labels = df_test.apply(get_labels, axis=1).to_list()


multilabel_predictions = predictor.predict_batch(text_list) # !!!!!!PROBLEM!!!!!!?????
predictions = [[l[0] for l in line[:K]] for line in multilabel_predictions]
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

eval_micro = eval(predictions, true_labels, k=K)
precision_micro = eval_micro[0]
recall_micro = eval_micro[1]
f1atk_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro)
print('Precision: ', precision_micro, 'Recall: ', recall_micro, 'F1 at k: ', f1atk_micro)

#example-based metric
#print('Precision: ', res_df['precision'].mean(), 'Recall: ', res_df['recall'].mean(), 'F1 at k: ', res_df['F1_at_k'].mean())


