import csv, fasttext, argparse, os, sys
from pathlib import Path
from nltk import word_tokenize, sent_tokenize
csv.field_size_limit(sys.maxsize)
import numpy as np, pandas as pd

def unique(sequence):
    # convert list to set without changing order of elements
    return dict.fromkeys(sequence).keys()

def precision(actual, predicted, k):
    act_set = unique(actual)
    
    if len(predicted) < k:
        pred_set = unique(predicted)
    else:
        pred_set = unique(predicted[:k])

    if len(act_set) ==0 or k == 0: 
        result = 0.0
    else:
        result = len(act_set & pred_set) / float(k)
    return result

def recall(actual, predicted, k):
    act_set = unique(actual)

    if len(predicted) < k:
        pred_set = unique(predicted)
    else:
        pred_set = unique(predicted[:k])

    if len(act_set) == 0:
        result = 1.0
    else:
        result = len(act_set & pred_set) / float(len(act_set))
    return result

def nDCG(true_labels, pred_labels, k, form="linear"):
    """ Returns normalized Discounted Cumulative Gain
    Args:
        true_labels (list): true label list (the first being the principal IPC)
        pred_labels (list): predicted relevance list
        k (int): number of top k to keep in rel_pred
        form (string): two types of nDCG formula, 'linear' or 'exponential'
    Returns:
        ndcg (float): normalized discounted cumulative gain score [0, 1]
    """
    rel_true = np.ones(len(true_labels[:k]))
    # rel_true[0] = 2.    # score pertinence plus élevé pour IPC principale

    predicted_correct = unique(true_labels) & unique(pred_labels[:k])
    rel_pred =  np.asarray([(1 if label in predicted_correct else 0) for label in pred_labels[:k]])

    idiscount = 1 / (np.log2(np.arange(min(k, len(rel_true))) + 2))
    discount = 1 / (np.log2(np.arange(k) + 2))

    if form == "linear":
        idcg = np.sum(rel_true * idiscount)
        dcg = np.sum(rel_pred * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true] * idiscount)
        dcg = np.sum([2**x - 1 for x in rel_pred] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg

def read_tsv(file_path, labels_list, ipc_level=4):
    texts = []
    labels = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                texts.append(row['text'])
                labels_to_check = list(set([l[:ipc_level] for l in row['group_ids'].split(',')]))
                labels_checked = [l for l in labels_to_check if l in labels_list]
                labels.append(','.join(labels_checked))
        df = pd.DataFrame(zip(texts, labels), columns=['text','IPC' + str(ipc_level)])
    except KeyError:
        df = pd.read_csv(file_path, sep="\t", skipinitialspace=True, usecols=['text','group_ids'], dtype=object)
        df['group_ids'] = df['group_ids'].apply(str)
        df['text'] = df['text'].apply(str)
        df = df.rename(columns={"group_ids": 'IPC' + str(ipc_level)})
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input directory.")
    parser.add_argument("--in_dir", type=str, help="Path to input directory containing train.tsv and test.tsv")
    parser.add_argument("--out_dir", type=str, default='./models', help="Path to output directory to save models") 
    
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")
    parser.add_argument("--from_pretrained", default="wiki.fr.vec", help="Whether to use pretrained vectors for initializatoin.")
    parser.add_argument("--max_wordNgrams", type=int, default=1, help="max length of word ngram [1].")
    parser.add_argument("--one_vs_all", action="store_true", help="Whether to use independent binary classifiers for each label.")

    parser.add_argument("--fr_stop_words_file", default="stopwords-fr.txt", type=str)
    parser.add_argument("--en_stop_words_file", default="stopwords-en.txt", type=str)
    parser.add_argument("--de_stop_words_file", default="stopwords-de.txt", type=str)
    parser.add_argument("--label_file", default='../data/ipc-sections/20210101/labels_group_id_4.tsv', type=str)

    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--epoch", default=100, type=int, help="Number of epochs.")

    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 1000, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")
    parser.add_argument("--remove_stop_words", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to remove stop words from input text") 


    args = parser.parse_args()

    if args.from_pretrained:
        indice_pretrained = "pretrainedVector_"+args.from_pretrained.split('.')[0]
    else:
        indice_pretrained = 'noPretrainedVector'   
    if args.one_vs_all:
        indice_loss = 'loss_OVA'
    else:
        indice_loss = 'loss_NS'
    if args.remove_stop_words:
        indice_sw = "noStopWords"
    else:
        indice_sw = "withStopWords"
    if args.in_dir:
        data_name = args.in_dir.strip("/").split("/")[-1]
    else:
        data_name = "INPI"

    output_path = os.path.join(args.out_dir, '-'.join(["fastText", data_name,"maxInputLength_"+str(args.max_input_length), "maxNgrams_"+str(args.max_wordNgrams), "nbEpochs_"+str(args.epoch), indice_pretrained, indice_loss, indice_sw]))
    output_path = Path(output_path)

    print("***** Reading standard label file *****")
    label = 'IPC' + str(args.pred_level)
    if args.pred_level == 1:
        labels_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    else:
        with open(args.label_file, 'r') as in_f:
            lines = in_f.read().splitlines()[1:]
        labels_list = [l.split('\t')[0] for l in lines]   

    print("***** Creating training and testing data *****")
    global secs_name
    if args.in_dir:
        train_path = os.path.join(output_path, 'train.txt')
        test_path = os.path.join(output_path, 'test.txt')
        secs_name = data_name
    else:
        dict_target_secs = {'title': 'title',
                            'abstract': 'abs',
                            'claims': 'claims',
                            'description': 'desc'}
        target_sections = [dict_target_secs[s] for s in args.target_section]
        secs_name = '_'.join(target_sections)
        year = args.split_by_year      

        train_path = os.path.join(output_path,f'{secs_name}-{year}-{label}-train.txt')
        test_path = os.path.join(output_path,f'{secs_name}-{year}-{label}-test.txt')
    
    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    if args.in_dir:                 
        df_train = read_tsv(os.path.join(args.in_dir, 'train.tsv'), labels_list, ipc_level=args.pred_level)
        df_train[label] = df_train[label].apply(lambda x: ' '.join(['__label__' + l.replace(" ","") for l in str(x).split(',')]))

        df_test = read_tsv(os.path.join(args.in_dir, 'test.tsv'), labels_list, ipc_level=args.pred_level)
        df_test["label_orig"] = df_test[label].apply(lambda x: ",".join([l.replace(" ","") for l in x.split(",")]))  # string of labels split by ","
        df_test[label] = df_test[label].apply(lambda x: ' '.join(['__label__' + l.replace(" ","") for l in str(x).split(',')]))
    else:   
        # Import dataset
        df = pd.read_csv(args.in_file, dtype=object, engine="python")
        for sec in target_sections:
            df[sec] = df[sec].apply(str)

        df['text'] = df[target_sections].apply('. '.join, axis=1)
        if args.max_input_length > 0:
            df['text'] = df['text'].apply(lambda x: ' '.join(x.split(' ')[:args.max_input_length]))
        df['text'] = df['text'].apply(lambda x: x.replace('\n', ''))
        df["label_orig"] = df[label].apply(lambda x: ",".join([l.replace(" ","") for l in str(x).split(",")]))  # string of labels split by ","

        df[label] = df[label].apply(lambda x: ' '.join(['__label__' + l.replace(" ","") for l in str(x).split(',') if l in labels_list]))
        
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_train = df_train[[label, 'text']].dropna()

        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        df_test = df_test[[label, 'text', "label_orig"]].dropna()
        
    print(df_train)
    print(df_test)

    with open(train_path, 'w') as out_f1:
        out_f1.write('\n'.join(df_train[label] + ' ' + df_train['text']))
    with open(test_path, 'w') as out_f2:
        out_f2.write('\n'.join(df_test[label] + ' ' + df_test['text']))       
    
    if args.one_vs_all:
        loss_func = "ova"
    else:
        loss_func = "ns"

    if args.do_train:
        if args.from_pretrained:
            model = fasttext.train_supervised(train_path, dim=300, epoch=args.epoch, wordNgrams=args.max_wordNgrams, pretrainedVectors=args.from_pretrained, loss=loss_func)
        else:
            model = fasttext.train_supervised(train_path, dim=300, epoch=args.epoch, wordNgrams=args.max_wordNgrams, loss=loss_func)
        model.save_model(os.path.join(output_path, f'fastText-{secs_name}-{args.max_input_length}-{args.split_by_year}-{args.pred_level}.bin'))
    
    if args.do_test:
        from collections import defaultdict
        model = fasttext.load_model(os.path.join(output_path, f'fastText-{secs_name}-{args.max_input_length}-{args.split_by_year}-{args.pred_level}.bin'))

        print(f'fastText-{secs_name}-{args.max_input_length}-{args.split_by_year}-{args.pred_level}.bin')

        pre_n_1 = []
        pre_n_3 = []
        pre_n_5 = []

        rec_n_1 = []
        rec_n_3 = []
        rec_n_5 = []

        predictions = []
        y_test = []
        
        labels = [l.split(",") for l in df_test["label_orig"].tolist()]
        texts = df_test["text"].tolist()

        # prediction of label scores
        pred_scores = defaultdict(list)

        for i in range(len(texts)):
            true = labels[i]
            labels_l, scores_l = model.predict(texts[i], k=len(model.labels))
            for l, s in zip(labels_l, scores_l):
                pred_scores[l.replace("__label__", "")].append(s)
         
            pred = [res.replace("__label__", "") for res in labels_l] 

            y_test.append(true)
            predictions.append(pred)

            pre_1 = precision(true, pred, 1)
            pre_3 = precision(true, pred, 3)
            pre_5 = precision(true, pred, 5)

            rec_1 = recall(true, pred, 1)
            rec_3 = recall(true, pred, 3)
            rec_5 = recall(true, pred, 5)

            pre_n_1.append(pre_1)
            pre_n_3.append(pre_3)
            pre_n_5.append(pre_5)
        
            rec_n_1.append(rec_1)
            rec_n_3.append(rec_3)
            rec_n_5.append(rec_5)

        res_df = pd.DataFrame({'true_labels': y_test, 
                               'predict_labels': predictions, 
                               'precision@1': pre_n_1, 
                               'precision@3': pre_n_3,
                               'precision@5': pre_n_5,
                               'recall@1': rec_n_1, 
                               'recall@3': rec_n_3,
                               'recall@5': rec_n_5,                              
                               })

        # save prediction scores
        score_df = pd.DataFrame.from_dict(pred_scores)
        score_df.to_csv(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.score'), index=False)
        # save predictions results 
        res_df.to_csv(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.res'), index=False)

        for col in ["precision@1", "precision@3", "precision@5", "recall@1", "recall@3", "recall@5"]:
            print(col + ": ", res_df[col].mean())

if __name__ == "__main__":
    main()
