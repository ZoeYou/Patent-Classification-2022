import pandas as pd 
import numpy as np
import pickle, os, argparse, sys, csv 
from pathlib import Path
csv.field_size_limit(sys.maxsize)

# import nltk ###nltk.download('stopwords')
from nltk import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer 

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer

from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from tqdm import tqdm
import spacy

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

    discount = 1 / (np.log2(np.arange(k) + 2))
    idiscount = 1 / (np.log2(np.arange(min(k, len(rel_true))) + 2))

    if form == "linear":
        idcg = np.sum(rel_true * idiscount)
        dcg = np.sum(rel_pred * discount)
    elif form == "exponential" or form == "exp":
        idcg = np.sum([2**x - 1 for x in rel_true] * idiscount)
        dcg = np.sum([2**x - 1 for x in rel_pred] * discount)
    else:
        raise ValueError("Only supported for two formula, 'linear' or 'exp'")

    return dcg / idcg


def eval(predictions, labels, k=1):
    """
    Return weighted precision and recall after getting all the predictions
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
    
global language 
language = {'fr': 'french', 'en':'english'}
            
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
    parser.add_argument("--in_file", default='../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input file of INPI-CLS data.")
    parser.add_argument("--in_dir", type=str, help="Path to input directory containing train.tsv and test.tsv")
    parser.add_argument("--out_dir", type=str, default='./models', help="Path to output directory to save models")
    
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")
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
    parser.add_argument("--model", type=str, required=True, choices={"SVM","NB","LR"}, help="Model selected for training. (SVM stands for Support Vector Machine, NB stands for Naive Baysian, and LR stands for Logistic Regression.)")
    
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    
    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (especially for INPI-CLS data <split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 1000, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")

    parser.add_argument("--feature_dimension", type=int, default=10000, help="Dimension of input features (of tf-idf) for classifier.")
    parser.add_argument("--keep_stop_words", action="store_true", help="Whether to keep stop words instead of removing them")
    parser.add_argument("--do_stemmer", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to apply a stemmer.")
    parser.add_argument("--do_lemma", default=False, type=lambda x: (str(x).lower() == 'true'), help="Whether to apply lemmatization.")
    parser.add_argument("--max_iter", default=100, type=int, help="Max number of iterations for optimisation of classifier. (Especially for Logistic Regression)")


    args = parser.parse_args()

    if args.keep_stop_words:
        indice_stop_words = "stopWords_True"
    else:
        indice_stop_words = "stopWords_False"
    if args.do_stemmer:
        indice_do_stemmer = "withStemmer"
    else:
        indice_do_stemmer = "noStemmer"
    if args.do_lemma:
        indice_do_lemma = "withLemmatizer"
    else:
        indice_do_lemma = "noLemmatizer"
    if args.in_dir:
        data_name = args.in_dir.strip("/").split("/")[-1]
    else:
        data_name = "INPI"

    # lemmatizer and stemmer can not be used at the same time  
    assert args.do_lemma != args.do_stemmer

    output_path = os.path.join(args.out_dir,'-'.join([args.model, data_name, "maxInputLength_"+str(args.max_input_length), "featureDimension_"+str(args.feature_dimension), indice_stop_words, indice_do_stemmer, indice_do_lemma, "maxIter_"+str(args.max_iter)]))
    output_path = Path(output_path)

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    print("***** Reading standard label file *****")
    if args.pred_level == 1:
        labels_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    else:
        with open(args.label_file, 'r') as in_f:
            lines = in_f.read().splitlines()
        labels_list = [l.split('\t')[0] for l in lines]

            
    if args.do_stemmer:
        global stemmer 
        stemmer = SnowballStemmer(language[args.lang])
    if args.do_lemma:
        global lemmatizer 
        model_name = f"{args.lang}_core_news_sm"
        lemmatizer = spacy.load(model_name, disable = ['parser','ner'])

    if not args.keep_stop_words:
        print("***** Creating stop words list *****")
        global stop_words
        stop_words = []
        if args.lang == 'fr':
            # source1: https://github.com/stopwords-iso/stopwords-fr
            with open(args.fr_stop_words_file,'r') as in_f:
                lines = in_f.read().splitlines()
            stop_words += lines
            # source2: nltk
            from nltk.corpus import stopwords
            stop_words += stopwords.words('french')
            # source3: spacy 
            from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
            stop_words += list(fr_stop)
            stop_words = list(set(stop_words))
        elif args.lang == 'en':
            # source1: https://countwordsfree.com/stopwords
            with open(args.en_stop_words_file,'r') as in_f:
                lines = in_f.read().splitlines()
            stop_words += lines
            # source2: nltk
            from nltk.corpus import stopwords
            stop_words += stopwords.words('english')
            # source3: spacy 
            from spacy.lang.en.stop_words import STOP_WORDS as en_stop
            stop_words += list(en_stop)
            stop_words = list(set(stop_words))
        elif args.lang == 'de':
            with open(args.de_stop_words_file,'r') as in_f:
                lines = in_f.read().splitlines()
            stop_words += lines
            # source2: nltk
            from nltk.corpus import stopwords
            stop_words += stopwords.words('german')
            # source3: spacy 
            from spacy.lang.de.stop_words import STOP_WORDS as de_stop
            stop_words += list(de_stop)
            stop_words = list(set(stop_words))
        print(' Done! Number of stop words: ', len(stop_words))

    year = args.split_by_year
    label = 'IPC' + str(args.pred_level)

    def tokenize_and_stem(text, lang=args.lang):
        tokens = [word for sent in sent_tokenize(text, language=language[lang]) for word in word_tokenize(sent, language=language[lang])]
        if args.max_input_length > 0:
            tokens = tokens[:args.max_input_length]
        #remove numbers and other symbols
        #filtered_tokens = []
        #for token in tokens:
        #   if not (token.isdigit() or token.isalnum()):
        #       filtered_tokens.append(token)

        #return tokens if don't do lemmatisation/ stemming
        if args.do_stemmer:
            tokens = [stemmer.stem(t) for t in tokens]
        elif args.do_lemma:
            doc = lemmatizer(' '.join(tokens))
            tokens = [t.lemma_ for t in doc]

        #exclude stopwords from stemmed/lemmatized words
        if not args.keep_stop_words:
            tokens = [t for t in tokens if t.lower() not in stop_words]

        return ' '.join(tokens)

    if args.do_train: 
        print("***** Creating training and testing data *****")
        target_sections = ""
        if args.in_dir:                 
            df_train = read_tsv(os.path.join(args.in_dir, 'train.tsv'), labels_list, ipc_level=args.pred_level)
            df_test = read_tsv(os.path.join(args.in_dir, 'test.tsv'), labels_list, ipc_level=args.pred_level)

        else:
            df = pd.read_csv(args.in_file, dtype=str, engine="python") #.dropna() #TODO TBD
            dict_target_secs = {'title': 'title', 'abstract': 'abs', 'claims': 'claims', 'description': 'desc'}

            target_sections = [dict_target_secs[s] for s in args.target_section]
            for sec in target_sections:
                df.loc[:,sec] = df[sec].apply(str)
            df.loc[:,'text'] = df[target_sections].apply('. '.join, axis=1)
            df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
            df_train = df_train[['text', label]]
            df_train[label] = df_train[label].apply(lambda x: ",".join([l for l in str(x).split(",") if l in labels_list]))
        
            df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
            df_test = df_test[['text', label]]
            df_test[label] = df_test[label].apply(lambda x: ",".join([l for l in str(x).split(",") if l in labels_list]))
            
        df_train = df_train.sample(frac=1, random_state=666).reset_index(drop=True)
        X_train = df_train['text'].apply(tokenize_and_stem).values
        y_train = [label.split(',') for label in df_train[label].values]

        df_test = df_test.sample(frac=1, random_state=666).reset_index(drop=True)
        X_test = df_test['text'].apply(tokenize_and_stem).values
        y_test = [label.split(',') for label in df_test[label].values]
        print(df_train)
        print(df_test)

        ### Label encoding
        mlb = MultiLabelBinarizer()
        mlb.fit(y_train+y_test)
        print(mlb.classes_)
        print(len(mlb.classes_))

        y_train = mlb.transform(y_train)
        print('Datasets Converted!')

        # create pipeline of models
        featureVectorizer = TfidfVectorizer(max_features=args.feature_dimension, max_df=0.9)

        if args.model == "SVM":
            classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True, random_state=24, class_weight='balanced'), n_jobs=1)
        elif args.model == "NB":
            classifier = OneVsRestClassifier(MultinomialNB(fit_prior=True, class_prior=None))
        elif args.model == "LR":
            classifier = OneVsRestClassifier(LogisticRegression(solver='sag', random_state=24, max_iter = args.max_iter), n_jobs=1)

        # fit model
        pipeline = Pipeline(steps=[
            ('vectorizer', featureVectorizer),
            ('clf', classifier)])
        print("***** Running training *****")
        pipeline.fit(X_train.copy(), y_train)

        # save model
        secs_name = '_'.join(target_sections)
        pickle.dump((pipeline, mlb), open(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.pkl'), 'wb'))
        print('Model saved!')

    if args.do_test:
        if args.in_dir:
            df_test = read_tsv(str(os.path.join(args.in_dir, 'test.tsv')), labels_list, ipc_level=args.pred_level)
            target_sections = ""
        else:
            df = pd.read_csv(args.in_file, dtype=str, engine="python") #.dropna() #TODO TBD
            dict_target_secs = {'title': 'title', 'abstract': 'abs', 'claims': 'claims', 'description': 'desc'}

            target_sections = [dict_target_secs[s] for s in args.target_section]
            for sec in target_sections:
                df.loc[:,sec] = df[sec].apply(str)
            df.loc[:,'text'] = df[target_sections].apply('. '.join, axis=1)
            df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
            df_test = df_test[['text', label]]
            df_test[label] = df_test[label].apply(lambda x: ",".join([l for l in str(x).split(",") if l in labels_list]))



        df_test = df_test[['text', label]].dropna()
        df_test = df_test.sample(frac=1, random_state=666).reset_index(drop=True)
        X_test = df_test['text'].apply(tokenize_and_stem).values
        y_test = [label.split(',') for label in df_test[label].values]

        # read model
        secs_name = '_'.join(target_sections)
        pipeline, mlb = pickle.load(open(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.pkl'), 'rb'))
        print('Model loaded!')

        class_list = list(mlb.classes_)

        # Get predictions for test data
        print("***** Running prediction *****")
        y_test_pred = pipeline.predict_proba(X_test)


        pre_n_1 = []
        pre_n_3 = []
        pre_n_5 = []

        ndcg_n_1 = []
        ndcg_n_3 = []
        ndcg_n_5 = []

        predictions = []

        for i in tqdm(range(len(y_test))):
            true = y_test[i] 
            pred = [x for _,x in sorted(zip(y_test_pred[i], class_list), reverse=True)]
            predictions.append(pred)

            pre_1 = precision(true, pred, 1)
            pre_3 = precision(true, pred, 3)
            pre_5 = precision(true, pred, 5)

            ndcg_1 = nDCG(true, pred, 1)
            ndcg_3 = nDCG(true, pred, 3)
            ndcg_5 = nDCG(true, pred, 5)

            if pre_1 != ndcg_1:
                print(666666, true, pred)

            pre_n_1.append(pre_1)
            pre_n_3.append(pre_3)
            pre_n_5.append(pre_5)
        
            ndcg_n_1.append(ndcg_1)
            ndcg_n_3.append(ndcg_3)
            ndcg_n_5.append(ndcg_5)

        res_df = pd.DataFrame({'true_labels': y_test, 
                               'predict_labels': predictions, 
                               'precision@1': pre_n_1, 
                               'precision@3': pre_n_3,
                               'precision@5': pre_n_5,
                               'nDCG@1': ndcg_n_1, 
                               'nDCG@3': ndcg_n_3,
                               'nDCG@5': ndcg_n_5,                              
                               })

        res_df.to_csv(os.path.join(output_path, f'SVC-{secs_name}-{args.split_by_year}-{args.pred_level}.res'), index=False)
        print(res_df)
        
        for col in ["precision@1", "precision@3", "precision@5", "nDCG@1", "nDCG@3", "nDCG@5"]:
            print(col + ": ", res_df[col].mean())

if __name__ == "__main__":
    main()


