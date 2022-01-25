from xmlrpc.client import boolean
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
    
global language 
language = {'fr': 'french', 'en':'english'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/INPI/inpi_final.csv', type=str, help="Path to input directory.")
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")

    parser.add_argument("--fr_stop_words_file", default="stopwords-fr.txt", type=str)
    parser.add_argument("--en_stop_words_file", default="stopwords-en.txt", type=str)

    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        required=True, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    parser.add_argument("--model", type=str, required=True, choices={"SVM","NB","LR"}, help="Model selected for training. (SVM stands for Support Vector Machine, NB stands for Naive Baysian, and LR stands for Logistic Regression.)")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--K", default=1, type=int, help="Selection of K for the metrics (Precision/Recall @ k).")
    parser.add_argument("--split_by_year", default=2017, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 128, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")
    parser.add_argument("--decay_sampling", action="store_true", help="Whether to sample examples using decay distribution.")
    parser.add_argument("--feature_dimension", type=int, default=728, help="Dimension of input features (of tf-idf) for classifier.")
    parser.add_argument("--keep_stop_words", action="store_true", help="Whether to keep stop words instead of removing them")
    parser.add_argument("--do_lemma", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to do lemmatization.")
    parser.add_argument("--max_iter", default=100, type=int, help="Max number of iterations for optimisation of classifier. (Especially for Logistic Regression)")


    args = parser.parse_args()

    if args.keep_stop_words:
        indice_stop_words = "sw"
    else:
        indice_stop_words = "nosw"
    if args.do_lemma:
        indice_do_lemma = "lem"
    else:
        indice_do_lemma = "nolem"
    output_path = '_'.join(["./models_tfidf", args.model, str(args.max_input_length), str(args.feature_dimension), indice_stop_words, indice_do_lemma, str(args.max_iter)])
    output_path = Path(output_path)

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")
            

    print("***** Creating stop words list *****")
    stemmer = SnowballStemmer(language[args.lang])

    if not args.keep_stop_words:
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
            from spacy.lang.fr.stop_words import STOP_WORDS as en_stop
            stop_words += list(en_stop)
            stop_words = list(set(stop_words))
        print(' Done! Number of stop words: ', len(stop_words))

    print("***** Creating training and testing data *****")
    ### READ DATA
    df = pd.read_csv(args.in_file, dtype=str)
    dict_target_secs = {'title': 'title',
                        'abstract': 'abs',
                        'claims': 'claims',
                        'description': 'desc'}
    target_sections = [dict_target_secs[s] for s in args.target_section]
    for sec in target_sections:
        df[sec] = df[sec].apply(str)
    df['text'] = df[target_sections].apply('. '.join, axis=1)
  
    year = args.split_by_year
    label = 'IPC' + str(args.pred_level)

    # ### Especially for decay sampling TODO
    # if decay_sampling:

    def tokenize_and_stem(text, lang=args.lang, stemmer=stemmer):
        tokens = [word for sent in sent_tokenize(text, language=language[lang]) for word in word_tokenize(sent, language=language[lang])]
        if args.max_input_length > 0:
            tokens = tokens[:args.max_input_length]
        #keep only words but not numbers or other symbols
        #filtered_tokens = []
        #for token in tokens:
        #    if re.search('[a-zA-Z]', token):
        #        filtered_tokens.append(token)

        #return tokens if don't do lemmatisation
        if not args.do_lemma:
            return ' '.join(tokens)

        #exclude stopwords from stemmed words
        if args.keep_stop_words:
            stems = [stemmer.stem(t) for t in tokens]
        else:
            stems = [stemmer.stem(t) for t in tokens if t not in stop_words]
        return ' '.join(stems)

    if args.do_train: 
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_train = df_train[['text', label]].dropna()
        df_train = df_train.sample(frac=1, random_state=666).reset_index(drop=True)
        X_train = df_train['text'].apply(tokenize_and_stem).values
        y_train = [label.split(',') for label in df_train[label].values]

        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        df_test = df_test[['text', label]].dropna()
        df_test = df_test.sample(frac=1, random_state=666).reset_index(drop=True)
        X_test = df_test['text'].apply(tokenize_and_stem).values
        y_test = [label.split(',') for label in df_test[label].values]

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
        pickle.dump((pipeline, mlb), open(os.path.join(output_path, f'SVC-{secs_name}-{args.split_by_year}-{args.pred_level}.pkl'), 'wb'))
        print('Model saved!')

    if args.do_test:
        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        df_test = df_test[['text', label]].dropna()
        df_test = df_test.sample(frac=1, random_state=666).reset_index(drop=True)
        X_test = df_test['text'].apply(tokenize_and_stem).values
        y_test = [label.split(',') for label in df_test[label].values]

        # read model
        secs_name = '_'.join(target_sections)
        pipeline, mlb = pickle.load(open(os.path.join(output_path, f'SVC-{secs_name}-{args.split_by_year}-{args.pred_level}.pkl'), 'rb'))
        print('Model loaded!')

        class_list = list(mlb.classes_)

        # Get predictions for test data
        print("***** Running prediction *****")
        y_test_pred = pipeline.predict_proba(X_test)

        K = args.K
        precisions = []
        recalls = []
        f1_at_ks = []
        predictions = []

        for i in tqdm(range(len(y_test))):
            true = y_test[i] 
            print(true)
            pred = [x for _,x in sorted(zip(y_test_pred[i], class_list), reverse=True)][:K]
            print(pred)
            predictions.append(pred)

            pre = precision(true, pred, K)
            rec = recall(true, pred, K)
            
            if pre != 0 and rec != 0:
                f1_at_k = 2 * pre * rec / (pre + rec)
            else:
                f1_at_k = 0.0

            precisions.append(pre)
            recalls.append(rec)
            f1_at_ks.append(f1_at_k)
            # print(predictions)

        res_df = pd.DataFrame({'true_labels': y_test, 
                               'predict_labels': predictions, 
                               'precision': precisions, 
                               'recall': recalls, 
                               'F1_at_k': f1_at_ks})

        res_df.to_csv(os.path.join(output_path, f'SVC-{secs_name}-{args.split_by_year}-{args.pred_level}.res'), index=False)

        print(res_df)
        print('Precision: ', res_df['precision'].mean(), 'Recall: ', res_df['recall'].mean(), 'F1 at k: ', res_df['F1_at_k'].mean())

if __name__ == "__main__":
    main()


