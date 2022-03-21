import csv, argparse, os, sys, collections
from pathlib import Path
import pandas as pd
import scipy.sparse
import omikuji, inspect

from nltk import word_tokenize, sent_tokenize
csv.field_size_limit(sys.maxsize)

from sklearn.preprocessing import MultiLabelBinarizer


def create_dataset(input_vectors,labels,nb_labels,name):
    """
    After converting each patent to a TF-IDF feature vector in "train_vectors"
    This function converts the feature vectors and labels of each patent to the right format:
    label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    """
    printcounter = 0
    with open(name,'w') as fichier: 
        lines = []
        erreur=0 # count when a label doesn't have vectors due to the TF-IDF process.

        for i in range(input_vectors.shape[0]):  
            line = ""

            # add labels
            label_line = labels[i]
            
            label_line = ",".join([str(i+1) for i in range(len(label_line)) if int(label_line[i]) == 1])
            line += (label_line) + " "

            # add the feature vector
            cx = scipy.sparse.coo_matrix(input_vectors[i][-1], input_vectors[i][:-1])
            unordered = dict(zip(cx.col, cx.data))
            ordered = collections.OrderedDict(sorted(unordered.items()))
            line += " ".join(["{0}:{1:.6f}".format(k,v) for k,v in ordered.items()])

            # append lines
            if len(ordered)==0:
                erreur+=1 #Count errors (examples with no features)
            else:
                lines.append(line)
            
            #keep updated with progress
            printcounter+=1
            if printcounter==100:
                printcounter=0
                progress = i/len(labels) * 100
                text = "\rPercent: {0:.2f}% {1} erreur:{2}".format(progress,i,erreur)
                sys.stdout.write(text)
                sys.stdout.flush()
        print("Number of examples without features:", erreur)

        fichier.write('{} {} {}\n'.format(len(lines),input_vectors.shape[1], nb_labels))
        fichier.write("\n".join(lines))
        
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input directory.")
    parser.add_argument("--in_dir", type=str, help = "Input directory (specially for EPO dataset)")
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")

    parser.add_argument("--fr_stop_words_file", default="../baselines/stopwords-fr.txt", type=str)
    parser.add_argument("--en_stop_words_file", default="../baselines/stopwords-en.txt", type=str)
    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str,
                        required=True, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--feature_dimension", type=int, default=10000, help="Dimension of input features (of tf-idf) for classifier.")
    parser.add_argument("--K", default=1, type=int, help="Selection of K for the metrics (Precision/Recall @ k).")
    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 1000, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")
    parser.add_argument("--remove_stop_words", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to remove stop words from input text") 
    parser.add_argument("--depth", default=3, type=int, help="The depth of the tree.")
    parser.add_argument("--feature_type", default="tfidf", type=str, choices={"tfidf", "bow"}, help="Type of feature as input.")
   
    args = parser.parse_args()  
    if args.remove_stop_words:
        indice_sw = "nosw"
    else:
        indice_sw = "withsw"

    if args.in_dir:
        output_path = '_'.join(["./parabel_EPO", args.feature_type, str(args.pred_level), indice_sw, str(args.depth)])
    else: 
        output_path = '_'.join(["./parabel_INPI", args.feature_type, str(args.pred_level), indice_sw, str(args.depth)])
    output_path = Path(output_path)

    dict_target_secs = {'title': 'title',
                        'abstract': 'abs',
                        'claims': 'claims',
                        'description': 'desc'}
    target_sections = [dict_target_secs[s] for s in args.target_section]
    secs_name = '_'.join(target_sections)
    year = args.split_by_year

    train_path = os.path.join(output_path,f'{secs_name}-train.txt')
    test_path = os.path.join(output_path,f'{secs_name}-test.txt')
    model_path = os.path.join(output_path, f'{secs_name}_model')
    

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    if args.in_dir:  
        df_train = pd.read_csv(os.path.join(args.in_dir, "train.tsv"))  # TODO
        df_test = pd.read_csv(os.path.join(args.in_dir, "test.tsv"))    # TODO csvreader
    else:
        # Import dataset
        df = pd.read_csv(args.in_file, dtype=object, engine="python").dropna()
        for sec in target_sections:
            df.loc[:,sec] = df[sec].apply(str)

        df.loc[:, 'text'] = df[target_sections].apply('. '.join, axis=1)
        if args.max_input_length > 0:
            df.loc[:,'text'] = df['text'].apply(lambda x: ' '.join(x.split(' ')[:args.max_input_length]))
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]

    if args.remove_stop_words:
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

        def rm_sw(text, lang=args.lang): 
            language = {'fr': 'french', 'en': 'english'}
            tokens = [word for sent in sent_tokenize(text, language=language[lang]) for word in word_tokenize(sent, language=language[lang])]
            #exclude stopwords from stemmed words
            stems = [t for t in tokens if t not in stop_words]
            return ' '.join(stems)
        df_train.loc[:,'text'] = df_train['text'].apply(rm_sw)
        df_test.loc[:,'text'] = df_test['text'].apply(rm_sw)
    
    label = 'IPC' + str(args.pred_level) 
    df_train = df_train[[label, 'text']]
    df_test = df_test[[label, 'text']]

    # transform input data
    X_train = df_train['text'].to_list()
    X_test = df_test['text'].to_list()
    if args.feature_type == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        featureVectorizer = TfidfVectorizer(max_features=args.feature_dimension)
    elif args.feature_type == "bow":
        from sklearn.feature_extraction.text import CountVectorizer
        featureVectorizer = CountVectorizer(max_features=args.feature_dimension)
    X_train = featureVectorizer.fit_transform(X_train.copy())
    X_test = featureVectorizer.transform(X_test.copy())

    # transform output data
    y_train = [label.split(',') for label in df_train[label].values]
    y_test = [label.split(',') for label in df_test[label].values]

    ### Label encoding
    mlb = MultiLabelBinarizer()
    mlb.fit(y_train+y_test)
    print(mlb.classes_)
    print(len(mlb.classes_))

    y_train = mlb.transform(y_train)
    y_test = mlb.transform(y_test)
    print('Datasets Converted!')    

    create_dataset(X_train, y_train, len(mlb.classes_), train_path)
    create_dataset(X_test, y_test, len(mlb.classes_), test_path)


    if args.do_train:
        hyper = omikuji.Model.default_hyper_param()
        ## This function allows us to visualize the parameters we can modify
        print(inspect.getmembers(omikuji.Model.default_hyper_param(), lambda a:not(inspect.isroutine(a))))
        asdf 
        
        model = omikuji.Model.train_on_data(train_path, hyper, n_threads=3)
        model.save(model_path)
        

    if args.do_test:
        model = omikuji.Model.load(model_path)
        y_true = [label.split(',') for label in df_test[label].values]

        y_pred = []
        for test in X_test:
            cx = scipy.sparse.coo_matrix(test[-1], test[:-1])
            input_test = [(c, v) for c,v in zip(cx.col, cx.data)]
            y_pred.append(model.predict(input_test, top_k = args.K))
        y_pred = [[mlb.classes_[label-1] for label, score in line] for line in y_pred]
   
        Precision_at_k, Recall_at_k = eval(y_pred, y_true, args.K)
        F1_at_k = 2 * Precision_at_k * Recall_at_k / (Precision_at_k + Recall_at_k)
        

        print(f"Number of lines in testing set : {len(y_true)}")
        print(f"Precision at {args.K} : {Precision_at_k}")
        print(f"Recall at {args.K} : {Recall_at_k}")
        print(f"F1 at {args.K} : {F1_at_k}")


if __name__ == "__main__":
    main()
