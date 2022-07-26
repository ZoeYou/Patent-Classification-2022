import csv, argparse, os, sys, collections
from pathlib import Path
import pandas as pd
import scipy.sparse
import omikuji, inspect
from tqdm import tqdm

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
    parser.add_argument("--in_file", default='../../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input directory.")
    parser.add_argument("--in_dir", type=str, help = "Input directory (especially for EPO dataset)")
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")

    parser.add_argument("--fr_stop_words_file", default="../../baselines/stopwords-fr.txt", type=str)
    parser.add_argument("--en_stop_words_file", default="../../baselines/stopwords-en.txt", type=str)
    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")

    parser.add_argument("--feature_dimension", type=int, default=10000, help="Dimension of input features (of tf-idf) for classifier.")
    parser.add_argument("--label_file", type=str, default="../../data/ipc-sections/20210101/labels_group_id_4.tsv", help="Path to the label file.")

    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--start_year", default=2000, type=int, help="The start year for the training data.")
    parser.add_argument("--max_input_length", default = 1000, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")
    parser.add_argument("--remove_stop_words", default=True, type=lambda x: (str(x).lower() == 'true'), help="Whether to remove stop words from input text") 
    parser.add_argument("--n_trees", default=3, type=int, help="Number of tree.")
    parser.add_argument("--min_branch_size", default=100, type=int, help="Number of labels below which no further clustering & branching is done.")
    parser.add_argument("--feature_type", default="tfidf", type=str, choices={"tfidf", "bow"}, help="Type of feature as input.")
    parser.add_argument("--beam_size", default=10, type=int, help="Beam size for beam search")
   
    args = parser.parse_args()  
    if args.remove_stop_words:
        indice_sw = "nosw"
    else:
        indice_sw = "withsw"

    if args.in_dir:
        output_path = '_'.join(["./parabel", args.in_dir.split("/")[-1], args.feature_type, str(args.pred_level), indice_sw, "NbTrees-" + str(args.n_trees), "MinBranchSize-" + str(args.min_branch_size), "BeamSize-" + str(args.beam_size)])
    else: 
        output_path = '_'.join(["./parabel_INPI", args.feature_type, str(args.pred_level), indice_sw, "StartYear-" + str(args.start_year), "NbTrees-" + str(args.n_trees), "MinBranchSize-" + str(args.min_branch_size), "BeamSize-" + str(args.beam_size)])
    output_path = Path(output_path)

    dict_target_secs = {'title': 'title',
                        'abstract': 'abs',
                        'claims': 'claims',
                        'description': 'desc'}

    if args.in_dir:
        train_path = os.path.join(output_path, 'train.txt')
        test_path = os.path.join(output_path, 'test.txt')
        model_path = os.path.join(output_path, 'model')
    else:
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

    with open(args.label_file, "r") as in_f:
        lines = in_f.read().splitlines()
        labels = [l.split("\t")[0] for l in lines][1:]

    if args.in_dir:  
        df_train = pd.read_csv(os.path.join(args.in_dir, "train.tsv"))  
        df_test = pd.read_csv(os.path.join(args.in_dir, "test.tsv"))
        label = "group_ids"
        if args.pred_level == 4:
            df_train[label] = df_train[label].apply(lambda x: ",".join(list(set([l[:4] for l in str(x).split(",")]))))
            df_test[label] = df_test[label].apply(lambda x: ",".join(list(set([l[:4] for l in str(x).split(",")]))))
        else:
            pass #TODO
    else:
        # Import dataset
        df = pd.read_csv(args.in_file, dtype=object, engine="python")#.dropna()
        for sec in target_sections:
            df.loc[:,sec] = df[sec].apply(str)

        df.loc[:, 'text'] = df[target_sections].apply('. '.join, axis=1)
        if args.max_input_length > 0:
            df.loc[:,'text'] = df['text'].apply(lambda x: ' '.join(x.split(' ')[:args.max_input_length]))
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=args.start_year)]
        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        label = 'IPC' + str(args.pred_level) 

    df_train.loc[:,label] = df_train[label].apply(lambda x: ",".join([l.replace(" ","") for l in str(x).split(",") if l in labels]))
    df_test.loc[:,label] = df_test[label].apply(lambda x: ",".join([l.replace(" ","") for l in str(x).split(",") if l in labels]))
    df_train = df_train[[label, 'text']].dropna()
    df_test = df_test[[label, 'text']].dropna()

    print(df_train)
    print(df_test)

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
    

    # transform input data
    X_train = df_train['text'].to_list()
    X_test = df_test['text'].to_list()

    print(X_train[:5])
    print(X_test[:5])

    if args.feature_type == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        featureVectorizer = TfidfVectorizer(max_features=args.feature_dimension)
    elif args.feature_type == "bow":
        from sklearn.feature_extraction.text import CountVectorizer
        featureVectorizer = CountVectorizer(max_features=args.feature_dimension)
    X_train = featureVectorizer.fit_transform(X_train.copy())
    X_test = featureVectorizer.transform(X_test.copy())

    # transform output data
    y_train = [[e for e in label.split(',') if e] for label in df_train[label].values]
    y_test = [[e for e in label.split(',') if e] for label in df_test[label].values]

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

    class_list = list(mlb.classes_)


    if args.do_train:
        hyper_param = omikuji.Model.default_hyper_param()
        ## This function allows us to visualize the parameters we can modify
        #print(inspect.getmembers(omikuji.Model.default_hyper_param(), lambda a:not(inspect.isroutine(a))))
        hyper_param.n_trees = args.n_trees
        hyper_param.min_branch_size = args.min_branch_size
        
        model = omikuji.Model.train_on_data(train_path, hyper_param, n_threads=3)
        model.save(model_path)
        

    if args.do_test:
        from collections import defaultdict

        model = omikuji.Model.load(model_path)
        y_true = [label.split(',') for label in df_test[label].values]

        y_pred = []
        for test in X_test:
            cx = scipy.sparse.coo_matrix(test[-1], test[:-1])
            input_test = [(c, v) for c,v in zip(cx.col, cx.data)]
            y_pred.append(model.predict(input_test, top_k = len(class_list), beam_size = args.beam_size))
        
        # save prediction scores
        y_pred_score = defaultdict(list)
        for line in y_pred:
            print(len(line))
            for label, score in line:
                y_pred_score[class_list[label-1]].append(score)
        
        #for k, v in y_pred_score.items():
        #    print(k)
        #    print(len(v))
        #    print("----------------------")


        y_pred = [[class_list[label-1] for label, score in line] for line in y_pred]

        pre_n_1 = []
        pre_n_3 = []
        pre_n_5 = []

        predictions = []

        for i in tqdm(range(len(y_true))):
            true = y_true[i]
            pred = y_pred[i]
            predictions.append(pred)

            pre_1 = precision(true, pred, 1)
            pre_3 = precision(true, pred, 3)
            pre_5 = precision(true, pred, 5)

            pre_n_1.append(pre_1)
            pre_n_3.append(pre_3)
            pre_n_5.append(pre_5)
        
        res_df = pd.DataFrame({'true_labels': y_true, 
                               'predict_labels': predictions, 
                               'precision@1': pre_n_1, 
                               'precision@3': pre_n_3,
                               'precision@5': pre_n_5                           
                               })
        score_df = pd.DataFrame.from_dict(y_pred_score)

        try:
            res_df.to_csv(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.res'), index=False)
            score_df.to_csv(os.path.join(output_path, f'{secs_name}-{args.split_by_year}-{args.pred_level}.score'), index=False)
        except UnboundLocalError:
            res_df.to_csv(os.path.join(output_path, f'{args.pred_level}.res'), index=False)
            score_df.to_csv(os.path.join(output_path, f'{args.pred_level}.score'), index=False)
        print(res_df)
        
        for col in ["precision@1", "precision@3", "precision@5"]:
            print(col + ": ", res_df[col].mean())
  
        #Precision_at_k, Recall_at_k = eval(y_pred, y_true, args.K)
        #F1_at_k = 2 * Precision_at_k * Recall_at_k / (Precision_at_k + Recall_at_k)
        

        #print(f"Number of lines in testing set : {len(y_true)}")
        #print(f"Precision at {args.K} : {Precision_at_k}")
        #print(f"Recall at {args.K} : {Recall_at_k}")
        #print(f"F1 at {args.K} : {F1_at_k}")


if __name__ == "__main__":
    main()
