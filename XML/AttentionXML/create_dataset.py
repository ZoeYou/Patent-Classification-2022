import csv, argparse, os, sys, scipy, collections
from email.policy import default
from pathlib import Path
import pandas as pd
from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
csv.field_size_limit(sys.maxsize)


def label_encoding(line, label_list, label_map):
    labels = line.split(",")
    labels = [l for l in labels if l in label_list]
    #return " ".join(labels)
    idx = [str(label_map[l]) for l in labels if l in label_list]
    if idx: 
        return " ".join(idx)

def label_preprocessor(line, IPC_level):
    if IPC_level == 4:
        line = list(dict.fromkeys([label[:4] for label in line.split(",")]))
        return ",".join(line)
    elif IPC_level == 6:
        line = list(dict.fromkeys([label.split("/")[0].replace(" ","") for label in line.split(",")]))
        return ",".join(line)
    else:
        return line

def filter_train_labels(line, train_labels):
    if not line:
        return line

    line_filtered = " ".join([l for l in line.split() if l in train_labels])
    # test
    if line != line_filtered:
        print("before:",line)
        print("after:",line_filtered)

    if line_filtered:
        return line_filtered


def create_v1(input_vectors,labels,name):
    """
    After converting each patent to a TF-IDF feature vector in "train_vectors"
    This function converts the feature vectors and labels of each patent to the right format:
    label1,label2,...labelk ft1:ft1_val ft2:ft2_val ft3:ft3_val .. ftd:ftd_val
    """
    to_remove = []
    printcounter = 0
    with open(name,'w') as fichier: 
        lines = []
        erreur=0 # count when a label doesn't have vectors due to the TF-IDF process.

        for i in range(input_vectors.shape[0]):  
            line = ""

            # add labels
            label_line = labels[i]
            
            label_line = ",".join(label_line.split(" "))
            line += (label_line) + " "

            # add the feature vector
            cx = scipy.sparse.coo_matrix(input_vectors[i][-1], input_vectors[i][:-1])
            unordered = dict(zip(cx.col, cx.data))
            ordered = collections.OrderedDict(sorted(unordered.items()))
            line += " ".join(["{0}:{1:.6f}".format(k,v) for k,v in ordered.items()])

            # append lines
            if len(ordered)==0:
                to_remove.append(i)
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
        fichier.write("\n".join(lines))
    return to_remove


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
    parser.add_argument("--in_file", default='../../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input directory.")
    parser.add_argument("--in_dir", type=str, help = "Input directory (especially for EPO dataset)")

    parser.add_argument("--label_file", default="../../data/ipc-sections/20210101/labels_group_id_4.tsv", type=str, help="path to the label file")
    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")

    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")

    args = parser.parse_args()  

    dict_target_secs = {'title': 'title',
                        'abstract': 'abs',
                        'claims': 'claims',
                        'description': 'desc'}
    

    if args.in_dir:
        if "USPTO" in args.in_dir:
            output_path = '_'.join(["./USPTO2M", str(args.pred_level)])
        else:
            output_path = '_'.join([args.in_dir.strip("/").split("/")[-1], str(args.pred_level)])
    else: 
        target_sections = [dict_target_secs[s] for s in args.target_section]
        secs_name = '_'.join(target_sections)
        year = args.split_by_year
        output_path = '_'.join(["./INPI", secs_name, str(year), str(args.pred_level)])
    output_path = Path(os.path.join("data", output_path))

    train_data_path = os.path.join(output_path,f'train_raw_texts.txt')
    test_data_path = os.path.join(output_path,f'test_raw_texts.txt')
    
    train_label_path = os.path.join(output_path, f'train_labels.txt')
    test_label_path = os.path.join(output_path, f'test_labels.txt')

    train_sparse_path = os.path.join(output_path,f'train_v1.txt')

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    if args.pred_level == 8:
        labels = [l.split("\t")[0] for l in open(args.label_file, "r").read().splitlines()[1:]]
    else:
        labels = [l.split("\t")[0].replace(" ","") for l in open(args.label_file, "r").read().splitlines()[1:]]


    if args.in_dir: 
        df_train = pd.read_csv(os.path.join(args.in_dir, "train.tsv"), dtype="object")
        df_test = pd.read_csv(os.path.join(args.in_dir, "test.tsv"), dtype="object")

        #label = 'IPC' + str(args.pred_level)
        label = "group_ids"
        df_train[label] = df_train[label].apply(lambda x: label_preprocessor(str(x), args.pred_level))
        df_test[label] = df_test[label].apply(lambda x: label_preprocessor(str(x), args.pred_level))
    
    else:
        # Import dataset
        df = pd.read_csv(args.in_file, dtype=object, engine="python").dropna()
        for sec in target_sections:
            df.loc[:, sec] = df[sec].apply(str)

        df.loc[:, 'text'] = df[target_sections].apply(('  /SEP/  ').join, axis=1)
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
    
        label = 'IPC' + str(args.pred_level)
        if args.pred_level == 6:
            df_train.loc[:, label] = df_train[label].apply(lambda x: label_preprocessor(str(x), args.pred_level))
            df_test.loc[:, label] = df_test[label].apply(lambda x: label_preprocessor(str(x), args.pred_level))

    df_train = df_train[[label, 'text']]
    df_test = df_test[[label, 'text']]

    label_map = dict(zip(labels, range(0, len(labels))))

    # remove "\n" in the texts
    df_train['text'] = df_train['text'].apply(lambda x: str(x).replace("\n", " ").replace("   ", " "))
    df_test['text'] = df_test['text'].apply(lambda x: str(x).replace("\n", " ").replace("   ", " "))

    ### transform labels into numbers
    # remove labels that do not exist in label file
    df_train[label] = df_train[label].apply(lambda x: label_encoding(str(x), labels, label_map))
    df_test[label] = df_test[label].apply(lambda x: label_encoding(str(x), labels, label_map))

    print(df_train)
    print(df_test)

    ## remove labels in test set that are not in train set
    #train_labels = [e for l in df_train[label].dropna().to_list() for e in l.split()]
    #df_test[label] = df_test[label].apply(lambda x: filter_train_labels(x, train_labels))

    # drop patent examples which do not have corresponding labels
    df_train = df_train.dropna().reset_index(drop=True)
    df_test = df_test.dropna().reset_index(drop=True)

    # create tf-idf vectorizer 
    featureVectorizer = TfidfVectorizer(max_features=10000, max_df = 0.9)
    X_train = featureVectorizer.fit_transform(df_train['text'].tolist())
    
    # save v1 file
    to_remove = create_v1(X_train, df_train[label].tolist(), train_sparse_path)
    if to_remove:
        # remove corresponding rows by index
        rows = df_train.index[to_remove]
        df_train.drop(rows, inplace=True)
    
    # save data files
    df_train['text'].to_csv(train_data_path, index=False, header=False)
    df_test['text'].to_csv(test_data_path, index=False, header=False)

    # save label files
    with open(train_label_path, "w") as out_f:
        out_f.write("\n".join(df_train[label].tolist()))
    with open(test_label_path, "w") as out_f:
        out_f.write("\n".join(df_test[label].tolist()))

if __name__ == "__main__":
    main()
