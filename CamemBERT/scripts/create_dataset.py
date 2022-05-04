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
    parser.add_argument("--out_dir", type=str, default='./datasets', help="Path to output directory to save models")
    
    parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")
    parser.add_argument("--label_file", default='../data/ipc-sections/20210101/labels_group_id_4.tsv', type=str)

    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    
    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (especially for INPI-CLS data <split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 500, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")

    args = parser.parse_args()


    print("***** Reading standard label file *****")
    if args.pred_level == 1:
        labels_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    else:
        with open(args.label_file, 'r') as in_f:
            lines = in_f.read().splitlines()[1:]
        labels_list = [l.split('\t')[0] for l in lines]

    year = args.split_by_year
    label = 'IPC' + str(args.pred_level)

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
    df_test = df_test.sample(frac=1, random_state=666).reset_index(drop=True)

    train_dict = {}
    test_dict = {}

    for l in labels_list:
        train_dict[l] = df_train[label].apply(lambda x: 1 if l in x.split(",") else 0).to_list()
        test_dict[l] = df_test[label].apply(lambda x: 1 if l in x.split(",") else 0).to_list()
    attributes_train = pd.DataFrame.from_dict(train_dict)
    attributes_test = pd.DataFrame.from_dict(test_dict)

    df_train = pd.concat([df_train['text'], attributes_train], axis=1)
    df_test = pd.concat([df_test['text'], attributes_test], axis=1)

    print(df_train.head())
    print(df_test.head())

    if args.in_dir:
        data_name = args.in_dir.strip("/").split("/")[-1]
    else:
        data_name = "_".join(["INPI", "_".join(target_sections), str(args.pred_level)])

    output_path = os.path.join(args.out_dir,data_name)

    train_path = os.path.join(output_path, "train.csv")
    test_path = os.path.join(output_path, "test.csv")

    output_path = Path(output_path)

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)

if __name__ == "__main__":
    main()


