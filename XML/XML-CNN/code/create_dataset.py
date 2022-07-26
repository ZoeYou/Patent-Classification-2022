import os, csv, sys, pickle
import argparse
import pandas as pd
import numpy as np
csv.field_size_limit(sys.maxsize)

def create_data(df_train, df_test, target_sections, IPC_level, label_map, filename, if_INPI=True):
    """
    train/test data are a list of document dicts with key 'text' for the plain text document and 'catgy' for the label.
    """
    if if_INPI:
        train_texts = df_train[target_sections].apply('. '.join, axis=1).to_list()
        test_texts = df_test[target_sections].apply('. '.join, axis=1).to_list()
    else:
        train_texts = df_train['text'].to_list()
        test_texts = df_test['text'].to_list()

    train_labels = [[label_map[l] for l in line.split(",") if l in label_map] for line in df_train[f'IPC{IPC_level}'].to_list()]
    test_labels = [[label_map[l] for l in line.split(",") if l in label_map] for line in df_test[f'IPC{IPC_level}'].to_list()]

    train = [{'text': text, 'catgy': labels} for text, labels in zip(train_texts, train_labels)]
    test = [{'text': text, 'catgy': labels} for text, labels in zip(test_texts, test_labels)]

    # save as pickle
    with open(filename, 'wb') as f:
        pickle.dump([train, test], f)


global dict_target
dict_target = {'claims': 'claims',
            'description': 'desc',
            'title': 'title',
            'abstract': 'abs'}


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

    ##Required parameters
    parser.add_argument("--input_file",
                        default='../../../data/INPI/new_extraction/output/inpi_new_final.csv',
                        type=str,
                        help="original input file")

    parser.add_argument("--target", 
                        type=str,
                        action="append",
                        choices={"title","abstract","description","claims"},
                        help="The target section(s) of patent corpus.")

    parser.add_argument("--in_dir",
                        type=str,
                        help="Path to the input dataset directory (including train.tsv and test.tsv)")

    parser.add_argument("--label_file",
                        default='../../../data/ipc-sections/20210101/labels_group_id_4.tsv',
                        type=str,
                        help="corresponding label file")

    parser.add_argument("--IPC_level",
                        default=4,
                        type=str,
                        help="target IPC classification level")

    parser.add_argument("--split_year",
                        default=2020,
                        type=str,
                        help="The year used to split training/testing data. (<split_year for training data, >=split_year for testing data.)")

    parser.add_argument("--out_dir",
                        default="../datasets",
                        type=str)

    args = parser.parse_args()

    # load labels
    global label_list
    with open(args.label_file, 'r') as in_f:
        lines = in_f.read().splitlines()[1:]
        label_list = [l.split('\t')[0] for l in lines]

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # create label file if does not exist
    if not os.path.isfile('../datasets/labels_group_id_' + str(args.IPC_level) + '.csv'):
        with open('labels_group_id_' + str(args.IPC_level) + '.csv', 'w') as out_f:
            for lab in label_list:
                out_f.write(lab + '\n')

    if args.in_dir:
        train_df0 = read_tsv(os.path.join(args.in_dir, 'train.tsv'), label_list, ipc_level=args.IPC_level)
        test_df0 = read_tsv(os.path.join(args.in_dir, 'test.tsv'), label_list, ipc_level=args.IPC_level)
        output_path = args.out_dir + '/' + '_'.join([args.in_dir.strip('/').split('/')[-1]]) + '.p'

        create_data(train_df0, test_df0, None, args.IPC_level, label_map, output_path, False)

    else:
        target_sections = [dict_target[s] for s in args.target]              
        sections_name = '_'.join(target_sections)
        output_path = args.out_dir + '/' + '_'.join([args.input_file.split('/')[-1].split('.')[0], sections_name, str(args.IPC_level)]) + '.p'

        # load data
        df0 = pd.read_csv(args.input_file).dropna()

        train_df0 = df0[df0['date'].apply(lambda x: x < int(f'{args.split_year}0000'))].reset_index(drop=True)
        test_df0 = df0[df0['date'].apply(lambda x: x >= int(f'{args.split_year}0000'))].reset_index(drop=True)

        create_data(train_df0, test_df0, target_sections, args.IPC_level, label_map, output_path)

if __name__ == "__main__":
    main()
