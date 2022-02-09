import os 
import argparse
import pandas as pd


def multi_hot(row):
    # get list of label
    label_id_list = []
    example_label_list = row.split(',')
    for label_ in example_label_list:
        try:
            label_id_list.append(label_map[label_])
        except KeyError:
            continue # because this label does not exist in new version of classification system anymore!

    # convert to multi-hot vectors
    label_id = [0 for l in range(len(label_map))]
    for label_index in label_id_list:
        label_id[label_index] = 1
    return label_id

def create_df(df, file_name, target_sections, IPC_level, add_context_tokens, concatenate_sections):
    column_names = list(label_map.keys())


    if add_context_tokens:
        for s in target_sections:
            df[s] = df[s].apply(lambda x: f"<{s}> {x}")
    if concatenate_sections:
        text_list = []
        for s in target_sections:
            text_list.append(df[s].to_list())
    else:
        text_list = df[target_sections].apply('. '.join, axis=1).to_list()

    labels = df[f'IPC{IPC_level}'].apply(multi_hot).to_list()

    output_df = pd.DataFrame(labels, columns = column_names)
    output_df['text'] = text_list
    output_df.to_csv(file_name, index=False)


global dict_target
dict_target = {'claims': 'claims',
            'description': 'desc',
            'title': 'title',
            'abstract': 'abs'}

def main():
    parser = argparse.ArgumentParser()

    ##Required parameters
    parser.add_argument("--input_file",
                        default='../data/INPI/inpi_final.csv',
                        type=str,
                        help="original input file")

    parser.add_argument("--target", 
                        type=str,
                        action="append",
                        choices={"title","abstract","description","claims"},
                        help="The target section(s) of patent corpus.")

    parser.add_argument("--label_file",
                        default='../data/ipc-sections/20170101/labels_group_id_4.tsv',
                        type=str,
                        help="corresponding label file")

    parser.add_argument("--IPC_level",
                        default=4,
                        type=str,
                        help="target IPC classification level")

    parser.add_argument("--split_year",
                        default=2017,
                        type=str,
                        help="The year used to split training/testing data. (<split_year for training data, >=split_year for testing data.)")
    
    parser.add_argument("--add_context_tokens",
                        action="store_true",
                        help="Whether to add context special tokens at the beginning of corresponding section. e.g. <abstract>, <claims> (specifically for Bert for Patents)")

    parser.add_argument("--concat_sections",
                        action="store_true",
                        help="Whether to concatenate text from different sections as differernt data example.")

    args = parser.parse_args()

    # load labels
    global label_list, label_map
    with open(args.label_file, 'r') as in_f:
        lines = in_f.read().splitlines()[1:]
        label_list = [l.split('\t')[0] for l in lines]

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    # create label file if does not exist
    if not os.path.isfile('labels_group_id_' + args.IPC_level + '.csv'):
        with open('labels_group_id_' + args.IPC_level + '.csv', 'w') as out_f:
            for lab in label_list:
                out_f.write(lab + '\n')

    target_sections = [dict_target[s] for s in args.target]              
    sections_name = '_'.join(target_sections)

    if args.add_context_tokens:
        indice_ct = "ct"
    else:
        indice_ct = "noct"
    
    if args.concat_sections:
        indice_cs = "cs"
    else:
        indice_cs = "nocs"

    output_path = '_'.join[args.input_file.split('/')[-1].split('.')[0], sections_name, args.IPC_level, indice_ct, indice_cs]

    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass

    # load data
    df0 = pd.read_csv(args.input_file).dropna()
    train_df0 = df0[df0['date'].apply(lambda x: x < int(f'{args.split_year}0000'))].reset_index(drop=True)
    test_df0 = df0[df0['date'].apply(lambda x: x >= int(f'{args.split_year}0000'))].reset_index(drop=True)

    create_df(train_df0, output_path+'/train.csv', args.target, args.IPC_level, args.add_context_tokens, args.concat_sections)
    create_df(test_df0, output_path+'/test.csv', args.target, args.IPC_level, args.add_context_tokens, args.concat_sections)


if __name__ == "__main__":
    main()
