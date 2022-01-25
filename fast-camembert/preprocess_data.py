import os 
import pandas as pd
from collections import defaultdict


input_file = '../data/INPI/inpi_final.csv' 
target = 'description'
label_file = '../data/ipc-sections/labels_group_id.tsv'
IPC_level = '4'
split_year = 2017

# load labels
global label_list, label_map
with open(label_file, 'r') as in_f:
    lines = in_f.read().splitlines()[1:]
    label_list = [l.split('\t')[0] for l in lines]

label_map = {}
for (i, label) in enumerate(label_list):
    label_map[label] = i

# create label file
#with open('labels_group_id_' + IPC_level + '.csv', 'w') as out_f:
#    for lab in label_list:
#        out_f.write(lab + '\n')


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
    for j, label_index in enumerate(label_id_list):
        label_id[label_index] = 1
    return label_id

def create_df(df, file_name, target):
    dict_target = {'claims': 'claims',
                   'description': 'desc',
                   'title': 'title',
                   'abstract': 'abs'}
    column_names = list(label_map.keys())
    
    if target == 'title+abstract':
        text_list = (df[dict_target['title']] + ' ' + df[dict_target['abstract']]).to_list()
    else:
        text_list = df[dict_target[target]].to_list() 
    labels = df[f'IPC{IPC_level}'].apply(multi_hot).to_list()

    output_df = pd.DataFrame(labels, columns = column_names)
    output_df['text'] = text_list
    output_df.to_csv(file_name, index=False)



output_path = input_file.split('/')[-1].split('.')[0] + '_' + target + '_' + IPC_level
try:
    os.makedirs(output_path)
except FileExistsError:
    pass

# load data
df0 = pd.read_csv(input_file).dropna()
train_df0 = df0[df0['date'].apply(lambda x: x < int(f'{split_year}0000'))].reset_index(drop=True)
test_df0 = df0[df0['date'].apply(lambda x: x >= int(f'{split_year}0000'))].reset_index(drop=True)

create_df(train_df0, output_path+'/train.csv', target)
create_df(test_df0, output_path+'/test.csv', target)
