from pathlib import Path
import re


# enter your level of IPC/CPC here
level = 4   # {4, 6, 8}
DIR = './20210101'

def get_list(in_dir = './'):
    input_path = Path(in_dir)
    files_list = list(input_path.glob('*.txt'))
    files_list = sorted([str(file) for file in files_list])
    return files_list

def remove_replicates(labells, titlels):
    res = {}
    for i in range(len(labells)):
        label = labells[i]
        title = titlels[i]
        res[label] = title

    res_labels = list(res.keys())
    res_titles = list(res.values())
    return res_labels, res_titles

def reform_IPC8(line):
    code = " ".join([line[:4], re.sub(r'\w{4}[0]*', '', line[:-6]) + "/" +  "".join(line[-6:-4]) + "".join([c for c in line[-4:] if c!="0"])])
    return code


files_list = get_list(in_dir = DIR)
labels_list = []
titles_list = []

for f in files_list:
    with open(f, 'r') as in_f:
        lines = in_f.read().splitlines()
    if level == 4:
        labels = [line.split('\t')[0] for line in lines if len(line.split('\t')[0]) == 4]
        titles = [line.split('\t')[1] for line in lines if len(line.split('\t')[0]) == 4]

    if level == 6:
        labels = [line.split('\t')[0][:-6] for line in lines if len(line.split('\t')[0]) > 4 and line.split('\t')[0][-6:] == '000000']
        labels = [" ".join([line[:4], re.sub(r'\w{4}[0]*', '', line)]) for line in labels]

        titles = [line.split('\t')[-1] for line in lines if len(line.split('\t')[0]) > 4 and line.split('\t')[0][-6:] == '000000']

    if level == 8:
        labels0 = [line.split('\t')[0] for line in lines if len(line.split('\t')[0]) > 12 and line.split('\t')[0][-6:] != '000000']
        labels = [reform_IPC8(labels0[0])]

        for i in range(1, len(labels0)):
            curr = reform_IPC8(labels0[i])
            prev = labels[-1]

            curr_split = curr.split("/")
            prev_split = prev.split("/")
            
            while curr_split[0] == prev_split[0] and int(curr_split[1]) < int(prev_split[1]) and len(curr_split[1]) <= 8:   # TODO
                curr = curr + "0"
                curr_split = curr.split("/")
            labels.append(curr)

        titles = [line.split('\t')[-1] for line in lines if len(line.split('\t')[0]) > 12 and line.split('\t')[0][-6:] != '000000']
    
    labels_list.extend(labels)
    titles_list.extend(titles)

if level == 6 or level == 8:
    labels_list, titles_list = remove_replicates(labels_list, titles_list)

assert len(labels_list) == len(titles_list) == len(set(labels_list))

with open(f'{DIR}/labels_group_id_{level}.tsv', 'w') as out_f:
    out_f.write('id\ttitle\n')

    for l, t in zip(labels_list, titles_list):
        out_f.write(l + '\t' + t + '\n')


