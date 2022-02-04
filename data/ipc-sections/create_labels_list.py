from pathlib import Path

# enter your level of IPC/CPC here
level = 8

def get_list(in_dir = './'):
    input_path = Path(in_dir)
    files_list = list(input_path.glob('*.txt'))
    files_list = sorted([str(file) for file in files_list])
    return files_list

def remove_replicates(labells, titlels):
    res_labels = []
    res_titles = []

    for i in range(len(labells)):
        label = labells[i]
        if label not in res_labels:
            res_labels.append(label)
            res_titles.append(titlels[i])
    return res_labels, res_titles


files_list = get_list()
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
        titles = [line.split('\t')[-1][:-6] for line in lines if len(line.split('\t')[0]) > 4 and line.split('\t')[0][-6:] == '000000']
    if level == 8:
        labels = [line.split('\t')[0] for line in lines if len(line.split('\t')[0]) > 12 and line.split('\t')[0][-6:] != '000000']
        titles = [line.split('\t')[-1] for line in lines if len(line.split('\t')[0]) > 12 and line.split('\t')[0][-6:] != '000000']
    
    labels_list.extend(labels)
    titles_list.extend(titles)

if level == 6 or level == 8:
    labels_list, titles_list = remove_replicates(labels_list, titles_list)

print(labels_list)
print(len(labels_list))
print(len(set(labels_list)))

with open(f'labels_group_id_{level}.tsv', 'w') as out_f:
    out_f.write('id\ttitle\n')

    for l, t in zip(labels_list, titles_list):
        out_f.write(l + '\t' + t + '\n')


