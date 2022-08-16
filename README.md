# Patent Classification using Extreme Multi-label Learning: A Case Study of French Patents

This repository includes the codes and dataset for the implementation of patent classification on INPI's French patent documents.

# Dataset
We present a French Patents corpus, named  INPI-  
CLS, with IPC labels at all levels, and we test different models at the  
subclass and group levels on it. Our published French patents are  
extracted from the INPI (French National Institute of Industrial Property)  internal database, and contain all parts of  
patent texts (title, abstract, claims, description) published from 2002  
to 2021, each patent being annotated with all levels of IPC from sections to  
the subgroup labels.

The dataset is available in the following link: [https://drive.google.com/drive/folders/1tfBsUkQwIpwwgDyw28EOZctaiiJqZr1Q?usp=sharing](https://drive.google.com/drive/folders/1tfBsUkQwIpwwgDyw28EOZctaiiJqZr1Q?usp=sharing). Once you have obtained the data, unzip it and place it under the directory of `/data`.

To create dataset for models *AttentionXML* and *LightXML*, you have to run the code as follows:

```
cd XML/AttentionXML/
```
```
python create_dataset.py \
	 --in_file ../../data/inpi_new_final.csv \
	 --label_file ../../data/ipc-sections/20210101/labels_group_id_4.tsv \
	 --pred_level 4 \
	 --target_section title \
	 --target_section abstract \
	 --split_by_year 2020
```
You can create your own training and test datasets by adjusting the parameters:

- **label_file**: the label list used to filter removed labels (labels were defined in IPC system some years ago, but have been removed in the later versions).
- **pred_level**: the target prediction level of IPC (1,3,4,6,8 represent respectively IPC’s section, class, subclass, group, and subgroup level).
- **target_section**: the part of the patent content (title, abstract, claims, description) you want to use for your patent classification model; this parameter can have multiple inputs; the order will be considered.
- **split_by_year**: Patent documents published before this year will be split into training data, and files published after or including this year will be split into test data.
