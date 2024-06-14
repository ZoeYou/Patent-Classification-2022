# Patent Classification using Extreme Multi-label Learning: A Case Study of French Patents

This repository includes the codes and dataset for the implementation of patent classification on INPI's French patent documents.

# Dataset
We present a French Patents corpus, named  INPI-CLS, with IPC labels at all levels, and we test different models at the subclass and group levels on it. Our published French patents are extracted from the INPI (French National Institute of Industrial Property)  internal database, and contain all parts of patent texts (title, abstract, claims, description) published from 2002 to 2021, each patent being annotated with all levels of IPC from sections to the subgroup labels.

The dataset is available in the following link: [https://drive.google.com/drive/folders/1aY1bLSpUshDbyzcUTTaCIs9bpErBdu8k?usp=sharing](https://drive.google.com/drive/folders/1aY1bLSpUshDbyzcUTTaCIs9bpErBdu8k?usp=sharing). Once you have obtained the data, unzip it and place it under the directory of `/data`.

Or you can download the dataset from huggingface: [https://huggingface.co/datasets/ZoeYou/INPI-CLS](https://huggingface.co/datasets/ZoeYou/INPI-CLS).


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


# Citation

If you use this work, please cite the following article:

You Zuo, Houda Mouzoun, Samir Ghamri Doudane, Kim Gerdes, Benoît Sagot. [Patent Classification using Extreme Multi-label Learning: A Case Study of French Patents](https://hal.archives-ouvertes.fr/hal-03850405v1). SIGIR 2022 - PatentSemTech workshop, Jul 2022, Madrid, Spain.

```
@inproceedings{zuo:hal-03850405,
  TITLE = {{Patent Classification using Extreme Multi-label Learning: A Case Study of French Patents}},
  AUTHOR = {Zuo, You and Mouzoun, Houda and Ghamri Doudane, Samir and Gerdes, Kim and Sagot, Beno{\^i}t},
  URL = {https://hal.archives-ouvertes.fr/hal-03850405},
  BOOKTITLE = {{SIGIR 2022 - PatentSemTech workshop}},
  ADDRESS = {Madrid, Spain},
  YEAR = {2022},
  MONTH = Jul,
  KEYWORDS = {IPC prediction ; Clustering and Classification ; Extreme Multi-label Learning ; French ; Patent},
  PDF = {https://hal.archives-ouvertes.fr/hal-03850405/file/PatentSemTech_2022___extended_abstract.pdf},
  HAL_ID = {hal-03850405},
  HAL_VERSION = {v1},
}
```

