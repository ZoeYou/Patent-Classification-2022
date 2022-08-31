"""
script for creating datasets using textRank to reorder sentences.
"""

from summa.summarizer import summarize
from tqdm import tqdm
import os
import pandas as pd
import multiprocessing


IN_DIR = "./data/INPI_title_desc_2022_6_TextRank"   # change your inpur directory here

train_file = f"{IN_DIR}/train_raw_texts.txt"
test_file = f"{IN_DIR}/test_raw_texts.txt"

OUT_DIR = IN_DIR + "_TextRank"
out_train_file = f"{OUT_DIR}/train_raw_texts.txt"
out_test_file = f"{OUT_DIR}/test_raw_texts.txt"


try:
    os.mkdir(OUT_DIR)
except FileExistsError:
    pass

def text_summarize(line):
    title, desc = line.split("/SEP/")
    rtn = title + "/SEP/ " + " ".join(summarize(desc, language="french", split=True)).replace("\n"," ")
    return rtn


train_lines = pd.read_csv(train_file, header=None, dtype=str)
train_lines = train_lines[0].to_list()
test_lines = pd.read_csv(test_file, header=None, dtype=str)
test_lines = test_lines[0].to_list()

if "claims" in IN_DIR:
    train_res = []
    test_res = []
    
    for line in tqdm(train_lines):
        train_res.append(" ".join(summarize(line, language="french", split=True)).replace("\n", " "))
    for line in tqdm(test_lines):
        test_res.append(" ".join(summarize(line, language="french", split=True)).replace("\n", " "))
else:   # if the context is a combination of title + abstract or title + description and seperated by ' /SEP/ '
    pool = multiprocessing.Pool(processes=32)
    train_res = pool.map(text_summarize, train_lines, chunksize=1)
    test_res = pool.map(text_summarize, test_lines, chunksize=1)


df_train = pd.DataFrame({'train': train_res})
df_test = pd.DataFrame({'test': test_res})

df_train.to_csv(out_train_file, index=False, header=False)
df_test.to_csv(out_test_file, index=False, header=False)

