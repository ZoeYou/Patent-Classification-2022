from pathlib import Path 
import collections
from tqdm import tqdm
import requests
import psutil
import time
from bs4 import BeautifulSoup
from multiprocessing import Pool
import multiprocessing
import pandas as pd


def get_list(in_dir = './epo_en', if_recent = True):
    input_path = Path(in_dir)
    files_list = list(input_path.glob('**/*.txt'))
    files_list = [str(file.stem) for file in files_list]
    if if_recent:
        recent_files = [f for f in files_list if int(f.split('_')[1]) >= 2000 ]
        files_list = recent_files
    return files_list


def get_IPCs(pubid):
    url = f'https://data.inpi.fr/brevets/FR{pubid}?q=FR{pubid}#FR{pubid}'

    headers = {
        'User-Agent': 'My User Agent 1.0',
        'From': 'zuoyouwk@gmail.com'
    }

    s = requests.Session()
    try:
        strhtml = s.get(url, headers=headers)
    except:
        try:
            time.sleep(3)
            strhtml = s.get(url, headers=headers)
        except:
            return (pubid, [])

    soup=BeautifulSoup(strhtml.text,'lxml')
    divs = soup.find_all('div', {'class':'bloc-detail-notice position-relative'})
    res = []

    try:
        div = divs[0]
        ps = div.find_all('p')
        #if ps[0].text == 'Classification CIB':
        for a in ps[1].find_all('a', {'class': 'link-blue'}):
            res.append(a.text.strip()) 
    except IndexError:
        print("NOT FIND: ", url)

    return (pubid, res)

"""
filename_list = ['epo_en', 'epo_de', 'epo_fr']

PUBLN_list = []

for file_name in filename_list:
    file_list = get_list(file_name)

    pubids = [f.split('_')[2] for f in file_list]
    PUBLN_list.extend(pubids)
PUBLN_list = set(PUBLN_list)

 
with open('epo_ipc.txt', 'r') as in_f:
    lines = in_f.read().splitlines()
    already_done = [line.split('\t')[0] for line in lines]
PUBLN_list = PUBLN_list - set(already_done)

df = pd.read_csv("epo_ipc.csv", header=None, names = ['pub_id', 'ipcs'])
"""
PUBLN_list = df[df.isnull().T.any()]['pub_id'].to_list()    # TODO: use your own list of publication codes
print(len(PUBLN_list))


# multiprocessing
ti = time.time()
filestodo = sorted(PUBLN_list)
#print('files to do:', filestodo)

pbar = tqdm(total=len(filestodo))

threadn = psutil.cpu_count() 
print(f'we have {threadn} precessors')

pool = multiprocessing.Pool(threadn)

results = []
epos = []

with open('epo_ipc.txt', 'w') as out_f:
    for pubid, resu in pool.imap_unordered(get_IPCs, filestodo):
        pbar.update()
        if isinstance(resu, list):
            resu = ','.join(resu) 
        out_f.write(str(pubid) + '\t' + resu + '\n')
print('with',threadn,'threads','it took',time.time()-ti,'seconds')
