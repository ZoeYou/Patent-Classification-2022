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

def get_abrege(pubid):
    url = f'https://data.inpi.fr/brevets/EP{pubid}?q=EP{pubid}#EP{pubid}'
    
    headers = {
        'User-Agent': 'My User Agent 2.0',
        'From': 'your_email_address@gmail.com' 
    }

    s = requests.Session()
    try:
        strhtml = s.get(url, headers=headers)
    except:
        try:
            time.sleep(3)
            strhtml = s.get(url, headers=headers)
        except:
            return (pubid, "", "")
        
    soup=BeautifulSoup(strhtml.text,'lxml')
    divs = soup.find_all('div', {'class':'bloc-detail-notice'})
    
    kind = ''
    res = ''
    try:
        kind_div = divs[2]
        abrege_div = divs[8]

        KIND = kind_div.find('p', {'class': 'font-size-15 highlight-text'}).text.strip()
        abrege = abrege_div.find('p', {'class': 'font-size-15 highlight-text'}).text.strip()

        kind = KIND
        res = abrege
    except (IndexError, AttributeError):
        try:
            kind_div = divs[2]
            KIND = kind_div.find('p', {'class': 'font-size-15 highlight-text'}).text.strip()

            for div in divs:
                if div.find('p', {'class': 'mb-0 font-weight-300 font-size-13 inpi-light'}).text == 'Abrégé ': 
                    abrege_div = div
                    break

            abrege = abrege_div.find('p', {'class': 'font-size-15 highlight-text'}).text.strip()

            kind = KIND
            res = abrege
        except (IndexError, AttributeError):           
            print("NOT FIND: ", url)

    return (pubid, kind, res)

df_epo_kind_abrege = pd.read_csv('./EPO/epo_kind_abrege.csv')
epo_to_do = df_epo_kind_abrege[df_epo_kind_abrege.isnull().T.any()]
PUBLN_list = epo_to_do['pub_id'].apply(lambda x: str(x)).to_list()

df_already_done = pd.read_csv('EPO/epo_kind_abrege.txt', sep="\|\|\|", header=None, engine='python')
already_done = df_already_done.iloc[:, 0]
PUBLN_list = list(set(PUBLN_list) - set(already_done))
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
kinds = []
epos = []

with open('./EPO/epo_kind_abrege.txt', 'a') as out_f:
    for pubid, kind, resu in pool.imap_unordered(get_abrege, filestodo):
        pbar.update()
        out_f.write(str(pubid) + '|||' + kind + '|||' + resu + '\n')
        
print('with',threadn,'threads','it took',time.time()-ti,'seconds')
