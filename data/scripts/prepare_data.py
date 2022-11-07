#!/usr/bin/env python
import os,re,csv,argparse,random
from itertools import compress

import pandas as pd
from wasabi import msg
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()

    ##Required parameters
    parser.add_argument("--lang",
                        default="en",
                        type=str,
                        help="language of target corpus")

    parser.add_argument("--input_dir",
                        default="EPO",
                        type=str,
                        help="Path to the input directory.")

    parser.add_argument("--out_dir",
                        default="EPO/epo_data",
                        type=str,
                        help="Path to output directory.")

    parser.add_argument("--start_year",
                        default=2015,
                        type=int,
                        help="Collect corpus from which year.")

    parser.add_argument("--end_year",
                        default=2020,
                        type=int,
                        help="Collect corpus til which year.")

    parser.add_argument("--split_year",
                        default=2020,
                        type=int,
                        help="The year used to split training/testing data. (<split_year for training data, >=split_year for testing data.)")

    parser.add_argument("--target", 
                       type=str,
                        action="append",
                        choices={"title","abstract","description","claims"},
                        help="The target section(s) of patent corpus.")

    parser.add_argument("--pubid_ipcs_file",
                        default="EPO/epo_ipc.csv",
                        type=str)

    parser.add_argument("--max_training_size",
                        type=int,
                        default=-1,
                        help="The maximum number of datasets in training set.")

    parser.add_argument("--max_text_length",
                        type=int,
                        default=-1,
                        help="Max length for output text. (split by ' ')")
    
    parser.add_argument("--add_context_tokens",
                        action="store_true",
                        help="Whether to add context special tokens at the beginning of corresponding section. e.g. [abstract], [claims] (specifically for Bert for Patents)"
    )

    parser.add_argument("--desc_token",
                        type=str,
                        default="[summary]",
                        choices={"[summary]", "[invention]"},
                        help="Whether to add context special tokens at the beginning of corresponding section. e.g. [abstract], [claims] (specifically for Bert for Patents)"
    )    

        
    args = parser.parse_args()

    output_path = Path(args.out_dir)
    input_path = Path(args.input_dir)
    lang = args.lang
    target = args.target

    if not input_path.exists():
        raise FileNotFoundError("Input directory does not exist.")
    
    if not output_path.exists():
        output_path.mkdir(parents=True)
        msg.good(f"Created output directory {args.out_dir}")

    files_list = []
    if lang == 'en':
        dir = os.path.join(args.input_dir,'epo_en')
    elif lang == 'fr':
        dir = os.path.join(args.input_dir,'epo_fr')
    elif lang == 'de':
        dir = os.path.join(args.input_dir,'epo_de')
    target_years = [str(y) for y in range(args.start_year, args.end_year+1)]

    input_dir = Path(dir)
    for y in target_years:  
        files_list.extend(list(input_dir.glob(f'{y}/*.txt')))
    
        if 'abstract' in target:
            abs_files_list = list(Path(os.path.join(args.input_dir,'INPI_results')).glob(f'{lang}_{y}*.txt'))
            files_list.extend(abs_files_list)

    if target == ['abstract']:
        files_list = [f for f in files_list if str(f).split('_')[-1].strip('.txt')[1] == '1' and lang in str(f)]
    elif target == ['description']:
        files_list = [f for f in files_list if str(f).split('_')[-1].strip('.txt')[2] == '1' and lang in str(f)]
    elif target == ['claims']:
        files_list = [f for f in files_list if str(f).split('_')[-1].strip('.txt')[3] == '1' and lang in str(f)]
    elif target == ['title']:
        files_list = [f for f in files_list if str(f).split('_')[-1].strip('.txt')[0] == '1' and lang in str(f)]
    elif target == ['title', 'description']:
        files_list = [f for f in files_list if str(f).split('_')[-1].strip('.txt')[0] == '1' and str(f).split('_')[-1].strip('.txt')[2] == '1' and lang in str(f)]    
    elif target == ['title','abstract'] or target == ['abstract', 'title']:
        files_list = [f for f in files_list if (str(f).split('_')[-1].strip('.txt')[:2] in ['11', '01'] or (str(f).split('_')[-1].strip('.txt')[0] == '1')) and lang in str(f)]
    else:
        raise ValueError('Unexpected target section or combination of sections!')
    random.shuffle(files_list)


    df_pubid_ipcs = pd.read_csv(args.pubid_ipcs_file, dtype={'pub_id': str, 'ipcs': str})
    dict_pubid_ipcs = dict(zip(df_pubid_ipcs.pub_id, df_pubid_ipcs.ipcs))

    columns = ['group_ids', 'id', 'date', 'text']
    dict_sec = {'abstract': 'ABSTR',
                'title': 'TITLE',
                'description': 'DESCR',
                'claims': 'CLAIM1'}

    dict_context_tokens = {'ABSTR': '[abstract]',
                           'CLAIM1': '[claim]',
                           'DESCR': args.desc_token}    # TODO or invention? need to be test
                        #    'TITLE': '[title]'   # NOT in vocabularies for Bert of Patents, ADDED BY ZY

    dict_context_tokens = defaultdict(lambda: "", dict_context_tokens)

    target_sections = [dict_sec[s] for s in args.target]              
    sections_name = '_'.join(target_sections)

    output_dir = f"epo-{lang}-{sections_name}-from-{args.start_year}to{args.end_year}-splitby{args.split_year}-{args.max_text_length}"
    if args.add_context_tokens:
        context_name = ""
        for s in target_sections:
            context_name += dict_context_tokens[s]
        output_dir = output_dir + '-' + context_name
        
    os.makedirs(output_path / output_dir, exist_ok=True)

    with open(output_path / output_dir / 'train.tsv', 'w', newline='') as out_f1, \
        open(output_path / output_dir / 'test.tsv', 'w', newline='') as out_f2:

        writer1 = csv.DictWriter(out_f1, columns)
        writer1.writeheader()
        writer2 = csv.DictWriter(out_f2, columns)
        writer2.writeheader()
        cnt = 0

        # for the combination of title + abstract
        if 'title' in args.target and 'abstract' in args.target:
            files_list = [str(f) for f in files_list]
            files_list2 = [f for f in files_list if f.split('_')[-1].strip('.txt')[:2] == '01']
            files_list3 = [f for f in files_list if f.split('_')[-1].strip('.txt')[:2] == '10']

            dict_abstract_title = {}
            for f2 in tqdm(files_list2):
                f3 = list(compress(files_list, ['_'.join(f2.split('/')[-1].split('_')[:4]) in f for f in files_list3]))
                if f3:
                    dict_abstract_title[Path(f2)] = Path(f3[0])

            for abs_file, title_file in dict_abstract_title.items():
                with title_file.open('r') as in_f1:
                    secs = in_f1.read().strip()
                    keys = re.findall('[\n\n]?(\w+) ::: ', secs)
                    vals = re.split('[\n\n]?\w+ ::: ', secs)[1:]

                    dict_kv = dict(zip(keys, vals))
                    res_dict = {}
                    
                with abs_file.open('r') as in_f2:
                    try:     
                        if args.add_context_tokens:
                            abs_content = in_f2.read().strip().split(' ::: ')
                            text = dict_kv['TITLE'] + '. [abstract] '
                            if len(abs_content) > 1:
                                text += abs_content[1]
                        else:
                            abs_content = in_f2.read().strip().split(' ::: ')
                            text = dict_kv['TITLE'] + '. '
                            if len(abs_content) > 1:
                                text += abs_content[1]

                    except KeyError:    # if no title then just abstract
                        if args.add_context_tokens:
                            abs_content = in_f2.read().strip().split(' ::: ')
                            text = '[abstract] ' 
                            if len(abs_content) > 1:
                                text += abs_content[1]
                        else:
                            abs_content = in_f2.read().strip().split(' ::: ')
                            if len(abs_content) > 1:
                                text = abs_content[1]
                            else:
                                text = ""

                    if len(text) > 10:
                        res_dict['text'] = text 
                    else:
                        continue

                res_dict['id'] = str(title_file).split('_')[-3].strip()
                res_dict['group_ids'] = dict_pubid_ipcs[res_dict['id']].strip()

                year_of_f = str(title_file).split('_')[-4]
                res_dict['date'] = f'{year_of_f}-01-01'

                if int(res_dict['date'][:4]) >= args.split_year:
                    writer2.writerow(res_dict)
                else:
                    writer1.writerow(res_dict)
                    cnt += 1
                    if args.max_training_size != -1 and cnt >= args.max_training_size:
                        return

            files_list = [Path(f) for f in files_list if f.split('_')[-1].strip('.txt')[:2] == '11']
        

        for pat_file in tqdm(files_list):
            with pat_file.open('r') as in_f:

                secs = in_f.read().strip()
                keys = re.findall('[\n\n]?(\w+) ::: ', secs)
                vals = re.split('[\n\n]?\w+ ::: ', secs)[1:]

                dict_kv = dict(zip(keys, vals))
                res_dict = {}

                if args.add_context_tokens:
                    text = '. '.join([dict_context_tokens[k] + ' ' + v for k, v in zip(keys, vals) if k in target_sections]).strip()
                else:
                    # /SEP/ added for LightXML
                    text = ' /SEP/ '.join([v.strip() for k, v in zip(keys, vals) if k in target_sections])

                if len(text) > 10:
                    res_dict['text'] = ' '.join(text.split(' ')[:args.max_text_length])
                else:
                    continue
                res_dict['id'] = str(pat_file).split('_')[-3].strip()
                res_dict['group_ids'] = str(dict_pubid_ipcs[res_dict['id']]).strip()

                year_of_f = str(pat_file).split('_')[-4]
                res_dict['date'] = f'{year_of_f}-01-01'

                if int(res_dict['date'][:4]) >= args.split_year:
                    writer2.writerow(res_dict)
                else:
                    writer1.writerow(res_dict)
                    cnt += 1
                    if args.max_training_size != -1 and cnt >= args.max_training_size:
                        return
        

if __name__ == "__main__":
    main()
