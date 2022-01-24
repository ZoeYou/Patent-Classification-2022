import time, csv, fasttext, tokenizers, argparse, os
from pathlib import Path

import numpy as np, pandas as pd
from gensim.utils import simple_preprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/INPI/inpi_final.csv', type=str, help="Path to input directory.")
    # parser.add_argument("--lang", default='fr', type=str, choices={"fr", "en", "de"}, help="Language of the input text.")
    # parser.add_argument("--fr_stop_words_file", default="stopwords-fr.txt", type=str)
    # parser.add_argument("--en_stop_words_file", default="stopwords-en.txt", type=str)
    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--target_section", 
                        type=str, 
                        required=True, 
                        choices={"title","abstract","description","claims"}, 
                        action = "append",
                        help="Target section(s) to be used to train the model.")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_test", action="store_true")
    parser.add_argument("--K", default=1, type=int, help="Selection of K for the metrics (Precision/Recall @ k).")
    parser.add_argument("--split_by_year", default=2017, type=int, help="The year used to split data. (<split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--max_input_length", default = 128, type=int, help="Max input sequence length. (-1 refers to input sequence without limit.)")
    # parser.add_argument("--decay_sampling", action="store_true", help="Whether to sample examples using decay distribution.")
    # parser.add_argument("--feature_dimension", type=str, default=728, help="Dimension of input features (of tf-idf) for classifier.")

    args = parser.parse_args()

    output_path = '_'.join(["./models_fastText", str(args.max_input_length)])
    output_path = Path(output_path)

    dict_target_secs = {'title': 'title',
                        'abstract': 'abs',
                        'claims': 'claims',
                        'description': 'desc'}
    target_sections = [dict_target_secs[s] for s in args.target_section]
    secs_name = '_'.join(target_sections)
    year = args.split_by_year

    train_path = os.path.join(output_path,f'{secs_name}-{year}-train.txt')
    test_path = os.path.join(output_path,f'{secs_name}-{year}-test.txt')
    
    if Path(train_path).exists() and Path(test_path).exists():
        pass
    else:
        if not output_path.exists():
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")

        # Import dataset
        df = pd.read_csv(args.in_file, dtype=object)
        for sec in target_sections:
            df[sec] = df[sec].apply(str)

        df['text'] = df[target_sections].apply('. '.join, axis=1)
        
        label = 'IPC' + str(args.pred_level)
        df[label] = df[label].apply(lambda x: ' '.join(['__label__' + l for l in x.split(',')]))
        
        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_train = df_train[[label, 'text']].dropna()

        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        df_test = df_test[[label, 'text']].dropna() 

        with open(train_path, 'w') as in_f1:
            in_f1.write('\n'.join(df_train[label] + ' ' + df_train['text']))
        with open(test_path, 'w') as in_f2:
            in_f2.write('\n'.join(df_test[label] + ' ' + df_test['text']))       
    
    if args.do_train:
        model = fasttext.train_supervised(train_path)
        model.save_model(os.path.join(output_path, f'fastText-{secs_name}-{args.max_input_length}-{args.split_by_year}-{args.pred_level}.bin'))
    if args.do_test:
        model = fasttext.load_model(os.path.join(output_path, f'fastText-{secs_name}-{args.max_input_length}-{args.split_by_year}-{args.pred_level}.bin'))
        
        pred_res = model.test(test_path, k=args.K)
        print(pred_res)

if __name__ == "__main__":
    main()
