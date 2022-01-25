from os import lseek
import sys
import torch
from wsgiref.simple_server import WSGIRequestHandler
from fast_bert.data_cls import BertDataBunch
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy
import logging

DATA_PATH = sys.argv[1]
LABEL_PATH = sys.argv[2]
MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
OUTPUT_DIR = DATA_PATH

with open(LABEL_PATH + '/labels.csv', 'r') as in_f:
    cols_label = in_f.read().splitlines()

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,    
                          tokenizer='camembert-base',
                          train_file='train.csv',
                          val_file='test.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col = cols_label,
                          batch_size_per_gpu=TRAIN_BATCH_SIZE,
                          max_seq_length=MAX_LEN,
                          multi_gpu=True,
                          multi_label=True,
                          model_type='camembert-base')


logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path='camembert-base',
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True,
						is_fp16=True,
						multi_label=False,
						logging_steps=50)

learner.fit(epochs=3,
			lr=5e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="adamw")

learner.save_model()

