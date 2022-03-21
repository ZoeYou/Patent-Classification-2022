import os
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
MAX_LEN = int(sys.argv[3]) # 128
n_epochs = int(sys.argv[4]) #4
TRAIN_BATCH_SIZE = int(sys.argv[5]) #64
model_name = sys.argv[6] #"camembert/ camembert-large"
try:
    model_path = sys.argv[7]
except:
    model_path = model_name


if "large" in model_name:
    indice_model_name = 'wiki'
else:
    indice_model_name = 'base'

OUTPUT_DIR = '_'.join([DATA_PATH, indice_model_name, str(MAX_LEN), str(n_epochs), str(TRAIN_BATCH_SIZE)])
try:
    os.makedirs(OUTPUT_DIR)
except FileExistsError:
    pass


with open(LABEL_PATH + '/labels.csv', 'r') as in_f:
    cols_label = in_f.read().splitlines()

databunch = BertDataBunch(DATA_PATH, LABEL_PATH,    
                          tokenizer=model_name,
                          train_file='train.csv',
                          val_file='test.csv',
                          label_file='labels.csv',
                          text_col='text',
                          label_col = cols_label,
                          batch_size_per_gpu=TRAIN_BATCH_SIZE,
                          max_seq_length=MAX_LEN,
                          multi_gpu=True,
                          multi_label=True,
                          model_type=model_name)

logger = logging.getLogger()
device_cuda = torch.device("cuda")
metrics = [{'name': 'accuracy', 'function': accuracy}]

if model_path != model_name:
    model_name = model_path

learner = BertLearner.from_pretrained_model(
						databunch,
						pretrained_path=model_name,
						metrics=metrics,
						device=device_cuda,
						logger=logger,
						output_dir=OUTPUT_DIR,
						finetuned_wgts_path=None,
						warmup_steps=500,
						multi_gpu=True,
						is_fp16=False,
						multi_label=True,
						logging_steps=50)

learner.fit(epochs=n_epochs,
			lr=3e-5,
			validate=True, 	# Evaluate the model after each epoch
			schedule_type="warmup_cosine",
			optimizer_type="adamw")

learner.save_model()

