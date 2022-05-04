import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from transformers import AutoModel, AdamW, get_cosine_schedule_with_warmup
import torch.nn as nn
import math
from torchmetrics.functional.classification import auroc
import torch.nn.functional as F
from tqdm import tqdm


class PMLC_Dataset(Dataset):

  def __init__(self, data_path, tokenizer, attributes, max_token_len: int = 128, sample = None):
    self.data_path = data_path
    self.tokenizer = tokenizer
    self.attributes = attributes
    self.max_token_len = max_token_len
    self.sample = sample
    self._prepare_data()

  def _prepare_data(self):
    data = pd.read_csv(self.data_path)
    # if self.sample is not None:
    #   unhealthy = data.loc[data[attributes].sum(axis=1) > 0]
    #   clean = data.loc[data[attributes].sum(axis=1) == 0]
    #   self.data = pd.concat([unhealthy, clean.sample(self.sample, random_state=7)])
    # else:
    #   self.data = data
    self.data = data
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    item = self.data.iloc[index]
    text = str(item.text)
    attributes = torch.FloatTensor(item[self.attributes])   # lists of 0 and 1
    tokens = self.tokenizer.encode_plus(text,
                                        add_special_tokens=True,
                                        return_tensors='pt',
                                        truncation=True,
                                        padding='max_length',
                                        max_length=self.max_token_len,
                                        return_attention_mask = True)
    return {'input_ids': tokens.input_ids.flatten(), 'attention_mask': tokens.attention_mask.flatten(), 'labels': attributes}

class PMLC_Data_Module(pl.LightningDataModule):

  def __init__(self, train_path, val_path, attributes, batch_size: int = 64, max_token_length: int = 128,  model_name='camembert-base'):
    super().__init__()
    self.train_path = train_path
    self.val_path = val_path
    self.attributes = attributes
    self.batch_size = batch_size
    self.max_token_length = max_token_length
    self.model_name = model_name
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

  def setup(self, stage = None):
    if stage in (None, "fit"):
      self.train_dataset = PMLC_Dataset(self.train_path, attributes=self.attributes, tokenizer=self.tokenizer)
      self.val_dataset = PMLC_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)
    if stage == 'predict':
      self.val_dataset = PMLC_Dataset(self.val_path, attributes=self.attributes, tokenizer=self.tokenizer, sample=None)

  def train_dataloader(self):
    return DataLoader(self.train_dataset, batch_size = self.batch_size, num_workers=4, shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

  def predict_dataloader(self):
    return DataLoader(self.val_dataset, batch_size = self.batch_size, num_workers=4, shuffle=False)

class PMLC_Classifier(pl.LightningModule):

  def __init__(self, config: dict):
    super().__init__()
    self.config = config
    self.pretrained_model = AutoModel.from_pretrained(config['model_name'], return_dict = True)
    self.hidden = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
    self.classifier = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.config['n_labels'])
    torch.nn.init.xavier_uniform_(self.classifier.weight)
    self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
    self.dropout = nn.Dropout()
    
  def forward(self, input_ids, attention_mask, labels=None):
    # roberta layer
    output = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
    pooled_output = torch.mean(output.last_hidden_state, 1)
    # final logits
    pooled_output = self.dropout(pooled_output)
    pooled_output = self.hidden(pooled_output)
    pooled_output = F.relu(pooled_output)
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    # calculate loss
    loss = 0
    if labels is not None:
      loss = self.loss_func(logits.view(-1, self.config['n_labels']), labels.view(-1, self.config['n_labels']))
    return loss, logits

  def training_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("train loss ", loss, prog_bar = True, logger=True)
    return {"loss":loss, "predictions":outputs, "labels": batch["labels"]}

  def validation_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    self.log("validation loss ", loss, prog_bar = True, logger=True)
    return {"val_loss": loss, "predictions":outputs, "labels": batch["labels"]}

  def predict_step(self, batch, batch_index):
    loss, outputs = self(**batch)
    return outputs

  def configure_optimizers(self):
    optimizer = AdamW(self.parameters(), lr=self.config['lr'], weight_decay=self.config['weight_decay'])
    total_steps = self.config['train_size']/self.config['batch_size']
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    warmup_steps = math.floor(total_steps * self.config['warmup'])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return [optimizer],[scheduler]

  # def validation_epoch_end(self, outputs):
  #   losses = []
  #   for output in outputs:
  #     loss = output['val_loss'].detach().cpu()
  #     losses.append(loss)
  #   avg_loss = torch.mean(torch.stack(losses))
  #   self.log("avg_val_loss", avg_loss)


def unique(sequence):
    # convert list to set without changing order of elements
    return dict.fromkeys(sequence).keys()

def precision(actual, predicted, k):
    act_set = unique(actual)
    
    if len(predicted) < k:
        pred_set = unique(predicted)
    else:
        pred_set = unique(predicted[:k])

    if len(act_set) ==0 or k == 0: 
        result = 0.0
    else:
        result = len(act_set & pred_set) / float(k)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, help="Path to input directory containing train.tsv and test.tsv")
    parser.add_argument("--label_file", default='../data/ipc-sections/20210101/labels_group_id_4.tsv', type=str)
    parser.add_argument("--checkpoint", type=str, help="Path to the saved checkpoing to test")


    args = parser.parse_args()

    print("***** Reading standard label file *****")

    with open(args.label_file, 'r') as in_f:
        lines = in_f.read().splitlines()[1:]
    attributes = [l.split('\t')[0] for l in lines]

    train_path = os.path.join(args.in_dir,'train.csv')
    val_path = os.path.join(args.in_dir,'test.csv')

    train_data = pd.read_csv(train_path)
    print(train_data)

    ########TEST#########
    #model_name = "camembert-base"
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #pmlc_ds = PMLC_Dataset(train_path, tokenizer, attributes=attributes)
    #pmlc_ds_val = PMLC_Dataset(val_path, tokenizer, attributes=attributes, sample=None)

    #print(pmlc_ds.__getitem__(0))
    #print(pmlc_ds.__getitem__(0)['labels'].shape, pmlc_ds.__getitem__(0)['input_ids'].shape, pmlc_ds.__getitem__(0)['attention_mask'].shape)
    #print(len(pmlc_ds))
    #print(len(pmlc_ds_val))
    #####################


    pmlc_data_module = PMLC_Data_Module(train_path, val_path, attributes=attributes)
    pmlc_data_module.setup()
    pmlc_data_module.train_dataloader()
    len(pmlc_data_module.train_dataloader())


    config = {
    'model_name': 'camembert-base',
    'n_labels': len(attributes),
    'batch_size': 64,
    'lr': 3e-6,
    'warmup': 0.2, 
    'train_size': len(pmlc_data_module.train_dataloader()),
    'weight_decay': 0,
    'n_epochs': 4}

    # model
    model = PMLC_Classifier(config)

    ###############TEST###############
    #idx=0
    #input_ids = pmlc_ds.__getitem__(idx)['input_ids']
    #attention_mask = pmlc_ds.__getitem__(idx)['attention_mask']
    #labels = pmlc_ds.__getitem__(idx)['labels']
    #model.cpu()
    #loss, output = model(input_ids.unsqueeze(dim=0), attention_mask.unsqueeze(dim=0), labels.unsqueeze(dim=0))
    #print(labels.shape, output.shape, output)
    ##################################

    # # trainer and fit
    data_name = args.in_dir.strip("/").split("/")[-1]
    checkpoint_callback = ModelCheckpoint(dirpath=f"./models/{data_name}", save_top_k=2, monitor="validation loss ")
    trainer = pl.Trainer(max_epochs=config['n_epochs'], gpus=1, num_sanity_val_steps=5, callbacks=[checkpoint_callback])
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        trainer.fit(model, pmlc_data_module)

    # predict 
    # method to convert list of comments into predictions for each text doc
    def classify_raw_texts(model, dm):
        predictions = trainer.predict(model, datamodule=dm)
        print("number of predictions: ", len(predictions))
        flattened_predictions = np.stack([torch.sigmoid(torch.Tensor(p)) for batch in predictions for p in batch])
        return flattened_predictions   
    y_test_pred = classify_raw_texts(model, pmlc_data_module)

    val_data = pd.read_csv(val_path)
    true_labels = np.array(val_data[attributes])

    ##from sklearn import metrics
    #plt.figure(figsize=(15, 8))
    #for i, attribute in enumerate(attributes):
    #  fpr, tpr, _ = metrics.roc_curve(
    #      true_labels[:,i].astype(int), predictions[:, i])
    #  auc = metrics.roc_auc_score(
    #      true_labels[:,i].astype(int), predictions[:, i])
    #  plt.plot(fpr, tpr, label='%s %g' % (attribute, auc))
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #plt.legend(loc='lower right')
    #plt.title('RoBERTa Trained on UCC Datatset - AUC ROC')


    # Get predictions for test data
    print("***** Running prediction *****")
    pre_n_1 = []
    pre_n_3 = []
    pre_n_5 = []

    predictions = []
    y_test = []

    for k in tqdm(range(len(y_test_pred))):
        true = [i for i,j in zip(attributes, true_labels[k,:]) if j]
        pred = [x for _,x in sorted(zip(y_test_pred[k,:], attributes), reverse=True)]

        predictions.append(pred)
        y_test.append(true)

        pre_1 = precision(true, pred, 1)
        pre_3 = precision(true, pred, 3)
        pre_5 = precision(true, pred, 5)

        pre_n_1.append(pre_1)
        pre_n_3.append(pre_3)
        pre_n_5.append(pre_5)
    
    res_df = pd.DataFrame({'true_labels': y_test, 
                            'predict_labels': predictions, 
                            'precision@1': pre_n_1, 
                            'precision@3': pre_n_3,
                            'precision@5': pre_n_5                          
                            })

    res_df.to_csv(f'./models/{data_name}.res', index=False)
    print(res_df)
    
    for col in ["precision@1", "precision@3", "precision@5"]:
        print(col + ": ", res_df[col].mean())



if __name__ == "__main__":
    main()
