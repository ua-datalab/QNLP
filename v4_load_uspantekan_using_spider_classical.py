# -*- coding: utf-8 -*-
"""v4_load_uspantekan_using_spider_classical.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YWBA_T_4phFMxts-dF8W7c1msBPhe4K1
"""

#should ideally have to do this only once per cyverse session

!pip install lambeq
!pip install wandb
!wandb login de268c256c2d4acd9085ee4e05d91706c49090d7
!python -m spacy download es_core_news_sm
!pip install lightning



import wandb
wandb.init(project="v4_uspantekan")

import torch
import wandb

BATCH_SIZE = 30
EPOCHS = 2
LEARNING_RATE = 0.05
SEED = 0
DATA_BASE_FOLDER= "sample_data"
TRAIN="uspantek_train.txt"
DEV="uspantek_dev.txt"
TEST="uspantek_test.txt"

import spacy
from lambeq import SpacyTokeniser
spanish_tokeniser = spacy.load("es_core_news_sm")
spacy_spanish_tokeniser = SpacyTokeniser()
spacy_spanish_tokeniser.tokeniser = spanish_tokeniser

import os
from lambeq import BobcatParser

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])
            sentences.append(line[1:].strip())
    return labels, sentences


train_labels, train_data = read_data(os.path.join(DATA_BASE_FOLDER,TRAIN))
val_labels, val_data = read_data(os.path.join(DATA_BASE_FOLDER,DEV))
test_labels, test_data = read_data(os.path.join(DATA_BASE_FOLDER,TEST))

TESTING = int(os.environ.get('TEST_NOTEBOOKS', '0'))

if TESTING:
    train_labels, train_data = train_labels[:2], train_data[:2]
    val_labels, val_data = val_labels[:2], val_data[:2]
    test_labels, test_data = test_labels[:2], test_data[:2]
    EPOCHS = 1

train_data[:5]

train_labels[:5]

from lambeq import spiders_reader
#
#parser = BobcatParser(verbose='text')


from tqdm import tqdm
def spanish_diagrams(list_sents,labels):
    list_target = []
    labels_target = []
    for sent, label in tqdm(zip(list_sents, labels),desc="reading sent"):
        #using bob cat parser- note: this wasn't compatible with spider ansatz
        # tokenized = spacy_spanish_tokeniser.tokenise_sentence(sent)
        # diag =parser.sentence2diagram(tokenized, tokenised= True)
        # diag.draw()
        # list_target.append(diag)

        tokenized = spacy_spanish_tokeniser.tokenise_sentence(sent)
        # tokenized = sent.split(" ")
        if len(tokenized)> 30:
            print(f"no of tokens inthis sentence is {len(tokenized)}")
            continue

        spiders_diagram = spiders_reader.sentence2diagram(sent)
        list_target.append(spiders_diagram)
        labels_target.append(label)

    print("no. of items processed= ", len(list_target))
    return list_target, labels_target

train_diagrams, train_labels_v2 = spanish_diagrams(train_data,train_labels)
val_diagrams, val_labels_v2 = spanish_diagrams(val_data,val_labels)
test_diagrams, test_labels_v2 = spanish_diagrams(test_data,test_labels)

train_labels = train_labels_v2
val_labels = val_labels_v2
test_labels = test_labels_v2

# val_diagrams = spanish_diagrams(val_data)
# test_diagrams = spanish_diagrams(test_data)

#print and assert statements for debugging
assert len(train_diagrams)== len(train_labels_v2)
print(len(train_diagrams), len(test_diagrams), len(val_diagrams))
assert len(train_diagrams)== len(train_labels)
assert len(val_diagrams)== len(val_labels)
assert len(test_diagrams)== len(test_labels)
train_diagrams[0].draw()
val_diagrams[0].draw()
test_diagrams[0].draw()

from lambeq.backend.tensor import Dim

from lambeq import AtomicType, SpiderAnsatz

ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(2),
                       AtomicType.SENTENCE: Dim(2),
                       AtomicType.PREPOSITIONAL_PHRASE: Dim(2),
                       })

train_circuits =  [ansatz(diagram) for diagram in train_diagrams]
val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

train_circuits[0].draw()

from lambeq import PytorchModel

all_circuits = train_circuits + val_circuits + test_circuits
model = PytorchModel.from_diagrams(all_circuits)

sig = torch.sigmoid

def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}

from lambeq import Dataset

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

print(len(train_labels), len(train_circuits))
#print and assert statements for debugging
print(len(train_circuits), len(val_circuits), len(test_circuits))
assert len(train_circuits)== len(train_labels)
assert len(val_circuits)== len(val_labels)
assert len(test_circuits)== len(test_labels)
train_circuits[0].draw()
val_circuits[0].draw()
test_circuits[0].draw()

sweep_config = {
    'method': 'random'
    }
metric = {
    'name': 'loss',
    'goal': 'minimize'
    }

sweep_config['metric'] = metric

parameters_dict = {
    'LEARNING_RATE': {
          'values': [0.3, 0.03, 0.003,0.0003]
        },
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': 1}
    })

import pprint

pprint.pprint(sweep_config)
sweep_id = wandb.sweep(sweep_config, project="uspantekan_spider_tuning")

from lambeq import PytorchTrainer

from lightning.pytorch.loggers import WandbLogger


def train_model():
  wandb_logger = WandbLogger()
  trainer = PytorchTrainer(
          model=model,
          loss_function=torch.nn.BCEWithLogitsLoss(),
          optimizer=torch.optim.AdamW,
          learning_rate=LEARNING_RATE,
          epochs=EPOCHS,
          evaluate_functions=eval_metrics,
          evaluate_on_train=True,
          verbose='text',
          seed=SEED)
  trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)

wandb.agent(sweep_id, train_model, count=1)

# print dev accuracy
dev_acc = accuracy(model(val_circuits), torch.tensor(val_labels))
print('Dev accuracy:', dev_acc.item())
wandb.log({"dev_accuracy":dev_acc.item()})

import matplotlib.pyplot as plt
import numpy as np

fig1, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, sharey='row', figsize=(10, 6))

ax_tl.set_title('Training set')
ax_tr.set_title('Development set')
ax_bl.set_xlabel('Epochs')
ax_br.set_xlabel('Epochs')
ax_bl.set_ylabel('Accuracy')
ax_tl.set_ylabel('Loss')

colours = iter(plt.rcParams['axes.prop_cycle'].by_key()['color'])
range_ = np.arange(1, trainer.epochs+1)
ax_tl.plot(range_, trainer.train_epoch_costs, color=next(colours))
ax_bl.plot(range_, trainer.train_eval_results['acc'], color=next(colours))
ax_tr.plot(range_, trainer.val_costs, color=next(colours))
ax_br.plot(range_, trainer.val_eval_results['acc'], color=next(colours))

# print dev accuracy
dev_acc = accuracy(model(val_circuits), torch.tensor(val_labels))
print('Dev accuracy:', dev_acc.item())
wandb.log({"dev_accuracy":dev_acc.item()})

# print test accuracy- use this ony once
# test_acc = accuracy(model(test_circuits), torch.tensor(test_labels))
# print('Test accuracy:', test_acc.item())
# wandb.log({"test_accuracy":test_acc.item()})