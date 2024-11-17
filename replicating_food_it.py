# -*- coding: utf-8 -*-
import os 
from lambeq import RemoveCupsRewriter
from tqdm import tqdm
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import torch
from torch import nn
import spacy
from lambeq import SpacyTokeniser
import numpy as np
import fasttext as ft
from lambeq import PytorchTrainer
from lambeq.backend.tensor import Dim
from lambeq import AtomicType
from lambeq import Dataset
from lambeq import PytorchModel, NumpyModel, TketModel, PennyLaneModel
from lambeq import TensorAnsatz,SpiderAnsatz
from lambeq import BobcatParser,spiders_reader
from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset

bobCatParser=BobcatParser()
parser_to_use = bobCatParser  #[bobCatParser, spiders_reader]
ansatz_to_use = SpiderAnsatz #[IQP, Sim14, Sim15,TensorAnsatz ]
model_to_use  =  PytorchModel #[numpy, pytorch]
trainer_to_use= PytorchTrainer #[PytorchTrainer, QuantumTrainer]


# maxparams is the maximum qbits (or dimensions of the tensor, as your case be)
MAXPARAMS = 300
BATCH_SIZE = 30
EPOCHS = 30
LEARNING_RATE = 3e-2
SEED = 0
DATA_BASE_FOLDER= "data"


USE_SPANISH_DATA=False
USE_USP_DATA=False
USE_FOOD_IT_DATA = True
USE_MRPC_DATA=False

#setting a flag for TESTING so that it is done only once.
#  Everything else is done on train and dev
TESTING = False

if(USE_USP_DATA):
    TRAIN="uspantek_train.txt"
    DEV="uspantek_dev.txt"
    TEST="uspantek_test.txt"

if(USE_SPANISH_DATA):
    TRAIN="spanish_train.txt"
    DEV="spanish_dev.txt"
    TEST="spanish_test.txt"


if(USE_MRPC_DATA):
    TRAIN="mrpc_train_80_sent.txt"
    DEV="mrpc_dev_10_sent.txt"
    TEST="mrpc_test_10sent.txt"

if(USE_FOOD_IT_DATA):
    TRAIN="mc_train_data.txt"
    DEV="mc_dev_data.txt"
    TEST="mc_test_data.txt"

sig = torch.sigmoid
def accuracy(y_hat, y):
        assert type(y_hat)== type(y)
        # half due to double-counting
        #todo/confirm what does he mean by double counting
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  

eval_metrics = {"acc": accuracy}
spacy_tokeniser = SpacyTokeniser()

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            labels.append([t, 1-t])            
            sentences.append(line[1:].strip())
    return labels, sentences

#read the base data, i.e plain text english.
train_labels, train_data = read_data(os.path.join(DATA_BASE_FOLDER,TRAIN))
val_labels, val_data = read_data(os.path.join(DATA_BASE_FOLDER,DEV))
test_labels, test_data = read_data(os.path.join(DATA_BASE_FOLDER,TEST))


# todo: not sure what they are doing here. need to figure out as and when we get to testing
# if TESTING:
#     train_labels, train_data = train_labels[:2], train_data[:2]
#     val_labels, val_data = val_labels[:2], val_data[:2]
#     test_labels, test_data = test_labels[:2], test_data[:2]
#     EPOCHS = 1


#change: this is different than our code oov.py - we use sentence2diagram- that needs to change   
train_diagrams = parser_to_use.sentences2diagrams(train_data)
val_diagrams = parser_to_use.sentences2diagrams(val_data)
test_diagrams = parser_to_use.sentences2diagrams(test_data)


#change: this is different than our code oov.py    
ansatz = ansatz_to_use({AtomicType.NOUN: Dim(2),
                    AtomicType.SENTENCE: Dim(2)                        
                    })


#use the anstaz to create circuits from diagrams
train_circuits =  [ansatz(diagram) for diagram in train_diagrams]
val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]        
print("length of each circuit in train is:")
print([len(x) for x in train_circuits])


#change: this is different than our code oov.py - i.e adding up all 3 circuits during initialization? is this to prevent OOV?   
qnlp_model = model_to_use.from_diagrams(train_circuits+val_circuits+test_circuits)


val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

print(len(train_labels), len(train_circuits))
#print and assert statements for debugging
print(len(train_circuits), len(val_circuits), len(test_circuits))
assert len(train_circuits)== len(train_labels)
assert len(val_circuits)== len(val_labels)
assert len(test_circuits)== len(test_labels)


trainer = trainer_to_use(
        model=qnlp_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        use_tensorboard=True, #todo: why isnt any visualization shown despite use_tensorboard=True
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)



train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)

val_dataset = Dataset(
            val_circuits,
            val_labels,
            batch_size=BATCH_SIZE)

#change: this is different than our code oov.py    - we dont pass val_dataset due to OOV
trainer.fit(train_dataset,val_dataset, log_interval=1)


# print test accuracy
test_acc = accuracy(qnlp_model(test_circuits), torch.tensor(test_labels))
print('Test accuracy:', test_acc.item())
    
    
