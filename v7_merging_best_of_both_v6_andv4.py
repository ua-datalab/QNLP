# -*- coding: utf-8 -*-
"""v7
#name: v7_*
# status: this version stops at .fit()- rather some bug. v4 has my code which ran end to end- atleast passes .fit(). s
#will create a new file v7 which takes best of both worlds.
"""


import wandb
# wandb.init(project="v4_uspantekan")

import torch
from torch import nn
import wandb
import spacy
from lambeq import SpacyTokeniser
import numpy as np
import fasttext as ft
from lambeq import PytorchTrainer
from lightning.pytorch.loggers import WandbLogger
from lambeq.backend.tensor import Dim
from lambeq import AtomicType, SpiderAnsatz

ansatz_to_use = SpiderAnsatz
embedding_model = ft.load_model('./embeddings-l-model.bin')


BATCH_SIZE = 30
EPOCHS = 2
LEARNING_RATE = 0.05
SEED = 0
DATA_BASE_FOLDER= "data"

USE_MRPC_DATA=False
USE_SPANISH_DATA=True
USE_USP_DATA=False

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


spacy_tokeniser = SpacyTokeniser()

if(USE_SPANISH_DATA) or (USE_USP_DATA):
    spanish_tokeniser=spacy.load("es_core_news_sm")
    spacy_tokeniser.tokeniser = spanish_tokeniser



#for english tokenizer
if(USE_MRPC_DATA):
    english_tokenizer = spacy.load("en_core_web_sm")
    spacy_tokeniser.tokeniser =english_tokenizer


import os
from lambeq import BobcatParser

#the initial phases of the gates
def generate_initial_parameterisation(train_circuits, test_circuits, embedding_model, qnlp_model):

    # extract the words from the circuits
    # Note that in this vocab, the same word can have multiple types, which each occur separately
    train_vocab = {symb.name.rsplit('_', 1)[0] for d in train_circuits for symb in d.free_symbols}
    test_vocab = {symb.name.rsplit('_', 1)[0] for d in test_circuits for symb in d.free_symbols}

    print(len(test_vocab.union(train_vocab)), len(train_vocab), len(test_vocab))
    print(f'OOV word count: {len(test_vocab - train_vocab)} / {len(test_vocab)}')

    #the words i think depccg couldnt parse
    n_oov_symbs = len({symb.name for d in test_circuits for symb in d.free_symbols} - {symb.name for d in train_circuits for symb in d.free_symbols})
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in test_circuits for symb in d.free_symbols})}')

    def get_max_word_param_length(input_circuits):
        lengths=[]
        for d in input_circuits:
            for symb in d.free_symbols:
                x =  symb.name.split('_', 1)[1]
                y = x.split('__')[0]
                lengths.append(int(y))
        return lengths
    
    max_word_param_length=0
    if(ansatz_to_use==SpiderAnsatz):
        max_word_param_length_train = max(get_max_word_param_length(train_circuits))
        max_word_param_length_test = max(get_max_word_param_length(test_circuits))
        max_word_param_length = max(max_word_param_length_train, max_word_param_length_test) + 1

    assert max_word_param_length!=0
    
    # max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for d in train_circuits for symb in d.free_symbols),
    #                         max(int(symb.name.rsplit('_', 1)[1]) for d in test_circuits for symb in d.free_symbols)) + 1
    

    # max_word_param_length = max(, ) + 1
    print(f'Max params/word: {max_word_param_length}')

    #for each word in train and test vocab get its embedding from fasttext
    #spider ansatz alone writes the tokens in its vocabulary with a single underscore first and then a double underscore
    #other ansatz just write it as : _0_ so its standard parsing needed.
    if(ansatz_to_use==SpiderAnsatz):  
        # train_vocab_embeddings={}      
        def get_vocab_emb_dict(vocab):
            embed_dict={}
            for wrd in vocab:
                cleaned_wrd=wrd.split('_')[0].replace('\\','').replace(",","")
                if cleaned_wrd in embed_dict   :
                    print(f"error.  the word {cleaned_wrd} was already in dict")
                else:
                    embed_dict[cleaned_wrd]= embedding_model[cleaned_wrd] 
            return embed_dict


        train_vocab_embeddings = get_vocab_emb_dict(train_vocab)
        test_vocab_embeddings = get_vocab_emb_dict(test_vocab)

        # train_vocab_embeddings = {wrd: embedding_model[wrd.split('_')[0].replace('\\','')] for wrd in train_vocab}
        # test_vocab_embeddings = {wrd: embedding_model[wrd.split('_')[0].replace('\\','')] for wrd in test_vocab}

    else:
        train_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in train_vocab}
        test_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in test_vocab}


    #to store all the initial weights
    initial_param_vector = []

    for sym in qnlp_model.symbols:
        #@sep2nd2024-not sure what idx is supposed to do, am giong to give it the number associated with the word
        if(ansatz_to_use==SpiderAnsatz):  
            wrd =  sym.name.split('_', 1)[0].replace("\\","").replace("(","")
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            #@sep2nd2024/ end of day: getting key error for lots of words - e.g. aldea..but why are words in qnlpmodel.symbols not being done fasttext emb on the fly? why are we separating train_embeddings earlier?        
            #what is the meaning of symbols in qnlp.model
            #todo a) read the lambeq documentation on symbols b) read the 2010 discocat and CQM paper onwards up, chronologically
            #no point turning knobs without deeply understanding what symbols do
            if wrd in train_vocab_embeddings:
                initial_param_vector.append(train_vocab_embeddings[wrd][int(idx)])
            else:
                #todo: lots of words are getting hit with OOV- conirm why they are not there in fasttext emb
                # my guess is its all the unicode characters. In theory fast text is meant to create zero OOV..since it builds up from 1 gram 2 gram etc
                '''
                found that this word verdad, was OOV/not in fasttext emb
                found that this word vió, was OOV/not in fasttext emb
                found that this word yo, was OOV/not in fasttext emb
                found that this word yyyyyy was OOV/not in fasttext emb
                '''
                print(f"found that this word {wrd} was OOV/not in fasttext emb")

    qnlp_model.weights = nn.ParameterList(initial_param_vector)

    return train_vocab_embeddings, test_vocab_embeddings, max_word_param_length



def generate_OOV_parameterising_model(trained_qnlp_model, train_vocab_embeddings, max_word_param_length):
    """Read arguments from command line.

    Args:
    trained_qnlp_model- the trained_qnlp_model
    train_vocab_embeddings- the initial embeddings for words in the vocab got from fasttext
    max_word_param_length- what is the maximum size of a word

    Returns:
    Weights of a NN model which now understands/has weights for each word in fasttext as its original embedding influenced/mapped to the weights in the trained QNLP model

    """

    #dictionary that map words in the trained QNLP model to its weights at the end of QNLP training
    #todo: print and confirm if symbol means word
    trained_params_raw = {symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}

    print(trained_params_raw)
    # train_vocab_embeddings are the initial embeddings for words in the vocab got from fasttext- for each such word create an array of zeroes
    trained_param_vectors = {wrd: np.zeros(max_word_param_length) for wrd in train_vocab_embeddings}

    print(trained_param_vectors)
    
    #for each such symbol and parameter weight
    # assign the weights to the empty array created above.-
    #todo print and confirm, i think they are repeating the same weight value for every entry of the array in trained_param_vectors
    for symbol, train_val in trained_params_raw.items():
        wrd, idx = symbol.name.rsplit('_', 1)
        trained_param_vectors[wrd][int(idx)] = train_val

    wrds_in_order = list(train_vocab_embeddings.keys())

    #so the value to be trained now are the initial weights of each word from
    #fasttext, which will be trained against  gold label -trained_param_vectors i.e weights from the trained QNLP model
    NN_train_X = np.array([train_vocab_embeddings[wrd] for wrd in wrds_in_order])
    NN_train_Y = np.array([trained_param_vectors[wrd] for wrd in wrds_in_order])

    print(NN_train_X[0][:5])
    print(NN_train_Y[0][:5])

    OOV_NN_model = keras.Sequential([
      layers.Dense(int((max_word_param_length + MAXPARAMS) / 2), activation='tanh'),
      layers.Dense(max_word_param_length, activation='tanh'),
    ])

    OOV_NN_model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))

    # Embedding dim!
    OOV_NN_model.build(input_shape=(None, MAXPARAMS))

    hist = OOV_NN_model.fit(NN_train_X, NN_train_Y, validation_split=0.2, verbose=0, epochs=120)

    print(f'OOV NN model final epoch loss: {(hist.history["loss"][-1], hist.history["val_loss"][-1])}')

    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    return OOV_NN_model

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
        
        if( USE_MRPC_DATA):
            sent = sent.split('\t')[2]
        tokenized = spacy_tokeniser.tokenise_sentence(sent)
        
        
        if(not USE_MRPC_DATA):
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

"""val_diagrams = spanish_diagrams(val_data)<br>
test_diagrams = spanish_diagrams(test_data)

rint and assert statements for debugging
"""

assert len(train_diagrams)== len(train_labels_v2)
print(len(train_diagrams), len(test_diagrams), len(val_diagrams))
assert len(train_diagrams)== len(train_labels)
assert len(val_diagrams)== len(val_labels)
assert len(test_diagrams)== len(test_labels)
# train_diagrams[0].draw()
# val_diagrams[0].draw()
# test_diagrams[0].draw()



ansatz = ansatz_to_use({AtomicType.NOUN: Dim(4),
                       AtomicType.SENTENCE: Dim(2)
                    #    AtomicType.PREPOSITIONAL_PHRASE: Dim(2),
                       })

train_circuits =  [ansatz(diagram) for diagram in train_diagrams]
val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
test_circuits = [ansatz(diagram) for diagram in test_diagrams]

# train_circuits[0].draw()

from lambeq import PytorchModel

all_circuits = train_circuits + val_circuits + test_circuits
qnlp_model = PytorchModel.from_diagrams(all_circuits)

sig = torch.sigmoid

def accuracy(y_hat, y):
    return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}

from lambeq import Dataset

print([len(x) for x in train_circuits])

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
# train_circuits[0].draw()
# val_circuits[0].draw()
# test_circuits[0].draw()

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
# sweep_id = wandb.sweep(sweep_config, project="uspantekan_spider_tuning")







# wandb_logger = WandbLogger()
trainer = PytorchTrainer(
        model=qnlp_model,
        loss_function=torch.nn.BCEWithLogitsLoss(),
        optimizer=torch.optim.AdamW,
        learning_rate=LEARNING_RATE,
        epochs=EPOCHS,
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        seed=SEED)
trainer.fit(train_dataset, val_dataset, eval_interval=1, log_interval=1)

# by here the actual QNLP model is trained. Next is we are going to connect the model which learns relationsihp between fasttext emb and angles
train_embeddings, test_embeddings, max_w_param_length = generate_initial_parameterisation(train_circuits, val_circuits, embedding_model, qnlp_model)

print('BEGINNING DNN MODEL TRAINING')
NN_model = generate_OOV_parameterising_model(qnlp_model, train_embeddings, max_w_param_length)

   


