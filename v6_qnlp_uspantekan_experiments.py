#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/ua-datalab/QNLP/blob/megh_dev/OOV_MRPC_paraphrase_task.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[ ]:


# get_ipython().system('pip install lambeq')
# get_ipython().system('pip install fasttext')


# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# In[2]:


from psutil import virtual_memory
ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM\n'.format(ram_gb))

if ram_gb < 20:
  print('Not using a high-RAM runtime')
else:
  print('You are using a high-RAM runtime!')


# In[3]:


from lambeq.text2diagram.tree_reader import BobcatParser
import lambeq

parser= BobcatParser()


# In[4]:


MAXPARAMS = 108


# In[13]:


import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt


# In[7]:


# Load data and extract features and labels for train and test sets:
# Ensure data is accessible to notebook
import string

train_X = []
train_y = []

with open("./spanish_train.txt", encoding='utf-8-sig') as f:
    for line in f:
        procd_line = line.strip().split('  ')
        train_X.append(procd_line[1])
        train_y.append(int(procd_line[0]))

test_X = []
test_y = []

with open("./spanish_test.txt", encoding='utf-8-sig') as f:
    for line in f:
        procd_line = line.strip().split('  ')
        test_X.append(procd_line[1])
        test_y.append(int(procd_line[0]))


MAXLEN = 10


filt_train_X = []
filt_train_y = []

filt_test_X = []
filt_test_y = []

ctr_train = 0
for label, s in zip(train_y, train_X):
    if len(s.split(' ')) <= MAXLEN:
        ctr_train += 1
        filt_train_X.append(s.translate(str.maketrans('', '', string.punctuation)))
        this_y = [0, 0]
        this_y[label] = 1
        filt_train_y.append(this_y)

ctr_test = 0
for label, s in zip(test_y, test_X):
    if len(s.split(' ')) <= MAXLEN:
        ctr_test += 1
        filt_test_X.append(s.translate(str.maketrans('', '', string.punctuation)))
        this_y = [0, 0]
        this_y[label] = 1
        filt_test_y.append(this_y)

print(ctr_train, ctr_test)


#  # Using Fatext Spanish embeddings
# 
#  We don't need the model itself. We just need to find where in the code embeddings are used and just add the Spanish embeddings there.
# 
#  We download the salient model from [https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file)
# 
# Ex.: `embedding_model = ft.load_model(f'./dataset/cc.en.{MAXPARAMS}.bin')`

# In[8]:


## download embeddings:
# get_ipython().system('wget -c https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1 -O ./embeddings-l-model.bin')


# In[9]:


import fasttext as ft
embedding_model = ft.load_model('/content/embeddings-l-model.bin')


# In[10]:


from lambeq.text2diagram.tree_reader import BobcatParser
import lambeq

parser= BobcatParser()


# In[11]:


# Parse and create trees:
train_diags = parser.sentences2diagrams(filt_train_X, suppress_exceptions=False)

test_diags = parser.sentences2diagrams(filt_test_X, suppress_exceptions=False)


# In[14]:


from collections import Counter
# We omit any case where the 2 phrases are not parsed to the same type
# Khatri et al. is creating a circuit with the combination of X1 and X2.
# We are not joining, so not needed
# joint_diagrams_train = [d1 @ d2.r if d1.cod == d2.cod else None for d1 in train_diags]
# joint_diagrams_test = [d1 @ d2.r if d1.cod == d2.cod else None for d1 in test_diags]

#  Editing lines to get what we need from train_diags
train_diags_raw = [d for d in train_diags if d is not None]
train_y = np.array([y for d,y in zip(train_diags, filt_train_y) if d is not None])

test_diags_raw = [d for d in test_diags if d is not None]
test_y = np.array([y for d,y in zip(test_diags, filt_test_y) if d is not None])

print("FINAL DATASET SIZE:")
print("-----------------------------------")
print(f"Training: Sentences- {len(train_diags_raw)} {Counter([tuple(elem) for elem in train_y])}")
print(f"Testing : Sentences- {len(test_diags_raw)} {Counter([tuple(elem) for elem in test_y])}")


# In[15]:


from tqdm import tqdm
from lambeq import Rewriter, RemoveCupsRewriter

# rewriter will club cups in order to deal with the width of the sentence by
# collapsing cups based on user preferences, mainly for functional words
rewriter = RemoveCupsRewriter()
# rewriter = Rewriter(['prepositional_phrase',
#            'determiner', 'coordination', 'connector',
#            'prepositional_phrase'])

train_X = []
test_X = []

for d in tqdm(train_diags):
    if d is not None:
        train_X.append(rewriter(d).normal_form())
    else:
        print("found d is null")

for d in tqdm(test_diags):
              if d is not None:
                test_X.append(rewriter(d).normal_form())


# #  ToDo: find out what the following do:

# In[23]:


from lambeq import AtomicType, IQPAnsatz, Sim14Ansatz, Sim15Ansatz
from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset
import time
import json

SEED = 0
EPOCHS = 1000
BATCH_SIZE = 30

# Define types of elements for future Qbit assignment
# How many legs does a type of element need- lambeq's paper has defined this
N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE


def run_experiment(nlayers=1, seed=SEED):
    print("insdie run_experiment")
    print(f'RUNNING WITH {nlayers} layers and seed {seed}')
    #  given english language, convert
    # todo: check if the error comes from the machine not finding N, P, S
    ansatz = Sim15Ansatz({N: 1, S: 1, P:1}, n_layers=nlayers, n_single_qubit_params=3)
    print(f"ansatz: {ansatz}")
    #  Tuning- assign numebr of qbits per word, can be modified
    #  We assign Qbits based on how many outputs a given head will connect to
    #  PP "in" will need at least 3 QBITS
    train_circs = [ansatz(d) for d in train_X]
    test_circs = [ansatz(d) for d in test_X]
    print(f"train_circs size: {len(train_circs)}, train_circs example: {train_circs[:1]}")
    print(f"test_circs size: {len(test_circs)},test_circs example: {test_circs[:1]}")

    lmbq_model = NumpyModel.from_diagrams(train_circs, use_jit=True)

    trainer = QuantumTrainer(
        lmbq_model,
        loss_function=loss,
        epochs=EPOCHS,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose = 'text',
        seed=seed
    )
    print("trainer created")
    train_dataset = Dataset(
                train_circs,
                train_y,
                batch_size=BATCH_SIZE)

    np.random.seed(seed)

    train_embeddings, test_embeddings, max_w_param_length =\
      generate_initial_parameterisation(train_circs, test_circs, embedding_model, lmbq_model)


    print("done generating initial embeddings from fast text")
    print(f"type of of train_embeddings vocab is {type(train_embeddings)}")
    print(f"no of words in train_embeddings is {len(train_embeddings)}")
    print(f"the type of value of of of train_embeddings vocab is {len(train_embeddings)}")
    print(f"type of of test_embeddings vocab is {type(test_embeddings)}")
    print(f"length of of test_embeddings vocab is {len(test_embeddings)}")
    import sys
    sys.exit()

    print(f"BEGINNING QNLP MODEL TRAINING")

    # ERROR IN THE NEXT LINE:
    trainer.fit(train_dataset)
    print("fit complete")

    train_preds = lmbq_model.get_diagram_output(train_circs)
    train_loss = loss(train_preds, train_y)
    train_acc = acc(train_preds, train_y)
    print(f'TRAIN STATS: {train_loss, train_acc}')

    print('BEGINNING DNN MODEL TRAINING')
    NN_model = generate_OOV_parameterising_model(lmbq_model,
                                                 train_embeddings,
                                                 max_w_param_length)

    prediction_model = NumpyModel.from_diagrams(test_circs, use_jit=True)

    trained_wts = trained_params_from_model(lmbq_model, train_embeddings, max_w_param_length)

    print('Evaluating SMART MODEL')
    smart_loss, smart_acc = evaluate_test_set(prediction_model,
                                              test_circs,
                                              test_y,
                                              trained_wts,
                                              test_embeddings,
                                              max_w_param_length,
                                              OOV_strategy='model',
                                              OOV_model=NN_model)

    print('Evaluating EMBED model')
    embed_loss, embed_acc = evaluate_test_set(prediction_model,
                                              test_circs,
                                              test_y,
                                              trained_wts,
                                              test_embeddings,
                                              max_w_param_length,
                                              OOV_strategy='embed')

    print('Evaluating ZEROS model')
    zero_loss, zero_acc = evaluate_test_set(prediction_model,
                                              test_circs,
                                              test_y,
                                              trained_wts,
                                              test_embeddings,
                                              max_w_param_length,
                                              OOV_strategy='zeros')

    rand_losses = []
    rand_accs = []

    print('Evaluating RAND MODEL')
    for _ in range(1000):


        rl, ra = evaluate_test_set(prediction_model,
                                   test_circs,
                                   test_y,
                                   trained_wts,
                                   test_embeddings,
                                   max_w_param_length,
                                   OOV_strategy='random')

        rand_losses.append(rl)
        rand_accs.append(ra)

    res =  {'TRAIN': (train_loss, train_acc),
            'NN': (smart_loss, smart_acc),
            'EMBED': (embed_loss, embed_acc),
            'RAND': (rand_losses, rand_accs),
            'ZERO': (zero_loss, zero_acc)
           }
    print(f'ZERO: {res["ZERO"]}')
    print(f'EMBED: {res["EMBED"]}')
    print(f'NN: {res["NN"]}')

    return res


# In[42]:


# Helper functions from khatri et. al.:

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

eval_metrics = {"acc": acc}

def generate_initial_parameterisation(train_circuits, test_circuits, embedding_model, qnlp_model):

    # Note that in this vocab, the same word can have multiple types, which each occur separately
    train_vocab = {symb.name.rsplit('_', 1)[0] for d in train_circuits for symb in d.free_symbols}
    test_vocab = {symb.name.rsplit('_', 1)[0] for d in test_circuits for symb in d.free_symbols}

    print(f"length of combined vocab is {len(test_vocab.union(train_vocab))}")
    print(f"length of train vocab is {len(train_vocab)}")
    print(f"length of test vocab is {len(test_vocab)}")

    print(f"value of train vocab is {(train_vocab)}")
    print(f"value of test vocab is {(test_vocab)}")

    #wrong- not sure why he is doing OOV like this
    #print(f'OOV word count: {len(test_vocab - train_vocab)} / {len(test_vocab)}')

    n_oov_symbs = len({symb.name for d in test_circuits for symb in d.free_symbols} - {symb.name for d in train_circuits for symb in d.free_symbols})
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in test_circuits for symb in d.free_symbols})}')

    max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for d in train_circuits for symb in d.free_symbols),
                            max(int(symb.name.rsplit('_', 1)[1]) for d in test_circuits for symb in d.free_symbols)) + 1
    print(f'Max params/word: {max_word_param_length}')




    #for a given word get the corresponding
    train_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in train_vocab}
    test_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in test_vocab}



    initial_param_vector = []
    #for eah symbol in qnlp, find its embedding vector in fasttext and then pick the value in that vector at idex- still don't know hwy
    #todo- find what is idx
    print(f"printing each symbol already in qnlp model")
    for sym in qnlp_model.symbols:
        print(f"value of sym is:{sym}")
        wrd, idx = sym.name.rsplit('_', 1)
        initial_param_vector.append(train_vocab_embeddings[wrd][int(idx)])
    print(f"value of initial_param_vector is {initial_param_vector}")
    print(f"length of initial_param_vector is {len(initial_param_vector)}")


    #the initial weights of the model is the first value in the embedding of every single word in qnlp_model
    #not sure why he is using it to initialize here. I would have expected him to take the whole embedding for the word or a sum of it as
    #initial parameter assignment. but since even xavier glorot has to start somewher,e  i'll let it pass.
    #todo: dig deeper and find what symbols are in qnlp model is? that should give a better understanding.
    qnlp_model.weights = np.array(initial_param_vector)

    print(f"lenght of train_vocab embeddings are {len(train_vocab_embeddings)}")
    print(f"lenght of test_vocab embeddings are {len(test_vocab_embeddings)} ")

    return train_vocab_embeddings, test_vocab_embeddings, max_word_param_length


def generate_OOV_parameterising_model(trained_qnlp_model,
                                      train_vocab_embeddings,
                                      max_word_param_length):

    trained_params_raw = {symbol: param for symbol, param in zip(
                                                  trained_qnlp_model.symbols,
                                                  trained_qnlp_model.weights)}
    trained_param_vectors = {wrd: np.zeros(max_word_param_length)\
                             for wrd in train_vocab_embeddings}
    print("trained_params_raw: ", np.shape(trained_params_raw))
    print("trained_param_vectors:", np.shape(trained_param_vectors))
    for symbol, train_val in trained_params_raw.items():
        wrd, idx = symbol.name.rsplit('_', 1)
        trained_param_vectors[wrd][int(idx)] = train_val

    wrds_in_order = list(train_vocab_embeddings.keys())

    NN_train_X = np.array([train_vocab_embeddings[wrd] for wrd in wrds_in_order])
    NN_train_Y = np.array([trained_param_vectors[wrd] for wrd in wrds_in_order])

    print("dimensions of features: ", np.shape(NN_train_X))
    print("dimensions of labels: ", np.shape(NN_train_Y))

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


def evaluate_test_set(pred_model, test_circuits, test_labels, trained_params, test_vocab_embeddings, max_word_param_length, OOV_strategy='random', OOV_model=None):

    pred_parameter_map = {}

    # Use the words from train wherever possible, else use DNN prediction
    for wrd, embedding in test_vocab_embeddings.items():
        if OOV_strategy == 'model':
            pred_parameter_map[wrd] = trained_params.get(wrd, OOV_model.predict(np.array([embedding]), verbose=0)[0])
        elif OOV_strategy == 'embed':
            pred_parameter_map[wrd] = trained_params.get(wrd, embedding)
        elif OOV_strategy == 'zeros':
            pred_parameter_map[wrd] = trained_params.get(wrd, np.zeros(max_word_param_length))
        else:
            pred_parameter_map[wrd] = trained_params.get(wrd, 2 * np.random.rand(max_word_param_length)-1)

    pred_weight_vector = []

    for sym in pred_model.symbols:
        wrd, idx = sym.name.rsplit('_', 1)
        pred_weight_vector.append(pred_parameter_map[wrd][int(idx)])

    pred_model.weights = pred_weight_vector

    preds = pred_model.get_diagram_output(test_circuits)

    return loss(preds, test_labels), acc(preds, test_labels)


def trained_params_from_model(trained_qnlp_model, train_embeddings, max_word_param_length):

    trained_param_map = { symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
    trained_parameterisation_map = {wrd: np.zeros(max_word_param_length) for wrd in train_embeddings}

    for symbol, train_val in trained_param_map.items():
        wrd, idx = symbol.name.rsplit('_', 1)
        trained_parameterisation_map[wrd][int(idx)] = train_val

    return trained_parameterisation_map


# In[43]:


import tensorflow as tf
compr_results = {}

tf_seeds = [0, 1, 2]


for tf_seed in tf_seeds:
    print(f"seed: {tf_seed}")
    tf.random.set_seed(tf_seed)
    this_seed_results = []
    for nl in [3,2,1]:
        print(f"nl: {nl}")
        this_seed_results.append(run_experiment(nl, tf_seed))
    compr_results[tf_seed] = this_seed_results


# # ERROR HERE: CAN'T RUN TRAINER
# 
# Possible issues:
# 1. OOV- most of the words are not being identified, so they can't be binned into grammatical categories.

# In[ ]:


import json

bkup = compr_results

with open('./results/MSR_OOV_S15.json', 'w') as f:
    json.dump(bkup, f)

