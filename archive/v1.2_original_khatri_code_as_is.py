# -*- coding: utf-8 -*-
"""OOV_MRPC_paraphrase_task.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13W_oktxSFMAB6m5Rfvy8vidxuQDrCWwW
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MAXPARAMS = 300

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting

eval_metrics = {"acc": acc}

def generate_initial_parameterisation(train_circuits, test_circuits, embedding_model, qnlp_model):

    # Note that in this vocab, the same word can have multiple types, which each occur separately
    train_vocab = {symb.name.rsplit('_', 1)[0] for d in train_circuits for symb in d.free_symbols}
    test_vocab = {symb.name.rsplit('_', 1)[0] for d in test_circuits for symb in d.free_symbols}

    print(len(test_vocab.union(train_vocab)), len(train_vocab), len(test_vocab))
    print(f'OOV word count: {len(test_vocab - train_vocab)} / {len(test_vocab)}')

    n_oov_symbs = len({symb.name for d in test_circuits for symb in d.free_symbols} - {symb.name for d in train_circuits for symb in d.free_symbols})
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in test_circuits for symb in d.free_symbols})}')

    max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for d in train_circuits for symb in d.free_symbols),
                            max(int(symb.name.rsplit('_', 1)[1]) for d in test_circuits for symb in d.free_symbols)) + 1
    print(f'Max params/word: {max_word_param_length}')

    train_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in train_vocab}
    test_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in test_vocab}

    initial_param_vector = []

    for sym in qnlp_model.symbols:
        wrd, idx = sym.name.rsplit('_', 1)
        initial_param_vector.append(train_vocab_embeddings[wrd][int(idx)])

    qnlp_model.weights = np.array(initial_param_vector)

    return train_vocab_embeddings, test_vocab_embeddings, max_word_param_length


def generate_OOV_parameterising_model(trained_qnlp_model, train_vocab_embeddings, max_word_param_length):

    trained_params_raw = {symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
    trained_param_vectors = {wrd: np.zeros(max_word_param_length) for wrd in train_vocab_embeddings}

    for symbol, train_val in trained_params_raw.items():
        wrd, idx = symbol.name.rsplit('_', 1)
        trained_param_vectors[wrd][int(idx)] = train_val

    wrds_in_order = list(train_vocab_embeddings.keys())

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

import string

train_X_1 = []
train_X_2 = []
train_y = []

with open("data/msr_paraphrase_train.txt", encoding='utf-8-sig') as f:
    for line in f:
        procd_line = line.strip().split('\t')
        train_X_1.append(procd_line[3])
        train_X_2.append(procd_line[4])
        train_y.append(int(procd_line[0]))



test_X_1 = []
test_X_2 = []
test_y = []


with open("data/msr_paraphrase_test.txt", encoding='utf-8-sig') as f:
    for line in f:
        procd_line = line.strip().split('\t')
        test_X_1.append(procd_line[3])
        test_X_2.append(procd_line[4])
        test_y.append(int(procd_line[0]))


MAXLEN = 10


filt_train_X1 = []
filt_train_X2 = []
filt_train_y = []

filt_test_X1 = []
filt_test_X2 = []
filt_test_y = []

ctr_train = 0
for label, s1, s2 in zip(train_y, train_X_1, train_X_2):
    if max((len(s1.split(' ')), len(s2.split(' ')))) <= MAXLEN:
        ctr_train += 1
        filt_train_X1.append(s1.translate(str.maketrans('', '', string.punctuation)))
        filt_train_X2.append(s2.translate(str.maketrans('', '', string.punctuation)))
        this_y = [0, 0]
        this_y[label] = 1
        filt_train_y.append(this_y)

ctr_test = 0
for label, s1, s2 in zip(test_y, test_X_1, test_X_2):
    if max((len(s1.split(' ')), len(s2.split(' ')))) <= MAXLEN:
        ctr_test += 1
        filt_test_X1.append(s1.translate(str.maketrans('', '', string.punctuation)))
        filt_test_X2.append(s2.translate(str.maketrans('', '', string.punctuation)))
        this_y = [0, 0]
        this_y[label] = 1
        filt_test_y.append(this_y)

print(ctr_train, ctr_test)

import fasttext as ft

embedding_model = ft.load_model(f'./cc.en.{MAXPARAMS}.bin')

from lambeq import BobcatParser

parser = BobcatParser()

train_diags1 = parser.sentences2diagrams(filt_train_X1, suppress_exceptions=False)
train_diags2 = parser.sentences2diagrams(filt_train_X2, suppress_exceptions=False)

test_diags1 = parser.sentences2diagrams(filt_test_X1, suppress_exceptions=False)
test_diags2 = parser.sentences2diagrams(filt_test_X2, suppress_exceptions=False)

from collections import Counter
# We omit any case where the 2 phrases are not parsed to the same type
joint_diagrams_train = [d1 @ d2.r if d1.cod == d2.cod else None for (d1, d2) in zip(train_diags1, train_diags2)]
joint_diagrams_test = [d1 @ d2.r if d1.cod == d2.cod else None for (d1, d2) in zip(test_diags1, test_diags2)]


train_diags_raw = [d for d in joint_diagrams_train if d is not None]
train_y = np.array([y for d,y in zip(joint_diagrams_train, filt_train_y) if d is not None])

test_diags_raw = [d for d in joint_diagrams_test if d is not None]
test_y = np.array([y for d,y in zip(joint_diagrams_test, filt_test_y) if d is not None])

print("FINAL DATASET SIZE:")
print("-----------------------------------")
print(f"Training: {len(train_diags_raw)} {Counter([tuple(elem) for elem in train_y])}")
print(f"Testing : {len(test_diags_raw)} {Counter([tuple(elem) for elem in test_y])}")

from tqdm import tqdm
from lambeq import Rewriter, remove_cups

rewriter = Rewriter(['prepositional_phrase', 'determiner', 'coordination', 'connector', 'prepositional_phrase'])

train_X = []
test_X = []

for d in tqdm(train_diags_raw):
    train_X.append(remove_cups(rewriter(d).normal_form()))

for d in tqdm(test_diags_raw):
    test_X.append(remove_cups(rewriter(d).normal_form()))

from discopy.quantum.gates import CX, Rx, H, Bra, Id

equality_comparator = (CX >> (H @ Rx(0.5)) >> (Bra(0) @ Id(1)))
equality_comparator.draw()

from lambeq import AtomicType, IQPAnsatz, Sim14Ansatz, Sim15Ansatz
from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset
import time
import json

SEED = 0
EPOCHS = 1000
BATCH_SIZE = 30

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE


def run_experiment(nlayers=1, seed=SEED):
    print(f'RUNNING WITH {nlayers} layers')
    ansatz = Sim15Ansatz({N: 1, S: 1, P:1}, n_layers=nlayers, n_single_qubit_params=3)

    train_circs = [ansatz(d) >> equality_comparator for d in train_X]
    test_circs = [ansatz(d) >> equality_comparator for d in test_X]

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

    train_dataset = Dataset(
                train_circs,
                train_y,
                batch_size=BATCH_SIZE)

    np.random.seed(seed)

    train_embeddings, test_embeddings, max_w_param_length = generate_initial_parameterisation(train_circs, test_circs, embedding_model, lmbq_model)

    print('BEGINNING QNLP MODEL TRAINING')
    trainer.fit(train_dataset, logging_step=100)

    train_preds = lmbq_model.get_diagram_output(train_circs)
    train_loss = loss(train_preds, train_y)
    train_acc = acc(train_preds, train_y)
    print(f'TRAIN STATS: {train_loss, train_acc}')

    print('BEGINNING DNN MODEL TRAINING')
    NN_model = generate_OOV_parameterising_model(lmbq_model, train_embeddings, max_w_param_length)

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

import tensorflow as tf
compr_results = {}

tf_seeds = [0, 1, 2]

for tf_seed in tf_seeds:
    tf.random.set_seed(tf_seed)
    this_seed_results = []
    for nl in [3,2,1]:
        this_seed_results.append(run_experiment(nl, tf_seed))
    compr_results[tf_seed] = this_seed_results

import json

bkup = compr_results

with open('./results/MSR_OOV_S15.json', 'w') as f:
    json.dump(bkup, f)