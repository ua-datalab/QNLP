# -*- coding: utf-8 -*-
"""
#name: classify.py
# Code which takes  small data (100 sent) in a given language (e.g.uspantekan or spanish)
# has about two classes (e.g.education and dancing), runs it through a 
# QNLP model, which is supported by a fasttext model,
and two neural network models to learn and make prediction.
#to get blow by blow details of what this code does, refer to a section named
"how this code runs" inside the project plan
https://github.com/ua-datalab/QNLP/blob/main/Project-Plan.md

4 major models used in this code. 
1. QNLP model, called model1
2. Fast text embedding model , called model 2
3. NN model that learns mapping between fast text embedding and QNLP trained model's weights
4. Prediction model - which is same as model 1 but with weights learned from model 3use dto predict on test set.

"""



import argparse
from lambeq.text2diagram.ccg_parser import CCGParser
from lambeq.ansatz import BaseAnsatz
from lambeq.training.model import Model
from lambeq.training.trainer import  Trainer
import os
import os.path
import tensorflow as tf
from lambeq import RemoveCupsRewriter
from tqdm import tqdm
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import torch
import torchmetrics
from torchmetrics import F1Score
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
from lambeq import TensorAnsatz,SpiderAnsatz,Sim15Ansatz, IQPAnsatz,Sim14Ansatz
from lambeq import BobcatParser,spiders_reader
from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset, TreeReader,PennyLaneModel
import wget
import wandb
from pytket.extensions.qiskit import AerBackend
from lambeq import BinaryCrossEntropyLoss
import numpy as np
import keras_tuner
import keras
from keras import layers
import os.path




def f1(y_hat, y):
    f1 = F1Score(task="binary", num_classes=2, threshold=0.5)
    return f1(y_hat, y)


def accuracy(y_hat, y):
        sig = torch.sigmoid
        assert type(y_hat)== type(y)
        # half due to double-counting
        #todo/confirm what does he mean by double counting
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  


"""go through all the circuits in training data, 
    and pick the one which has highest type value
    i.e the symbol which has the highest vector dimension (if using a classical anstaz)
      or number of 
    qbits(if using quantum ansatz)
    """
def get_max_word_param_length_spider_ansatz(input_circuits):
        lengths=[]
        for d in tqdm(input_circuits, total=len(input_circuits)):
            for symb in d.free_symbols:
                x =  symb.name.split('_', 1)[1]
                y = x.split('__')[0]
                lengths.append(int(y))
        return lengths

def get_max_word_param_length_all_other_ansatz(input_circuits):
        lengths=[]
        for d in input_circuits:
            for symb in d.free_symbols:                               
                lengths.append(int(symb.name[-1]))
        return lengths

            
"""
given a word from training and dev vocab, get the corresponding embedding using fast text. 

:param vocab: vocabulary
:return: returns a dictionary of each word and its corresponding embedding
""" 
def get_vocab_emb_dict(vocab,embedding_model):            
            embed_dict={}
            for wrd in vocab:                
                cleaned_wrd_just_plain_text,cleaned_wrd_with_type=clean_wrd_for_spider_ansatz_coming_from_vocab(wrd)
                if cleaned_wrd_with_type in embed_dict   :                   
                    print(f"error.  the word {cleaned_wrd_with_type} was already in dict")
                else:
                    embed_dict[cleaned_wrd_with_type]= embedding_model[cleaned_wrd_just_plain_text] 
            return embed_dict




"""
given a set of circuits (e.g. train_circuit) extract the word from the 
symbol and create a dictionary full of it

:param vocab: circuits- created from diagrams using an anstaz
:return: returns a set of all unique words associated i.e vocab
""" 
def create_vocab_from_circuits(circuits,args):
    vocab=set()
    if(args.ansatz==SpiderAnsatz):  
        for d in circuits:
            for symb in d.free_symbols:                  
                    cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symb.name)                
                    vocab.add(cleaned_wrd_with_type)
    else:
        vocab = {symb.name.rsplit('_', 1)[0] for d in circuits for symb in d.free_symbols}        
    
    return vocab

"""
Initialize the weights of the given model (e.g. phases of the gates). This has to be done
because some words in the dev/test set can be out of vocabulary. So for those we need
to extract the corresponding embedding.

Note: this is a redundant step+
as of nov21st2024 - is hurting the first model's fit- i.e loss not reducing.

:param train_circuits: All training data in a circuit format
:param val_circuits: All dev/val data in a circuit format
:param embedding_model: Model used to create embeddings given a word (e.g. Fasttext)
:param qnlp_model: The model1 - main qnlp model which trains on training data 

:return: returns a dictionary of each word and its corresponding embedding

"""

def generate_initial_parameterisation(train_circuits, val_circuits, embedding_model, qnlp_model,args):   

    
    train_vocab=create_vocab_from_circuits(train_circuits,args)
    val_vocab=create_vocab_from_circuits(val_circuits,args)
    print(len(val_vocab.union(train_vocab)), len(train_vocab), len(val_vocab))    
    print(f"OOV word count: i.e out of {len(val_vocab)} words in the testing vocab there are  {len(val_vocab - train_vocab)} words that are not found in training. So they are OOV")
    oov_words=val_vocab - train_vocab
    # print(f"list of OOV words are {oov_words}")     

    #calculate all out of vocabulary word count. Note: aldea is a word while aldea_0__s is a symbol
    set_val={symb.name for d in val_circuits for symb in d.free_symbols}
    set_train = {symb.name for d in train_circuits for symb in d.free_symbols}
    oov_symbols= set_val - set_train
    n_oov_symbs = len(oov_symbols)
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in val_circuits for symb in d.free_symbols})}')
    print(f"OOV symbol count: i.e out of {len(set_train)} words in the val vocab there are  {n_oov_symbs} symbols that are not found in training. So they are OOV")
    # print(f"the symbols that are in symbol count but not in word count are:{oov_symbols-oov_words}")

   
    max_word_param_length=0

    if(args.ansatz==SpiderAnsatz):
        max_word_param_length_train = max(get_max_word_param_length_spider_ansatz(train_circuits))
        max_word_param_length_val = max(get_max_word_param_length_spider_ansatz(val_circuits))

    else: 
        max_word_param_length_train = max(get_max_word_param_length_all_other_ansatz(train_circuits))
        max_word_param_length_val = max(get_max_word_param_length_all_other_ansatz(val_circuits))

    max_word_param_length = max(max_word_param_length_train, max_word_param_length_val) + 1


    """ max param length should include a factor from dimension
          for example if bakes is n.r@s, and n=2 and s=2, the parameter 
          length must be 4. """
    max_word_param_length = max_word_param_length * max (args.base_dimension_for_noun,args.base_dimension_for_sent,args.base_dimension_for_prep_phrase)


    assert max_word_param_length!=0
    
    if(args.ansatz==SpiderAnsatz):               
        #for each word in train and test vocab get its embedding from fasttext
        train_vocab_embeddings = get_vocab_emb_dict(train_vocab,embedding_model)
        val_vocab_embeddings = get_vocab_emb_dict(val_vocab,embedding_model)            
    else:
        #for the words created by other ansatz other formatting is different
        train_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in train_vocab}
        val_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in val_vocab}


    #create a list to store all the initial weights
    initial_param_vector = []

    
    for sym,weight in zip(qnlp_model.symbols, qnlp_model.weights):
        """#for each qbit, the initial parameter size changes.
        # i.e if aldea had both aldea_0_s and aldea_1_s
        # its initial param will have two difference entries in the initial param vector
        Note that this is coming from deep qnlp_model.symbols where each qbit in a given
        word is stored separately. refer this symbols lambeq
        documentation:https://cqcl.github.io/lambeq-docs/tutorials/training-symbols.html"""
        if(args.ansatz==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            
            
            if cleaned_wrd_with_type in train_vocab_embeddings:
                if args.model == PytorchModel:                                                        
                    # dimension of initial param vector is decided by the actual dimension assigned in qnlp.weights for that word
                    list_of_params_for_this_word=[]
                    for i in range(len(weight)):
                        assert len(train_vocab_embeddings[cleaned_wrd_with_type]) > i
                        val= train_vocab_embeddings[cleaned_wrd_with_type][int(i)]
                        list_of_params_for_this_word.append(val)
                    tup = torch.tensor (list_of_params_for_this_word, requires_grad=False) #initializing with first two values of the embedding
                    initial_param_vector.append(tup)
                else:
                    initial_param_vector.append(train_vocab_embeddings[cleaned_wrd_with_type][int(idx)])
            else:                            
                print(f"ERROR: found that this word {cleaned_wrd_with_type} was OOV from train vocab")
             
    
    # assert len(qnlp_model.weights) == len(initial_param_vector)
    # # also assert dimension of every single symbol/weight matches that of initial_para_vector
    # for x,y in zip(qnlp_model.weights, initial_param_vector):
    #     assert len(x) == len(y)
    # qnlp_model.weights = nn.ParameterList(initial_param_vector)

    return train_vocab_embeddings, val_vocab_embeddings, max_word_param_length, n_oov_symbs

def clean_wrd_for_spider_ansatz(wrd):
    split_word= wrd.lower().split('_')
    cleaned_wrd=split_word[0].replace("(","").replace(")","").replace('\\','').replace(",","")
    wrd_with__s= cleaned_wrd+"__" +split_word[3]
    return cleaned_wrd, wrd_with__s

 
#in vocab the wrd is already stored in the form aldea__s. Weneed to extract plain text from it for finding embedding
def clean_wrd_for_spider_ansatz_coming_from_vocab(wrd):
    split_word= wrd.lower().split('_')
    cleaned_wrd=split_word[0].replace("(","").replace(")","").replace('\\','').replace(",","")    
    wrd_with__s= cleaned_wrd+"__" +split_word[2]
    return cleaned_wrd, wrd_with__s

"""
Args:
    trained_qnlp_model- the trained_qnlp_model
    train_vocab_embeddings- the initial embeddings for words in the vocab got from fasttext
    max_word_param_length- what is the maximum size of a wor

    Returns:
        a map between each word and its latest final weights
    """
def trained_params_from_model(trained_qnlp_model, train_embeddings, max_word_param_length):  

    trained_param_map = { symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
    trained_parameterisation_map = {wrd: np.zeros(max_word_param_length) for wrd in train_embeddings}

    cleaned_wrd_with_type=""
    for symbol, train_val in trained_param_map.items():        
        if(ansatz_to_use==SpiderAnsatz):              
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symbol.name)
            rest = symbol.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
        else:
            wrd, idx = symbol.name.rsplit('_', 1)
        
        
        assert cleaned_wrd_with_type != ""
        if cleaned_wrd_with_type in trained_parameterisation_map:
            trained_parameterisation_map[cleaned_wrd_with_type][int(idx)] = train_val

    return trained_parameterisation_map

def generate_OOV_parameterising_model(trained_qnlp_model, train_vocab_embeddings, max_word_param_length,args):
    """
    in the previous func `generate_initial_parameterisation` we took model 1 i.e the QNLP model
    and initialized its weights with the embeddings of the words

    here we will take the model 1, and create another NN i.e model 3, which will learn the mapping between
    train_vocab_embeddings and weights of trained_qnlp_model
    Args:
    trained_qnlp_model- the trained_qnlp_model
    train_vocab_embeddings- the initial embeddings for words in the vocab got from fasttext
    max_word_param_length- what is the maximum size of a word

    Returns:
    Weights of a NN model which now has learnt
      for each word in fasttext as its original embedding mapped to the weights in the trained QNLP model

    """    
    dict1 = {symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
    dict2 = {wrd: np.zeros(max_word_param_length) for wrd in train_vocab_embeddings}
    
    cleaned_wrd_with_type=""
    for symbol, trained_weights in dict1.items():
        if(args.ansatz==SpiderAnsatz):  
            #symbol and word are different. e.g. aldea_0. From this extract the word
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symbol.name)
            rest = symbol.name.split('_', 1)[1]
            idx = rest.split('__')[0]       
        else:
            cleaned_wrd_with_type, idx = symbol.name.rsplit('_', 1)
    
        assert cleaned_wrd_with_type  != "" 
       
       
        if cleaned_wrd_with_type in dict2:               
                dict2[cleaned_wrd_with_type] = trained_weights.detach().numpy()
        else:                
                print(f"inside OOV_generation-found that this word {cleaned_wrd_with_type} was not in trained_param_vectors")

                

    wrds_in_order = list(train_vocab_embeddings.keys())


    NN_train_X = np.array([train_vocab_embeddings[wrd] for wrd in wrds_in_order])

    #if the weight vector is less than max_param_length (e.g. 4) , pad the rest with zeroes
    NN_train_Y=[]
    for wrd in wrds_in_order:
        if len(dict2[wrd])==max_word_param_length:
            NN_train_Y.append(dict2[wrd])
        else:            
            pad= np.zeros(max_word_param_length-len(dict2[wrd]))
            combined = np.hstack([dict2[wrd],pad])                          
            NN_train_Y.append(combined)                                
    
    if (args.do_model3_tuning):
        build_model(keras_tuner.HyperParameters())  
    
        tuner = keras_tuner.GridSearch(
            hypermodel=build_model,
            objective="val_accuracy", #todo: replace this with F1 score
            max_trials=10,
            executions_per_trial=2,
            overwrite=True,
            directory="tuning_model",
            project_name="oov_model3",
        )
        
        tuner.search(NN_train_X, np.array(NN_train_Y),validation_split=0.2, verbose=2, epochs=EPOCHS_MODEL3_OOV_MODEL)
        print(tuner.search_space_summary())
        models = tuner.get_best_models(num_models=2)
        

        best_model = models[0]
        best_model.summary()
        
        print(tuner.results_summary())
    else:
        OOV_NN_model = keras.Sequential([
        layers.Dense(int((max_word_param_length + args.maxparams) / 2), activation='tanh'),
        layers.Dense(max_word_param_length, activation='tanh'),
        ])

        #standard keras stuff, initialize and tell what loss function and which optimizer will you be using
        OOV_NN_model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=10,
            verbose=2,
            mode='auto',
            baseline=None,
            restore_best_weights=True,
            start_from_epoch=0
        )
    
        OOV_NN_model.build(input_shape=(None, args.maxparams))

   
        hist = OOV_NN_model.fit(NN_train_X, np.array(NN_train_Y), validation_split=0.2, verbose=1, epochs=args.epochs_model3_oov_model,callbacks=[callback])
        print(hist.history.keys())
        print(f'OOV NN model final epoch loss: {(hist.history["loss"][-1], hist.history["val_loss"][-1])}')
        plt.plot(hist.history['loss'], label='loss')
        plt.plot(hist.history['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.legend()
        # plt.show() #code is expecting user closing the picture manually. commenting this temporarily since that was preventing the smooth run/debugging of code

        best_model= OOV_NN_model


    return best_model,dict2


def call_existing_code(lr):
    assert MAX_PARAM_LENGTH > 0
    OOV_NN_model = keras.Sequential([ layers.Dense(int((MAX_PARAM_LENGTH + MAXPARAMS) / 2), activation='tanh'),
      layers.Dense( MAX_PARAM_LENGTH, activation= 'tanh'),
    ])


    OOV_NN_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='mean_absolute_error',
        metrics=["accuracy","f1_score"],
    )
    return OOV_NN_model


def build_model(hp):
    # units_oov = hp.Int("units", min_value=32, max_value=512, step=16)
    # activation_oov =hp.Choice("activation", ["tanh", "relu","sigmoid","selu","softplus", "softmax","elu","exponential","leaky_relu","relu6","silu","hard_silu","gelu","hard_sigmoid","linear","mish","log_softmax"])
    # loss_fn_oov =hp.Choice("loss", ["categorical_crossentropy", "binary_crossentropy","binary_focal_crossentropy","kl_divergence", "sparse_categorical_crossentropy","poisson","mean_squared_error","hinge","mean_absolute_error"])
    # optimizers_oov =hp.Choice("optimizer", ["adam", "SGD","rmsprop","adamw","adadelta", "adagrad","adamax","adafactor","ftrl","lion","lamb"])
    # dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-6, max_value=1e-1, step=10, sampling="log")
    
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(lr=lr)
    return model

def evaluate_val_set(pred_model, val_circuits, val_labels, trained_weights, val_vocab_embeddings, max_word_param_length, args,OOV_strategy='random', OOV_model=None):   
    pred_parameter_map = {}

    #Use the words from train wherever possible, else use DNN prediction
    for wrd, embedding in val_vocab_embeddings.items():
        if OOV_strategy == 'model':
            pred_parameter_map[wrd] = trained_weights.get(wrd, OOV_model.predict(np.array([embedding]), verbose=1)[0])
        elif OOV_strategy == 'embed':
            pred_parameter_map[wrd] = trained_weights.get(wrd, embedding)
        elif OOV_strategy == 'zeros':
            pred_parameter_map[wrd] = trained_weights.get(wrd, np.zeros(max_word_param_length))
        else:
            pred_parameter_map[wrd] = trained_weights.get(wrd, 2 * np.random.rand(max_word_param_length)-1)

    
    #convert the dictionary pred_parameter_map into a list pred_weight_vector
    pred_weight_vector = []

    assert len(pred_model.symbols) == len(pred_model.weights)

    for sym,weight in zip(pred_model.symbols,pred_model.weights):
        if(args.ansatz==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            if cleaned_wrd_with_type in pred_parameter_map:
                if args.model == PytorchModel:
                    pred_weight_vector.append(pred_parameter_map[cleaned_wrd_with_type])

                    
                    

    
    assert len(pred_model.symbols) == len(pred_weight_vector)
    assert type(pred_model.weights) == type( nn.ParameterList(pred_weight_vector))
    #also assert dimension of every single symbol/weight matches that of initial_para_vector
    for x,y in zip(pred_model.weights, pred_weight_vector):
        assert len(x) == len(y)  
    pred_model.weights = nn.ParameterList(pred_weight_vector)

    
    #use the model now to create predictions on the test set.
    preds = pred_model.get_diagram_output(val_circuits)
    loss_pyTorch =torch.nn.BCEWithLogitsLoss()
    loss_val= loss_pyTorch(preds, torch.tensor(val_labels))
    acc_val=accuracy(preds, torch.tensor(val_labels))
    f1score_val= f1(preds,torch.tensor(val_labels),)

    return loss_val, acc_val, f1score_val


def read_glue_data(dataset_downloaded,split,lines_to_read=0):
        assert lines_to_read != 0
        line_counter=0
        labels, sentences = [], []
        desc_dynamic= f"reading {split} data"
        
        for line in tqdm(dataset_downloaded[split], desc=desc_dynamic, total=len(dataset_downloaded[split])):                                                    
                t = float(line['label']) 
                labels.append([t, 1-t])           
                sentences.append(line['sentence'])
                line_counter+=1
                if (line_counter> lines_to_read):
                    break 
        return labels, sentences




def read_data(filename):         
            labels, sentences = [], []
            with open(filename) as f:
                for line in f:           
                    t = float(line[0])            
                    labels.append([t, 1-t])            
                    sentences.append(line[1:].strip())
            return labels, sentences


def convert_to_diagrams_with_try_catch(parser_obj,list_sents,labels,tokeniser, split="train"):
    list_target = []
    labels_target = []
    sent_count_longer_than_32=0
    skipped_sentences_counter_due_to_cant_parse=0
    desc_long = f"converting {split} data to diagrams"
    for sent, label in tqdm(zip(list_sents, labels),desc=desc_long,total=len(list_sents)):                        
        tokenized_sent = tokeniser.tokenise_sentence(sent)                
        #when we use numpy, max size of array is 32- update. even in quantum computer
        if len(tokenized_sent)> 29:                
                 sent_count_longer_than_32+=1
                 continue
        try:
            if(parser_obj==spiders_reader):
                 sent_diagram = parser_obj.sentence2diagram(tokenized_sent, tokenised=True)
            else:
                 sent_diagram = parser_obj.sentence2diagram(tokenized_sent, suppress_exceptions=True, tokenised=True)
        except Exception as ex:             
            print(ex)
            skipped_sentences_counter_due_to_cant_parse+=1
            continue
        if(sent_diagram):
            list_target.append(sent_diagram)
            labels_target.append(label)
        else:
             print("found that there was a sentence which after conversion to diagram was None. i.e hit exception during parsing.")
    
    print(f"sent_count_longer_than_32={sent_count_longer_than_32}")
    print(f"out of a total of ={len(list_sents)} sentences {skipped_sentences_counter_due_to_cant_parse} were skipped because they were unparsable")
    print("no. of items processed= ", len(list_target))
    return list_target, labels_target


def convert_diagram_to_circuits_with_try_catch(diagrams, ansatz, labels,split):
    list_circuits =[]
    list_labels = []
    assert len(diagrams) == len(labels)
    desc_long =f"converting diagrams of {split} data to circuits"
    counter_skipped_data =0
    for diagram,label in tqdm(zip(diagrams,labels), desc=desc_long, total=len(labels)):
        try:
            circuit= ansatz(diagram)
        except:
            counter_skipped_data+=1
            continue
        list_circuits.append(circuit)
        list_labels.append(label)
    print(f"out of {len(labels)} data points in {split} {counter_skipped_data} were skipped since they couldnt be converted to circuits")
    return list_circuits, list_labels


def run_experiment(args,train_diagrams, train_labels, val_diagrams, val_labels,test_diagrams,test_labels,  eval_metrics,seed,embedding_model):
    if args.ansatz in [IQPAnsatz,Sim15Ansatz, Sim14Ansatz]:
        ansatz = args.ansatz ({AtomicType.NOUN: args.base_dimension_for_noun,
                    AtomicType.SENTENCE: args.base_dimension_for_sent,
                    AtomicType.PREPOSITIONAL_PHRASE: args.base_dimension_for_prep_phrase} ,n_layers= args.no_of_layers_in_ansatz,n_single_qubit_params=args.single_qubit_params)    
    else:
        ansatz = args.ansatz ({AtomicType.NOUN: Dim(args.base_dimension_for_noun),
                    AtomicType.SENTENCE: Dim(args.base_dimension_for_sent),
                    AtomicType.PREPOSITIONAL_PHRASE: args.base_dimension_for_prep_phrase}  )    

   
    assert len(train_diagrams) == len(train_labels)
    #use the anstaz to create circuits from diagrams
    train_circuits, train_labels =  convert_diagram_to_circuits_with_try_catch(diagrams=train_diagrams, ansatz=ansatz,labels=train_labels, split="train")        
    assert len(train_circuits) == len(train_labels)


    val_circuits, val_labels =  convert_diagram_to_circuits_with_try_catch(diagrams=val_diagrams, ansatz=ansatz,labels=val_labels, split="val")
    assert len(val_circuits) == len(val_labels)

    test_circuits, test_labels =  convert_diagram_to_circuits_with_try_catch(diagrams=test_diagrams, ansatz=ansatz,labels=test_labels, split="test")
    assert len(test_circuits) == len(test_labels)
    
    assert len(train_circuits) > 0
    assert len(val_circuits) > 0
    assert len(test_circuits) > 0

    print("length of each circuit in train is:")
    print([len(x) for x in train_circuits])

    if(args.model==TketModel):
        backend = AerBackend()
        backend_config = {
                    'backend': backend,
                    'compilation': backend.default_compilation_pass(2),
                    'shots': 8192
                }
        qnlp_model= TketModel.from_diagrams(train_circuits, backend_config=backend_config)

    elif(args.model==PennyLaneModel): #to run on an actual quantum computer
        
        from qiskit_ibm_provider import IBMProvider

        # Save the account, use overwrite=True if necessary
        IBMProvider.save_account(token='dab4b9f2ebfe284f0bd397651343563794ef7c2dfe99294a258a99c34c8be5dbd3a88e149b43ade9c9002fa8b071a615edcb58c8971e72e1348df58577b5d65a', overwrite=True)
        backend_config = {'backend': 'qiskit.ibmq',
                        'device': 'ibm_brisbane',
                        'shots': 1000}
        qnlp_model = PennyLaneModel.from_diagrams(train_circuits,
                                       probabilities=True,
                                       normalize=True,
                                       backend_config=backend_config)
        qnlp_model.initialise_weights()

    else:
        qnlp_model = args.model.from_diagrams(train_circuits + val_circuits)

    train_dataset = Dataset(
                train_circuits,
                train_labels,
                batch_size=args.batch_size)

    val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

    print(f"length of train_labels is {len(train_labels)} and there are  {len(train_circuits)} circuits in training")
    print(f"there are {len(train_circuits)} circuits currently in training, {len(val_circuits)} in val, and {len(test_circuits)} in testing")
    assert len(train_circuits)== len(train_labels)
    assert len(val_circuits)== len(val_labels)
    assert len(test_circuits)== len(test_labels)


    if(args.trainer==QuantumTrainer):
        trainer = QuantumTrainer(
        model=qnlp_model,
        loss_function=BinaryCrossEntropyLoss(),
        epochs=args.epochs_train_model1,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.001*args.epochs_train_model1}, #todo: move this abc values to argparse defaults
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        log_dir='RelPron/logs',
        seed=seed
        )
    else:
        trainer = args.trainer(
            model=qnlp_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=args.learning_rate_model1,            
            epochs=args.epochs_train_model1,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=seed)

    

    train_embeddings, val_embeddings, max_w_param_length, oov_word_count = generate_initial_parameterisation(
        train_circuits, val_circuits, embedding_model, qnlp_model,args=args)

    global MAX_PARAM_LENGTH
    MAX_PARAM_LENGTH = max_w_param_length
    print(qnlp_model.weights[0])
    print(type(train_dataset.targets[0]))

    trainer.fit(train_dataset, val_dataset,eval_interval=1, log_interval=1)
    print(qnlp_model.weights[0])

    print("***********Training of first model completed**********")
    """if there are no OOV words, we dont need the model 2 through model 4. 
    just use model 1 to evaluate and exit"""
    if oov_word_count==0:
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


        val_preds = qnlp_model.get_diagram_output(val_circuits)    
        loss_pyTorch =torch.nn.BCEWithLogitsLoss()
        val_loss= loss_pyTorch(val_preds, torch.tensor(val_labels))
        val_acc =accuracy(val_preds, torch.tensor(val_labels))
        print(f"value of val_loss={val_loss} and value of val_acc ={val_acc}")

        # print test accuracy- not the value above and below must be theoretically same, but isnt todo: find out why
        val_acc = accuracy(qnlp_model(val_circuits), torch.tensor(val_labels))
        print('Val accuracy:', val_acc.item())
    
        import sys
        sys.exit()

   
    NN_model,trained_wts = generate_OOV_parameterising_model(qnlp_model, train_embeddings, max_w_param_length,args)
    prediction_model = args.model.from_diagrams(val_circuits)

    trainer = args.trainer(
            model=prediction_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=args.learning_rate_model3,
            epochs=args.epochs_train_model1,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=seed)


    smart_loss, smart_acc, smart_f1 = evaluate_val_set(prediction_model,
                                                val_circuits,
                                                val_labels,
                                                trained_wts,
                                                val_embeddings,
                                                max_w_param_length,
                                                args,
                                                OOV_strategy='model',
                                                OOV_model=NN_model)
    print(f"value of smart_loss={smart_loss} , value of smart_acc ={smart_acc} value of smart_f1 ={smart_f1}")
    print('Evaluating EMBED model')

    
    res =  {
            'NN': (smart_loss, smart_acc)
            
            
           }
    
    return smart_loss.item(), smart_acc.item()


    
def perform_task(args):
    embedding_model= None
    if(args.dataset in ["uspantek", "spanish"]):
        # todo add wget ('wget -c https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1 -O ./embeddings-l-model.bin')
        embedding_model = ft.load_model('./embeddings-l-model.bin')
    else: 
        if not (os.path.isfile('cc.en.300.bin')):
            #todo: move this all to run_me_first.sh
            filename = wget.download(" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
        embedding_model= ft.load_model('cc.en.300.bin')

   

    parser_obj = args.parser

    #spiders reader, we are directly using the reader, whilbobcat needs someone to create an obj of it
    assert embedding_model!=None
    if(args.parser==BobcatParser):
        parser_obj=BobcatParser(verbose='text',root_cats=['N','NP','S'])

    

    

    #setting a flag for TESTING so that it is done only once.
    #  Everything else is done on train and dev
    TESTING = False



    if(args.dataset== "uspantek"):
        TRAIN="uspantek_train.txt"
        DEV="uspantek_dev.txt"
        TEST="uspantek_test.txt"
        

    if(args.dataset== "spanish"):
        TRAIN="spanish_train.txt"
        DEV="spanish_dev.txt"
        TEST="spanish_test.txt"
        

    if(args.dataset== "msr_paraphrase_corpus"):
        TRAIN="msr_paraphrase_train.txt"
        DEV="msr_paraphrase_test.txt"
        TEST="msr_paraphrase_test.txt"
        type_of_data = "pair"

    if(args.dataset== "food_it"):
        TRAIN="mc_train_data.txt"
        DEV="mc_dev_data.txt"
        TEST="mc_test_data.txt"
        

    # a unique name to identify this run inside wandb data and graph
    arch = f"{args.ansatz}+'_'+{args.dataset}+'_'+{args.parser}+'_'+{args.trainer}+'_'+{args.model}+'_'+{embedding_model}"

    wandb.init(    
        project="qnlp_nov2024_expts",    
        config={
        "learning_rate_model1": args.learning_rate_model1,
        "architecture": arch,
        "BASE_DIMENSION_FOR_NOUN".lower(): args.base_dimension_for_noun ,
        "BASE_DIMENSION_FOR_SENT".lower():args.base_dimension_for_sent,
        "MAXPARAMS".lower() :args.maxparams,
        "BATCH_SIZE".lower():args.batch_size,
        "EPOCHS".lower() : args.epochs_train_model1,
        "LEARNING_RATE_model3".lower() : args.learning_rate_model3,
        "SEED".lower() : args.seed , #todo: either pick this seed in args, or the one before calling run_expt. Not both
        "DATA_BASE_FOLDER".lower():args.data_base_folder,
        "EPOCHS_DEV".lower():args.epochs_model3_oov_model,
        "TYPE_OF_DATA_TO_USE".lower():args.dataset,
        "embedding_model_to_use".lower():embedding_model
        })


    
    eval_metrics = {"acc": accuracy, "F1":f1 }
    spacy_tokeniser = SpacyTokeniser()

    if args.dataset  in ["uspantek","spanish"]:
        spanish_tokeniser=spacy.load("es_core_news_sm")
        spacy_tokeniser.tokeniser = spanish_tokeniser
    else:
        english_tokenizer = spacy.load("en_core_web_sm")
        spacy_tokeniser.tokeniser =english_tokenizer
    
    if(args.dataset=="sst2"):
        ds = load_dataset("nyu-mll/glue", "sst2")
        train_labels, train_data = read_glue_data(ds,split="train", lines_to_read= args.no_of_training_data_points_to_use)
        val_labels, val_data = read_glue_data(ds,split="validation", lines_to_read= args.no_of_val_data_points_to_use)
        test_labels, test_data = read_glue_data(ds, split="test", lines_to_read= args.no_of_test_data_points_to_use)

    else:
        #read the base data, i.e plain text english.
        train_labels, train_data = read_data(os.path.join(args.data_base_folder,TRAIN))
        val_labels, val_data = read_data(os.path.join(args.data_base_folder,DEV))
        test_labels, test_data = read_data(os.path.join(args.data_base_folder,TEST))



    """#some datasets like spanish, uspantek, sst2 have some sentences which bobcat doesnt like. putting it
    in a try catch, so that code doesnt completely halt/atleast rest of the dataset can be used
    """
    if (args.dataset in ["uspantek","sst2","spanish"]):
        train_diagrams, train_labels = convert_to_diagrams_with_try_catch(parser_obj,train_data,train_labels,spacy_tokeniser, split="train")
        val_diagrams, val_labels= convert_to_diagrams_with_try_catch(parser_obj,val_data,val_labels,spacy_tokeniser,split="val")
        test_diagrams, test_labels = convert_to_diagrams_with_try_catch(parser_obj,test_data,test_labels,spacy_tokeniser,split="test")
    else:
        #convert the plain text input to ZX diagrams
        train_diagrams = parser_obj.sentences2diagrams(train_data, suppress_exceptions=True)
        val_diagrams = parser_obj.sentences2diagrams(val_data,suppress_exceptions=True)
        test_diagrams = parser_obj.sentences2diagrams(test_data,suppress_exceptions=True)

    train_X = []
    val_X = []

    print(f"count of train, test, val elements respectively are: ")
    print({len(train_diagrams)}, {len(test_diagrams)}, {len(val_diagrams)})
    assert len(train_diagrams)== len(train_labels)
    assert len(val_diagrams)== len(val_labels)
    assert len(test_diagrams)== len(test_labels)
    
    # if not args.ansatz==SpiderAnsatz: #for some reason spider ansatz doesnt like you removing cups
    #   remove_cups = RemoveCupsRewriter()
    #   train_X = []
    #   val_X = []
    #   for d in tqdm(train_diagrams):
    #       train_X.append(remove_cups(d).normal_form())

    #   for d in tqdm(val_diagrams):    
    #       val_X.append(remove_cups(d).normal_form())

    #   train_diagrams  = train_X
    #   val_diagrams    = val_X





        
    """todo: why is he setting random seed, that tooin tensor flow
    - especially since am using a pytorch model."""



    #ideally should be tested over more than 1 seed and layers 1 throught 3 minimally- and average take. 
    # But  commenting out due to lack of ram in laptop 
    tf_seed = args.seed
    tf.random.set_seed(tf_seed)

    return run_experiment(args,train_diagrams, train_labels, val_diagrams, val_labels,test_diagrams,test_labels, eval_metrics,tf_seed,embedding_model)
       


def parse_arguments():
    parser = argparse.ArgumentParser(description="Description of your script.")
    parser.add_argument('--dataset', type=str, required=False, default="sst2" ,help="type of dataset-choose from [sst2,uspantek,spanish,food_it,msr_paraphrase_corpus,sst2")
    parser.add_argument('--parser', type=CCGParser, required=False, default=spiders_reader, help="type of parser to use: [tree_reader,bobCatParser, spiders_reader,depCCGParser]")
    parser.add_argument('--ansatz', type=BaseAnsatz, required=False, default=SpiderAnsatz, help="type of ansatz to use: [IQPAnsatz,SpiderAnsatz,Sim14Ansatz, Sim15Ansatz,TensorAnsatz ]")
    parser.add_argument('--model', type=Model, required=False, default=PytorchModel , help="type of model to use: [numpy, pytorch,TketModel]")
    parser.add_argument('--trainer', type=Trainer, required=False, default=PytorchTrainer, help="type of trainer to use: [PytorchTrainer, QuantumTrainer]")
    parser.add_argument('--max_param_length_global', type=int, required=False, default=0, help="a global value which will be later replaced by the actual max param length")
    parser.add_argument('--do_model3_tuning', type=bool, required=False, default=False, help="only to be used during training, when a first pass of code works and you need to tune up for parameters")
    parser.add_argument('--base_dimension_for_noun', type=int, default=2, required=False, help="")
    parser.add_argument('--base_dimension_for_sent', type=int, default=2, required=False, help="")
    parser.add_argument('--base_dimension_for_prep_phrase', type=int, default=2, required=False, help="")
    parser.add_argument('--maxparams', type=int, default=300, required=False, help="")
    parser.add_argument('--batch_size', type=int, default=30, required=False, help="")
    parser.add_argument('--epochs_train_model1', type=int, default=30, required=False, help="")
    parser.add_argument('--epochs_model3_oov_model', type=int, default=100, required=False, help="")
    parser.add_argument('--learning_rate_model1', type=float, default=3e-2, required=False, help="")
    parser.add_argument('--seed', type=int, default=0, required=False, help="")
    parser.add_argument('--data_base_folder', type=str, default="data", required=False, help="")
    parser.add_argument('--learning_rate_model3', type=float, default=3e-2, required=False, help="")
    parser.add_argument('--no_of_layers_in_ansatz', type=int, default=3, required=False, help="")
    parser.add_argument('--no_of_training_data_points_to_use', type=int, default=20, required=False, help="65k of sst data was taking a long time. temporarily training on a smaller data")
    parser.add_argument('--no_of_val_data_points_to_use', type=int, default=10, required=False, help="65k of sst data was taking a long time. temporarily training on a smaller data")
    parser.add_argument('--no_of_test_data_points_to_use', type=int, default=10, required=False, help="65k of sst data was taking a long time. temporarily training on a smaller data")
    parser.add_argument('--single_qubit_params', type=int, default=3, required=False, help="")
    
    return parser.parse_args()




def main():
    args = parse_arguments()
    perform_task(args)

if __name__=="__main__":
        main()
     
        
        
