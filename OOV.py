# -*- coding: utf-8 -*-
"""v7
#name: v7_*
# Code which takes uspantekan or spanish small data (100 sent) about two classes
education and dancing, runs it through a QNLP model, which is supported by a fasttext model,
and two neural network models to learn and make prediction.
#to get blow by blow details of what this code does, refer to a section named
"how this code runs" inside the project plan
https://github.com/ua-datalab/QNLP/blob/main/Project-Plan.md

4 major models used in this code. 
1. QNLP model, called model1
2. Fast text embedding model , called model 2
3. NN model that learns mapping between fast text embedding and QNLP trained model's weights
4. Prediction model - which is use dto predict on test set.
#todo, find why not just do model1.predict?
"""


import wandb
from tqdm import tqdm
# wandb.init(project="v4_uspantekan")
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
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
from lambeq import AtomicType
from lambeq import Dataset
from lambeq import PytorchModel, NumpyModel, TketModel, PennyLaneModel
from lambeq import TensorAnsatz,SpiderAnsatz
from lambeq import BobcatParser,spiders_reader

parser_to_use = spiders_reader  #[BobcatParser(verbose='text'), spiders_reader]
ansatz_to_use = SpiderAnsatz #[IQP, Sim14, Sim15,TensorAnsatz ]
model_to_use=  PytorchModel #[numpy, pytorch]
trainer_to_use= PytorchTrainer #[PytorchTrainer, QuantumTrainer]

embedding_model = ft.load_model('./embeddings-l-model.bin')

#todo: find what maxparams is for
#update: answer: it is the maximum qbits (or dimensions of the tensor, as your case be)
MAXPARAMS = 300
BATCH_SIZE = 32
EPOCHS = 2
LEARNING_RATE = 0.1
SEED = 43434
DATA_BASE_FOLDER= "data"

USE_MRPC_DATA=False
USE_SPANISH_DATA=True
USE_USP_DATA=False

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

#todo: MRPC is a NLi kind of task. Use the 2 classes of information technology and food thing dataset instead if you want something for testing
if(USE_MRPC_DATA):
    TRAIN="mrpc_train_80_sent.txt"
    DEV="mrpc_dev_10_sent.txt"
    TEST="mrpc_test_10sent.txt"

# loss = lambda y_hat, y: -np.sum(y * np.log(y_hat)) / len(y)  # binary cross-entropy loss
# acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
sig = torch.sigmoid

def accuracy(y_hat, y):
        assert type(y_hat)== type(y)
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  # half due to double-counting

eval_metrics = {"acc": accuracy}
spacy_tokeniser = SpacyTokeniser()

if(USE_SPANISH_DATA) or (USE_USP_DATA):
    spanish_tokeniser=spacy.load("es_core_news_sm")
    spacy_tokeniser.tokeniser = spanish_tokeniser



#for english tokenizer
if(USE_MRPC_DATA):
    english_tokenizer = spacy.load("en_core_web_sm")
    spacy_tokeniser.tokeniser =english_tokenizer


import os


"""go through all thecircuits in training data, 
    and pick the one which has highest type value
    note that they are not using the literal length of the circuit, but
    the number attached to next to aldea_2...todo : find what exactly that does"""
def get_max_word_param_length(input_circuits):
        lengths=[]
        for d in input_circuits:
            for symb in d.free_symbols:
                x =  symb.name.split('_', 1)[1]
                y = x.split('__')[0]
                lengths.append(int(y))
        return lengths

def get_vocab_emb_dict(vocab):
            #i.e given a word from training and dev vocab, get the corresponding 
            # embedding using fast text. 
            #all this is stored as a key value pair in embed_dict, where the word is
            #the key and embedding is the value
            #todo: confirm if this is how Khatri does it too.
            embed_dict={}
            for wrd in vocab:
                """#spider ansatz alone writes the tokens in its vocabulary with a single underscore first and then a double underscore
                #so we need to parse accordingly
                #todo: some words are already in dictionary- i think this is because of the same
                #  words having multiple versions- mostly likely we shouldn't split the _1_ thing- i am thinking 
                #that denotes the nth TYPE of LIKES kinda adverbs."""
                cleaned_wrd_just_plain_text,cleaned_wrd_with_type=clean_wrd_for_spider_ansatz_coming_from_vocab(wrd)
                if cleaned_wrd_with_type in embed_dict   :
                    """#this shouldn't happen. a) does it happen for english data b) for 
                    fucks sake find out what is _0- is this standard protocol, i.e if a word occurs twice
                    we ignore it? confirm with original khatri code."""
                    print(f"error.  the word {cleaned_wrd_with_type} was already in dict")
                else:
                    embed_dict[cleaned_wrd_with_type]= embedding_model[cleaned_wrd_just_plain_text] 
            return embed_dict


"""
Spider anstaz writes the symbols different than other ansatz.
So creating a separate function itself which will be called only
for spider ansatz
todo: either raise a pull request with LAMBEQ guys or find out 
if they are doing this deliebrately and its my understanding which is lacking.
spider ansatz does: aldea_0__s or así_0__s

while other ansatz:aldea_s_0 #so we can use cleaned_wrd= symb.name.rsplit('_', 1)[0]

also todo: confirm if its a spider ansatz thing only. or are there any other ansatz that does that
Khatri's code uses IQP, Sim14 and Sim15, so we know that doesn't

update @oct29th2024. 
- Found that in khatri code he retains the __s i.e aldea_0__s becomes
aldea__s. So going to put that back in

- And also confirmed. Spider ansatz is messing up the order
In khatri's code using sim 15 he uses the format aldea__s_2 
i.e the s comes first and then 2

todo: clean up accents/utf-8 eventually?
"""
def clean_wrd_for_spider_ansatz(wrd):
    split_word= wrd.lower().split('_')
    cleaned_wrd=split_word[0].replace("(","").replace(")","").replace('\\','').replace(",","")

    #     //-update @oct29th2024.  Found that in khatri code he retains the __s i.e aldea_0__s becomes
    # aldea__s. So going to put that back in
    wrd_with__s= cleaned_wrd+"__" +split_word[3]
    return cleaned_wrd, wrd_with__s

#in vocab the wrd is already stored in the form aldea__s. Weneed to extract plain text from it for finding embedding
def clean_wrd_for_spider_ansatz_coming_from_vocab(wrd):
    split_word= wrd.lower().split('_')
    cleaned_wrd=split_word[0].replace("(","").replace(")","").replace('\\','').replace(",","")

    #     //-update @oct29th2024.  Found that in khatri code he retains the __s i.e aldea_0__s becomes
    # aldea__s. So going to put that back in
    wrd_with__s= cleaned_wrd+"__" +split_word[2]
    return cleaned_wrd, wrd_with__s

"""given a set of circuits (e.g. train_circuit) extract the word from the 
symbol and create a dictionary full of it
Todo: This works only for SpiderAnsatz. So modify code for other ansatz."""
def create_vocab_from_circuits(circuits):
    vocab=set()
    if(ansatz_to_use==SpiderAnsatz):  
        for d in circuits:
            for symb in d.free_symbols: 
                if "aldea" in symb.name:
                    print("found aldea")               
                cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symb.name)                
                vocab.add(cleaned_wrd_with_type)
    return vocab

"""#set the the initial phases of the gates.
Also note that in this function is where we are walking into OOV land.
i.e we check if there are any words that are found only in test/val set
and not in train set.
mithuns comment @26thsep2024
"""
def generate_initial_parameterisation(train_circuits, val_circuits, embedding_model, qnlp_model):

    """ extract the words from the circuits- i.e the training data
    # Note that in this vocab, the same word can have multiple types, which each occur separately
    # todo: what did he mean by same word having multiple types. 
    # is this the likes vs never example mentione din 1958 lambek paper?
    # update@sep29th2024. This is how the free_symbols look like.
    # distorting_0_s- right now we don't know what 0 or s stands for. My
    # guess after reading only 1958 paper is that s is sentence, which is part of the 
    # 2 fundamental units lambek specifices in 1958 paper- n and s"""
    # train_vocab = {symb.name.rsplit('_', 1)[0] for d in train_circuits for symb in d.free_symbols}
    train_vocab=create_vocab_from_circuits(train_circuits)
    val_vocab=create_vocab_from_circuits(val_circuits)

    
    
    #todo print: the total number of words in train, and test+ note it down
    #.answer: for spanish henderson there are 463 words in training and 89 in testing
    # out of the 89 words in test, 33 are not present in training, so they are OOV
    print(len(val_vocab.union(train_vocab)), len(train_vocab), len(val_vocab))    
    print(f"OOV word count: i.e out of {len(val_vocab)} words in the testing vocab there are  {len(val_vocab - train_vocab)} words that are not found in training. So they are OOV")
    oov_words=val_vocab - train_vocab
    print(f"list of OOV words are {oov_words}")     

    #todo: find the meaning of symbol count- what is the difference between symbol count and OOV or train_vocab
    """update @29thOct2024- Symbol is the term used for the word + number of qbits or dimension
    e.g aldea_0__s means, that symbol is for the word aldea for for its 1st qbit/dimension
    similarly aldea_1__s means it is the symbol for the 2nd qbit etc.
    So ideally the count of OOV symbols must be more than that of oov words.
    that is because most of the words will have more than 1 symbols.
    """
    oov_symbols={symb.name for d in val_circuits for symb in d.free_symbols} - {symb.name for d in train_circuits for symb in d.free_symbols}
    n_oov_symbs = len(oov_symbols)
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in val_circuits for symb in d.free_symbols})}')
    print(f"the symbols that are in symbol count but not in word count are:{oov_symbols-oov_words}")

    #######note that everything to do with OOV ends here. So far it was just FYI, there is OOV in this dataset's test partition
    
    max_word_param_length=0
    if(ansatz_to_use==SpiderAnsatz):
        max_word_param_length_train = max(get_max_word_param_length(train_circuits))
        max_word_param_length_test = max(get_max_word_param_length(val_circuits))
        max_word_param_length = max(max_word_param_length_train, max_word_param_length_test) + 1

    assert max_word_param_length!=0

    
    # max_word_param_length = max(, ) + 1
    print(f'Max params/word: {max_word_param_length}')

    """ # next , for each word in train and test vocab , we need to get its embedding from fasttext
    # mithuns comment @26thsep2024: 
    # note that there is some confusion between the input data Khatri used from MRPC
    # as oposed to the spanish one = rather how spider reader is storing it.
    # In MRPC and  bobcat parser, they store it in one format(i think its two underscore
    # while spider parser stores it with one underscore or two dashes or something
    its definitely a bug in their code. However, we bear the brunt since spider ansatz
    is the only one which didnt give errors for spanish data. So eventually this needs to be
    replaced/fixed/single format must be stored for all ansatz- evne see if you can
    create a pull request for this
    """
    if(ansatz_to_use==SpiderAnsatz):  
        # train_vocab_embeddings={}      
        

        """ #for each word in train and test vocab get its embedding from fasttext
        #note that even though the symbols per se have _0_, in the train_vocab_embedding 
        # dictionary it is stored in the
        #format of {wrd: embedding}- i.e only the word aldea out of aldea_0_ is separated out and used.
        # mithuns comment @26thsep2024: note that this is a hack, and ideally such data format
        # based difference shouldnt occur. 
        # TODO: run khastri code on MRPC and confirm who is screwing up.
        # is it spider ansatz which is messing up the data format or is it us?

        update@october29th2024: so in khatri code when he is giving the word
        to the embedding model he is using plain text word i.e "aldea"
        but when he is storing it in the dict train_vocab_embeddings, the key
        goes back to "aldea__s" while the value is the vector/embedding you got from
        fasttext
        # """
        train_vocab_embeddings = get_vocab_emb_dict(train_vocab)
        val_vocab_embeddings = get_vocab_emb_dict(val_vocab)
    
        

    else:
        #for the words created from other ansatz just write it as : _0_ so we can reuse the parsing from original khatri code. But here recording specifically for this instance
        train_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in train_vocab}
        val_vocab_embeddings = {wrd: embedding_model[wrd.split('__')[0]] for wrd in val_vocab}


    #to store all the initial weights
    initial_param_vector = []

    #todo: find what qnlp_model.symbols is- rather how is it different than train vocab?
    #ans: it is every word in the given list of circuits e.g. únicamente_0__s
    for sym in qnlp_model.symbols:
        #@sep2nd2024-not sure what idx is supposed to do, am giong to give it the number associated with the word
        if(ansatz_to_use==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            """#@sep2nd2024/ end of day: getting key error for lots of words - e.g. aldea..but why are words 
            # in qnlpmodel.symbols not getting the fasttext emb on the fly? why are we separating train_embeddings earlier?        
            #what is the meaning of symbols in qnlp.model
            #todo a) read the lambeq documentation on symbols
            #  b) read the 2010 discocat and CQM paper onwards up, chronologically
            #no point turning knobs without deeply understanding what symbols do
            
            #todo:compare the format ith mrpc data, and see if he is storing the initial param vector- and symbols with _0_ or not?
            # update@29th 0ct 2024. a) done
            #  b) done. Refer definition of symbol in comments near line 223
            # b) yes he is appending just the first element of the embedding as an entry to 
            # initial params."""
            if cleaned_wrd_with_type in train_vocab_embeddings:
                if model_to_use == PytorchModel:
                    # initial_param_vector.append(train_vocab_embeddings[cleaned_wrd_with_type][int(idx),1-int(idx)])
                    val1= train_vocab_embeddings[cleaned_wrd_with_type][int(idx)]
                    val2= train_vocab_embeddings[cleaned_wrd_with_type][int(idx)+1]
                    tup= torch.tensor ([val1,val2], requires_grad=True) #initializing with first two values of the embedding
                    initial_param_vector.append(tup)
                else:
                    initial_param_vector.append(train_vocab_embeddings[cleaned_wrd_with_type][int(idx)])
            else:
                '''
                #todo: lots of words are getting hit with OOV- conirm why they are not there in fasttext emb
                # my guess is its all the unicode characters. 
                # In theory fast text is meant to create zero 
                # OOV..since it builds up from 1 gram 2 gram etc
                #update: this might be caused because am 
                # removing the _0_ thing from the actual name, without
                #  realizing what it is doing.
                
                found that this word verdad, was OOV/not in fasttext emb
                found that this word vió, was OOV/not in fasttext emb
                found that this word yo, was OOV/not in fasttext emb
                found that this word yyyyyy was OOV/not in fasttext emb
                '''
                print(f"found that this word {cleaned_wrd_with_type} was OOV/not in fasttext emb")

    """
    Set the intialization of QNLP model's weight as that of the embeddings of each word
    Am not completely convinced about what he is doing here.
    FOr example in NN world, embedding is separate than weights of neurons.
    You might initialize the weights of neurons with random shit like Xavier glorot
    but almost nver initialize it with embedding itself.
    todo: qnlp_model.weights-is a list
     of size 463 while the np.array is creating a matrix of shape (463,) -which
     is then again being converted to nn.ParameterList-which is a list.
     We dont even know how/why we are going to numpy land and then converting it back to nn.parameter
     rather; there was no np.array initially.- gave some error. So i went to khatri's
     code - and exactly copy pasted np.array()- the error went away, so i didnt complain.
     But all this is being caused because we are working on someone else's code without
     realizing what the code does.
    """

    """#update @16th oct 2024. 
    this below code was what the error: the shape[2] not equal to shape[1]
    this meant,
     
    a) the LHS= qnlp_model.weights is a list of tensors i.e [Tensor([0.9,0.1],req_grad="true"] 
    but the RHS is giving a simple list i.e initial_param_vectorDont=[1]
     In short the error is saying the first value is meant to be a tensor over a tuple o
      of two elements, instead you are giving me a plain as 1 float value.
       
    b) I still dont know why he is initializing the weights of QNLP model
    (which looks like the weights of two classes,..while we are providing
     just the vector of embeddings- what is the relationship between
      a vector of embeddings and qnlp.weights? find out from KHatri's initial code ) 

      @octr17th2024
      commenting the code out until i find what its doing
    ."""
    # assert len(qnlp_model.symbols) == len(qnlp_model.weights)==  len(train_vocab_embeddings) +len(val_vocab_embeddings)
    assert len( qnlp_model.weights) == len(initial_param_vector)
    qnlp_model.weights = nn.ParameterList(initial_param_vector)
    

    return train_vocab_embeddings, val_vocab_embeddings, max_word_param_length

def trained_params_from_model(trained_qnlp_model, train_embeddings, max_word_param_length):

    """

     Args:
    trained_qnlp_model- the trained_qnlp_model
    train_vocab_embeddings- the initial embeddings for words in the vocab got from fasttext
    max_word_param_length- what is the maximum size of a wor

    Returns:
        a map between each word and its latest final weights
    """

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

def generate_OOV_parameterising_model(trained_qnlp_model, train_vocab_embeddings, max_word_param_length):
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


    """explanation of dict_training_symbols_vs_qnlp_trained_weights:
    dict_training_symbols_vs_qnlp_trained_weights is a dictionary that map symbols in the trained QNLP model to 
    its weights at the end of QNLP training i.e the training that happened to model 1 i.e the QNLP model
    
    #todo: print and confirm if symbol means word
    #update@sep 11th 2024 : symbol is not word, it is word+ that idx number- 
    # which i am suspecting is the number of Types a same word can have
    # for example if you read 1958 Lambek paper youc an see that adverb Likes can have two different TYPE representations.
    # but yes, todo: confirm this by comparing it with the original Khatri MRPC code.
    # todo: he is doing the same process above inside the function generate_initial_parameterisation
    around line 222. Find out why doing it so many times? are they same? so why not just pass around?- update@oct29th2024. the _2 in symbol denotes the qbit number or the dimension of the tensor. refer line 222 comments starting with update @29thOct2024
    # """

    
    dict_training_symbols_vs_qnlp_trained_weights = {symbol: param for symbol, param in
                                                      zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
   
   
    """
    train_vocab_embeddings are the initial embeddings 
    for words in the vocab we got from fasttext-     
    now for each such word create an array of zeroes called trained_param_vectors
    - this array is where the weights of the model 3, NN model which maps between embedding and learned weights of 
    QNLP model will be added.
    i.e for now for each such word in training vocabulary, create a vector filled with zeroes to represent
    trained parameters         
    
    Update@oct29th2024- in the below code it was np.zeros(max_word_param_length+1)
    however, when using pytorch trainer and pytorch model it needs weights in a tuple form 
    (todo find out why)- but max param length was 1- since we were using spider ansatz
    so as of now hardcoding the np.zeroes to be 2- will have to change this later as and when we move to other ansatz

    """
    dict_wrd_in_training_vs_weights = {wrd: np.zeros(max_word_param_length+1) for wrd in train_vocab_embeddings}
    
    '''for each such word in training vocabulary, 
      to the empty array/array of zeroes   created above.
      -get ist weights from the qnlp trained weights
    #todo 1) am still not sure why he is touching his nose in a circumambulated way
    # 2)  print and confirm, i think they are repeating the same
    #  weight value for every entry of the array in trained_param_vectors.
    # 
    # Note that this for loop below is purely done so as to extract word out of symbol.
    # the dictionary intuition remains same between both dictionaries. i.e dict_training_symbols_vs_qnlp_trained_weights and 
    # dict_wrd_in_training_vs_weights i.e key is either word/symbol while value is sambe for both, which is the corresponding 
    # weight of the same word in the trained QNLP model '''
    cleaned_wrd_with_type=""
    for symbol, trained_weights in dict_training_symbols_vs_qnlp_trained_weights.items():
        if(ansatz_to_use==SpiderAnsatz):  
            #symbol and word are different. e.g. aldea_0. From this extract the word
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symbol.name)
            rest = symbol.name.split('_', 1)[1]
            idx = rest.split('__')[0]       
        else:
            cleaned_wrd_with_type, idx = symbol.name.rsplit('_', 1)
    
        assert cleaned_wrd_with_type  != "" 
        #if that word is in train vocab (from the embedding side)- 99% should be there.
        #todo: confirm if any words are outside
        #stopping here at oct 16th 2024. 2pm- getting a newerror some key value mistake.
        # i think this has something to do with the wrd splitting thing
        #todo: in original code of khatri he is not storing aldea with plain english word
        #but he is doing with aldea_0_s- make sure we revert back to this later.
        if cleaned_wrd_with_type in dict_wrd_in_training_vs_weights:
                dict_wrd_in_training_vs_weights[cleaned_wrd_with_type][int(idx)] = trained_weights[int(idx)]
        else:                
                print(f"inside OOV_generation-found that this word {cleaned_wrd_with_type} was not in trained_param_vectors")

                

    wrds_in_order = list(train_vocab_embeddings.keys())

    """#For each word in a ordered list of training vocabulary words, create 2 arrays. 
    One for embeddings (NN_train_X) and the other for trained weights (NN_train_Y)
    Note, the goal of model 3 is to learn the mapping between these two things
    
    update@oct29th2024. currently we are passing only one value as label. but that is only because
    aldea_0 the max value is 0 (param length 1). and that happened only because we are using
    dumb spider ansatz. Eventually when we use any other ansatz, icnluding tensor ansatz
    every word should have more than 1 values. So NN_train_Y will be a list of 2 tuple arrays. Be ready
     for if and when this bombs """
    NN_train_X = np.array([train_vocab_embeddings[wrd] for wrd in wrds_in_order])
    NN_train_Y = np.array([dict_wrd_in_training_vs_weights[wrd] for wrd in wrds_in_order])
    
    """#this is model 3. i.e create a simple Keras NN model, which will learn the above mapping.
      todo: use a better model a) FFNN using pytorch b) something a little bit more complicated than a simple FFNN"""
    OOV_NN_model = keras.Sequential([
      layers.Dense(int((max_word_param_length + MAXPARAMS) / 2), activation='tanh'),
      layers.Dense(max_word_param_length, activation='tanh'),
    ])

    #standard keras stuff, initialize and tell what loss function and which optimizer will you be using
    OOV_NN_model.compile(loss='mean_absolute_error', optimizer=keras.optimizers.Adam(0.001))

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.01,
        patience=10,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )
    # Embedding dim!
    """#todo find why maxparams are hardcoded as 300 (is it the dimension of the fasttext embedding?)
    #ans: yes. The first layer needs to have the dimension of NN_train_X, which in turn has the dimension of the  
    Fasttext embedding"""
    OOV_NN_model.build(input_shape=(None, MAXPARAMS))

    #train that model 3
    hist = OOV_NN_model.fit(NN_train_X, NN_train_Y, validation_split=0.2, verbose=1, epochs=100,callbacks=[callback])
    print(hist.history.keys())
    print(f'OOV NN model final epoch loss: {(hist.history["loss"][-1], hist.history["val_loss"][-1])}')
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    return OOV_NN_model,dict_wrd_in_training_vs_weights

def evaluate_val_set(pred_model, val_circuits, val_labels, trained_weights, val_vocab_embeddings, max_word_param_length, OOV_strategy='random', OOV_model=None):
    """
    So this is where we do the final testing (even if its over dev)
    Here they are sending 
    a) pred_model:the newly created prediction model - which is same as the original QNLP model fundamentally.
    b) val_circuits , val_labels- what it says
    c) trained_params: trained_wts from the model 1- qnlp model (my guess is he is going to assign these weights to the 
    newly created model. Though why he is doing it in a circumambulated way while he could have directly 
    used qnlp_model i.e model 1 is beyond me)- update@oct30th2024. This is all zeroes. find and fix why
    d) val_vocab_embeddings: Take every word in test/val set, give it to fasttext, and get their embeddings
    e) max_word_param_length: refer function get_max_word_param_length
    f) oov_strategy= picking one hardcoded one from [zero,embed, model,random etc]- basically this is where you 
    decide what baseline model do you want to use for your model 3- rather, do you want to use any model at all
    or do you want to use things like baselines methods like random number generator, fill with zeroes etc.

    """


    pred_parameter_map = {}
    #Use the words from train wherever possible, else use DNN prediction
    for wrd, embedding in val_vocab_embeddings.items():
        if OOV_strategy == 'model':
            """
            for each word in test/dev vocabulary, give the word as input to model 3 (the NN model also called DNN here)
            which now will ideally return the corresponding weight of the QNLP trained model- because model 3 had learned
            the mapping between these two things. REfer: generate_oov_* function above.
            Note that all other strategies below are time pass/ boring hard coded baseline stuff
            """
            pred_parameter_map[wrd] = trained_weights.get(wrd, OOV_model.predict(np.array([embedding]), verbose=0)[0])
        elif OOV_strategy == 'embed':
            pred_parameter_map[wrd] = trained_weights.get(wrd, embedding)
        elif OOV_strategy == 'zeros':
            pred_parameter_map[wrd] = trained_weights.get(wrd, np.zeros(max_word_param_length))
        else:
            pred_parameter_map[wrd] = trained_weights.get(wrd, 2 * np.random.rand(max_word_param_length)-1)

    
    #convert the dictionary pred_parameter_map into a list pred_weight_vector
    pred_weight_vector = []

    for sym in pred_model.symbols:
        if(ansatz_to_use==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            if cleaned_wrd_with_type in pred_parameter_map:
                if model_to_use == PytorchModel:
                    val1= np.float32(pred_parameter_map[cleaned_wrd_with_type][int(idx)])
                    """# todo: there are some cleaned_wrd_with_type in pred_parameter_map which is empty. i.e size of tuple =1
                    #figure that out,. commenting this line until then and forcing val2 to be zero                
                    # val2= pred_parameter_map[cleaned_wrd_with_type][int(idx)+1]"""

                    if len(pred_parameter_map[cleaned_wrd_with_type])>1:
                        val2= np.float32(pred_parameter_map[cleaned_wrd_with_type][int(idx)+1])
                    else:
                        val2 = np.float32(0.0)
                    tup= torch.tensor ([val1,val2], requires_grad=True) #initializing with first two values of the embedding            
                    pred_weight_vector.append(tup)

    """#so he is assigning the weights he picked from model 3's outout (the DNN one)
    to that of model 4 - i.e the prediction model. I think this is the answer to the question
    of why is he using two QNLP models (model1 and model 4)- that is because if we could direclty use
    the trained weights of the original QNLP model model 1- we wouldnt have had to go through this 
    circumambulated way of finding the mapping between fast text and qnlp weights.
    The weights here are the new learnings created by model 3- the DNN/NN model specifically trained on
    model(a,b) where a is the weights from model1 and b is the embeddings from model2 -the fasttextmodel

    todo: a) why is the pred_model.weights zero
    b) why is it not a parameter list, like that of model 1?
    """

    assert len(pred_model.symbols) == len(pred_weight_vector)
    assert type(pred_model.weights) == type( nn.ParameterList(pred_weight_vector))
    # assert len(pred_model.weights) == len(pred_weight_vector)    
    pred_model.weights = nn.ParameterList(pred_weight_vector)

    
    #use the model now to create predictions on the test set.
    preds = pred_model.get_diagram_output(val_circuits)
    loss_pyTorch =torch.nn.BCEWithLogitsLoss()
    l= loss_pyTorch(preds, torch.tensor(val_labels))
    a=accuracy(preds, torch.tensor(val_labels))

    return l, a

def read_data(filename):
    labels, sentences = [], []
    with open(filename) as f:
        for line in f:
            t = float(line[0])
            if model_to_use == PytorchModel:
                labels.append([t, 1-t])
            else:
                labels.append(int(t))
            sentences.append(line[1:].strip())
    return labels, sentences

#back to the main thread after all functions are defined.

#read the base data, i.e plain text english.
train_labels, train_data = read_data(os.path.join(DATA_BASE_FOLDER,TRAIN))
val_labels, val_data = read_data(os.path.join(DATA_BASE_FOLDER,DEV))
test_labels, test_data = read_data(os.path.join(DATA_BASE_FOLDER,TEST))



if TESTING:
    train_labels, train_data = train_labels[:2], train_data[:2]
    val_labels, val_data = val_labels[:2], val_data[:2]
    test_labels, test_data = test_labels[:2], test_data[:2]
    EPOCHS = 1


"""
# not using bob cat parser- note: this wasn't compatible with spider ansatz

we are using spiders reader for this code, because pytorch trainer goes well
#with it. Note that this is being done in SEp 2024 to just get the
#  code to take off from the ground. However, other than this being a good baseline, 
spider reader should be soon discareded and switched to bob cat parser+ some
quantum traineres.



# """


"""spanish_diagrams is a dummy function I had created once when I had to test
both MRPC and uspantekan/spanish data at the same time. it is very useful in debugging
especially for converting aldea_0 kinda format issues. But do remove this
once not needed. Mithun@26th sep 2024"""
def spanish_diagrams(list_sents,labels):
    list_target = []
    labels_target = []
    for sent, label in tqdm(zip(list_sents, labels),desc="reading sent"):
        
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
        spiders_diagram = parser_to_use.sentence2diagram(sent)

        """was getting error, [2] is not a valid shape for
         ... this was marked as a soluion on the internet for """
        # from  lambeq import UnifyCodomainRewriter, AtomicType
        # rewriter = UnifyCodomainRewriter(AtomicType.SENTENCE)  #  Codomain should be always S
        # new_diagrams = [rewriter(d) for d in spiders_diagram]

        list_target.append(spiders_diagram)
        labels_target.append(label)
    print("no. of items processed= ", len(list_target))
    return list_target, labels_target

"""#convert the plain text input to ZX diagrams
# #todo: find who does the adding the underscore 0 part. is it ansatz or sentence2diagram?.
# Ans:value of _0 the 0 part, is extracted in spanish_diagrams. But the attaching it to 
# the nounc part aldea_0 is done by ansatz.
# Note that this is a confusion arising on sep 29th 2024: because we don't know what is the meaning of
# the _0 in aldea. Rather, i am yet to read the 2010 discocat paper. That should explain it
# Until then taking a guess"""
train_diagrams, train_labels_v2 = spanish_diagrams(train_data,train_labels)
val_diagrams, val_labels_v2 = spanish_diagrams(val_data,val_labels)
test_diagrams, test_labels_v2 = spanish_diagrams(test_data,test_labels)

train_labels = train_labels_v2 #doing because didnt want same variable going into th function and returning it. python lets you get away with it, but i dont trust it
val_labels = val_labels_v2
test_labels = test_labels_v2

"""
these are now orphan codes, but in reality, these code were there in khatri's original
code (https://colab.research.google.com/drive/13W_oktxSFMAB6m5Rfvy8vidxuQDrCWwW#scrollTo=0be9c058)
Mithun@27thsep2024-I have a bad feeling I might have removed all this an year ago, when it was giving "error"
Clearly my mental state at that time ws so messed pu that all i was trying to do it, somehow get it to work.
even if it means removing bug filled code...weird/sad but true.
-- 
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

"""


"""
print and assert statements for debugging
"""
assert len(train_diagrams)== len(train_labels_v2)
print(f"count of train, test, val elements respectively are: ")
print({len(train_diagrams)}, {len(test_diagrams)}, {len(val_diagrams)})
assert len(train_diagrams)== len(train_labels)
assert len(val_diagrams)== len(val_labels)
assert len(test_diagrams)== len(test_labels)

def run_experiment(nlayers=1, seed=SEED):

    """mithuns comment @26thsep2024typically spider ansatz only goes with spider reader. 
    like i mentioned earlier, spider was used to just get the code off the ground
    1) we need to definitely test with atleast bobcat parser
    2) Noun should have a higher dimension than sentence? how? 
    - go back and confirm the original 1958 paper by lambek. also how
    is the code in LAMBEQ deciding the dimensions or even what  data types to use?
    answer might be in 2010 discocat paper"""
    ansatz = ansatz_to_use({AtomicType.NOUN: Dim(4),
                        AtomicType.SENTENCE: Dim(2)
                        #    AtomicType.PREPOSITIONAL_PHRASE: Dim(2),
                        })
    
    """
    todo: his original code for ansatz is as below. Todo find out: why we switched to the above.
    I think it had something do with spider ansatz-
    todo: write the whole history of playing with this code- why spider ansatz etc- in one single
    word document, like chronological order.-for your own sanity

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE
print(f'RUNNING WITH {nlayers} layers')
    ansatz = Sim15Ansatz({N: 1, S: 1, P:1}, n_layers=nlayers, n_single_qubit_params=3)

    Also the two lines below is more to do with comparing two things, like NLI/MRPC, Might not be that
    relevant in say classification

    train_circs = [ansatz(d) >> equality_comparator for d in train_X]
    test_circs = [ansatz(d) >> equality_comparator for d in test_X]
    """

    #use the anstax to create circuits from diagrams
    train_circuits =  [ansatz(diagram) for diagram in train_diagrams]
    val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
    test_circuits = [ansatz(diagram) for diagram in test_diagrams]


    

    #mithuns comment @26thsep2024:pytorch model was the only one going well
    #  with spider reader and spider anstaz
    #also todo: should this not contain both val and train circuits as inputz?
    # qnlp_model = PytorchModel.from_diagrams(train_circuits)

    from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset
    # qnlp_model = NumpyModel.from_diagrams(train_circuits)
    qnlp_model = model_to_use.from_diagrams(train_circuits)
    

    
    print("length of each circuit in train is:")
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

    """
    #this is a;;wandb crap,- i was trying to automate a sweep across parameters. 
    Turned out to be more  hurting than helping. Will think about this as and when we get to expt running stage
    # comment out if becoming a pain in the butt. MLFlow is better than wandb imho
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
    # wandb_logger = WandbLogger()"""

    trainer = trainer_to_use(
            model=qnlp_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=LEARNING_RATE,
            use_tensorboard=True,
            epochs=EPOCHS,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=SEED)
    
    # trainer = QuantumTrainer(
    #     qnlp_model,        
    #     loss_function=torch.nn.BCEWithLogitsLoss(),
    #     epochs=EPOCHS,
    #     optimizer=SPSAOptimizer,
    #     optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
    #     evaluate_functions=eval_metrics,
    #     evaluate_on_train=True,
    #     verbose = 'text',
    #     seed=seed
    # )

    #get the embeddings etc to be used in models 2 through 4. Note
    # that one very interesting thing that happens as far as model 1is considered is that
    # inside the function generate_initial_parameterisation() the QNLP
    # model ka weights (i.e the angles of gates)
    # gets initialized with initial fast text embeddings of each word in training

    train_embeddings, val_embeddings, max_w_param_length = generate_initial_parameterisation(
        train_circuits, val_circuits, embedding_model, qnlp_model)

    #run ONLY the QNLP model.i.e let it train on the train_dataset. and test on val_dataset
    #todo:  somewhere you use the term val and somehwere else you use the term test.
    #  Fix it/use only one everywhere-bottomline: make val/dev explicitly different than test

    
    
    trainer.fit(train_dataset, log_interval=1)

    """#for experiments on october 14th 2024. i.e 
    just use 1 off the shelf model and spread spectrum/parameter search
      for out of hte box for usp"""
    

    """
    Uncomment this code eventually. Here he is using the model trained above on training circuits 
    itself, just to get training accuracy
    todo: do we want to bring in dev here now, and keep testing below as TEST partition? 
    else what is the point in doing prediction on training data- obviously it will  be 100%
    -on the bright side_ maybe it is a sanity check to ensure our model is getting trained atleast"""

    """error@ 16th 2024- 
    below code is egiving the error,  i.e when trying to evaluate on val dataset

     File "/Users/mithun/miniconda3/envs/qnlp/lib/python3.12/site-packages/lambeq/training/pytorch_model.py", line 142, in get_diagram_output
    raise KeyError(
KeyError: 'Unknown symbol: únicamente_0__s

    Note this is only 
    a) after commenting the qnlp_model.weights ka problem
    inside the function generate_initial_parameterisation()
    which means, we kind of ignore/commented out that problem, 
    and here is the next one. 

    b) when we are trying to evaluate on val dataset


    What this error means is that: únicamente_0__s is not found 
    to get the diagram output for. Mostlikely it is looking
    for embedding of únicamente_0__s inside train_embeddings.

    "the exact line 143 can be found in this part of the LAMBEQ code:https://github.com/CQCL/lambeq/blob/8109d952d707880b8588e0d04f24f0b5a94c3d59/lambeq/training/pytorch_model.py#L140

    what that line is doing is that, it is going through circuits of val data,
    extracting each of the root word from it, and 
    asking for the weights of that word from a dictionary called
      parameters = {k: v for k, v in zip(self.symbols, self.weights)}
      which has all symbols as keys and its weights as values
      i.e self.symbols and self.weights are zipped, and combined as a dictionary
      
      todo: 
      1. find if our self.symbols and self.weights have same size.
      update; answer =
      len(qnlp_model.symbols)
463
len(qnlp_model.weights)
463
(qnlp_model.symbols[0])
(aldea_0__s
qnlp_model.weights[0]
Parameter containing:
tensor([-0.0098,  0.7008], requires_grad=True)

      2. it is asking for únicamente_0__s- which shouldnt happen because 
      our dictionaries are all stored as {plain text name: embedding} i.e
       just "únicamente".
       3. unicamente is from val data, why are we comparing/asking for its embedding from 
       model.symbols- which has only training data?

       update: finally figured out what is going on:
       WE ARE NOT SUPPOSED TO USE THE BASIC QNLP MODEL ON VAL dataset
       because
        a) qnlp_model is by definition handicapped. i.e it doesn't know
         what to do with OOV words.SO using it for prediction on val dataset itself
         is wrong methodology because Qnlp_model doesn't know how to handle OOV
          b)  the whole purpose of creating all the models 2, 3,4 and then 
       the OOV_prediction model is those 4 models below, which KNOW HOW TO 
       HANDLE OOV
       I.E the final accuracy calculation on val/dev SHOULD ONLY BE DONE
       USING THESE 4 models and NEVER using the base qnlp_model.


    

    """
    # val_preds = qnlp_model.get_diagram_output(val_circuits)
    # loss_function = torch.nn.BCEWithLogitsLoss()
    # val_labels_pytorch_tensor= torch.Tensor(val_labels)
    # val_loss = loss_function(val_preds, val_labels_pytorch_tensor)
    # val_acc = accuracy(val_preds, val_labels_pytorch_tensor)
    # print(f'Val STATS: {val_loss, val_acc}')
    
    # import sys
    # sys.exit(1)

    train_preds = qnlp_model.get_diagram_output(train_circuits)
    # loss_pyTorch = nn.CrossEntropyLoss()
    loss_pyTorch =torch.nn.BCEWithLogitsLoss()
    train_loss= loss_pyTorch(train_preds, torch.tensor(train_labels))
    train_acc =accuracy(train_preds, torch.tensor(train_labels))
    print(f"value of train_loss={train_loss} and value of train_acc ={train_acc}")




    """
    

  

    

    # train_loss = loss(train_preds, train_labels)
    # train_acc = acc(train_preds, train_labels)
    # print(f'TRAIN STATS: {train_loss, train_acc}')"""


    """by here the actual QNLP model is trained. Next is we are going to connect the model which learns
    #  relationsihp between fasttext emb and angles. look inside the function
    generate_initial_parameterisation for more specific comments

    
    #now we get into the land of model 3- i.e the model which learns the connection between weights of model 1 
    i.e the QNLP model. 
    and embeddings of model 2
    i.e model 3=model(a,b) where a = embeddings and b = angles
    todo: find the difference between angles nad weights- why is it per word? """

    """"#model 3- i.e nn-model is the model which learns mapping between fasttext ka embeddings and QNlp model (i.e model 1) ka learned weights
    or in other words , if you give the embedding of a new sentence model 3 (NN_model) will
    give you the corresponding weights of model 1 (i.e QNLP model) which can then be used for
    predicting class of the new sentence"""
    NN_model,trained_wts = generate_OOV_parameterising_model(qnlp_model, train_embeddings, max_w_param_length)

    """#this is model 4, a separate model1 like model used only for predicting 
    on the test/dev data based only on test_circuits. Now you might have a qn why 
    cant we simply use model 1 itself for predicting the class of val/dev 
    data. This is the OOV problem. remember, Model 1 the way it predicts is
    if you give the circuits, it will multiply by its weights and tell you
    the exact class the circuit belongs to. however, remember, it has no idea how 
    to convert an OOV word to circuit. Which is why we went back to fasttext
    embeddings. But then we realized, just having fast tet embeddings of thew new
    dev/val data is not alone enough, but we want to instead get the corresponding
    weights of it from model 1. That is why we trained another NN model called
    model3 (aka NN_model) which gives you the corresponding weights of the new sentences.
    Now coming back to your question of why not use model 1. as of now model 1 has learned
    weights for only KNOWN words,. If you give the weights of UNKNOWN words , like those
    found in dev, it is WRONG. So the best way is, create a new QNLP model (exactly like
    model 1) called model4, assign its weights to be what model 3 (the OOV_NN) model predicted
    and now give the corresponding circuits of dev data (just like model 1 would have done
    if it could) and ask it to multiply circuitsx weights and give a class prediction.
    """
    
    #note/todo: differentiate difference between test and val. If it is val here, then make sure we are not passing val inside model1.fit()--done
    prediction_model = model_to_use.from_diagrams(val_circuits)

    trainer = trainer_to_use(
            model=prediction_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=SEED)

    """#Create a dictionary where key is word and weights are the trained weights after QNLP_model was
    trained. 
    Todo: What I dont understand is, why is he doing that agian here? this is exactly what he is doing
    in the first few lines of  generate_OOV_parameterising_model. 
    a) check if this is my mistake or its same in his original code
    b) debug and check if values are same. i.e in generate_oov_* and here. if yes, return from generate_oov_* 
    """
    # trained_wts = trained_params_from_model(qnlp_model, train_embeddings, max_w_param_length)


    """ now that we have 2 models ready, the OOV model, i.e model 3 (of 
    which there are 3 different types as shown below) and the
    original QNLP model 1, both well trained, let's evaluate it on the dev (and
    eventually test) set
    """


    smart_loss, smart_acc = evaluate_val_set(prediction_model,
                                                val_circuits,
                                                val_labels,
                                                trained_wts,
                                                val_embeddings,
                                                max_w_param_length,
                                                OOV_strategy='model',
                                                OOV_model=NN_model)
    print(f"value of smart_loss={smart_loss} and value of smart_acc ={smart_acc}")
    print('Evaluating EMBED model')
    #This is where he is directly using the embeddings of fasttext model as the weigh of prediction model- absolute crap/used only for creating random baseline
    embed_loss, embed_acc = evaluate_val_set(prediction_model,
                                              val_circuits,
                                                val_labels,
                                              trained_wts,
                                              val_embeddings,
                                              max_w_param_length,
                                              OOV_strategy='embed')
    
    print('Evaluating ZEROS model')
    zero_loss, zero_acc = evaluate_val_set(prediction_model,
                                              val_circuits,
                                                val_labels,
                                              trained_wts,
                                              val_embeddings,
                                              max_w_param_length,
                                              OOV_strategy='zeros')

    rand_losses = []
    rand_accs = []

   
    print('Evaluating RAND MODEL')
    """
    even though random model is yet another random number generator, he is using it 
    across multiple times, am guesssing to make sure its normalized, accuracy avg or whatever
    crap its called. i.e he is trying to make it as stupid/random as possible.

    update@oct30th2024. Random model is giving some semaphore shit. commenting out for now
    """
    # for _ in range(10):
    #     rl, ra  = evaluate_val_set(prediction_model,
    #                                           val_circuits,
    #                                             val_labels,
    #                                           trained_wts,
    #                                           val_embeddings,
    #                                           max_w_param_length,
    #                                           OOV_strategy='random')
        

    #     rand_losses.append(rl)
    #     rand_accs.append(ra)
    
    """#so by now, we have predictions on the test/dev/val set, using 4 different models. 1 is our trained NN model called 
    'model' and the rest are all baselines.
    He collects all the results (loss and accuracy of all 5 experiments)
    1. training using QNLP/model 1 and am guessing he is evaluating on a part of trained model itself
    2. using NN (aka model 2 lines above): which is our model 3 trained to learn mapping between fasttext embeddings and qnlp-model-1s weights 
    #rest are all the baselines models
    """
    res =  {'TRAIN': (train_loss, train_acc),
            'NN': (smart_loss, smart_acc),
            'EMBED': (embed_loss, embed_acc),
            'ZERO': (zero_loss, zero_acc)
           }
    print(f'for the seed={SEED} the accuracy given by the model ZERO: {res["ZERO"][1]}')
    print(f'for the seed={SEED} the accuracy given by the model EMBED: {res["EMBED"][1]}')
    print(f'for the seed={SEED} the accuracy given by the model NN: {res["NN"][1]}')

    return res



"""####final push-which calls run_experiment function above
todo: why is he setting random seed, that tooin tensor flow
- especially since am using a pytorch model.
"""
import tensorflow as tf
compr_results = {}

tf_seeds = [2]

for tf_seed in tf_seeds:
    tf.random.set_seed(tf_seed)
    this_seed_results = []
    for nl in [3]:
        this_seed_results.append(run_experiment(nl, tf_seed))
    compr_results[tf_seed] = this_seed_results

print(f"value of all evaluation metrics across all seeds is {compr_results}")
