# -*- coding: utf-8 -*-
"""v7
#name: v7_*
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
4. Prediction model - which is use dto predict on test set.

"""

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
from lambeq import TensorAnsatz,SpiderAnsatz,Sim15Ansatz, IQPAnsatz,Sim14Ansatz
from lambeq import BobcatParser,spiders_reader
from lambeq import TketModel, NumpyModel, QuantumTrainer, SPSAOptimizer, Dataset, TreeReader
import wget
import wandb
from pytket.extensions.qiskit import AerBackend
from lambeq import BinaryCrossEntropyLoss
import numpy as np
import keras_tuner
import keras
from keras import layers


TYPE_OF_DATA_TO_USE = "food_it" #["uspantek","spanish","food_it","msr_paraphrase_corpus"]
parser_to_use = BobcatParser    #[tree_reader,bobCatParser, spiders_reader,depCCGParser]
ansatz_to_use = SpiderAnsatz    #[IQPAnsatz,SpiderAnsatz,Sim14Ansatz, Sim15Ansatz,TensorAnsatz ]
model_to_use  = PytorchModel   #[numpy, pytorch,TketModel]
trainer_to_use= PytorchTrainer #[PytorchTrainer, QuantumTrainer]
embedding_model_to_use = "english" #[english, spanish]
MAX_PARAM_LENGTH=0


if(parser_to_use==BobcatParser):
    parser_to_use_obj=BobcatParser(verbose='text')


if(embedding_model_to_use=="spanish"):
    # get_ipython().system('wget -c https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1 -O ./embeddings-l-model.bin')
    embedding_model = ft.load_model('./embeddings-l-model.bin')
if(embedding_model_to_use=="english"):
    import os.path
    if not (os.path.isfile('cc.en.300.bin')):
        filename = wget.download(" https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
    embedding_model = ft.load_model('cc.en.300.bin')


arch = f"{ansatz_to_use}+{parser_to_use_obj}+{trainer_to_use}+{model_to_use}+{embedding_model_to_use}"


# maxparams is the maximum qbits (or dimensions of the tensor, as your case be)
BASE_DIMENSION_FOR_NOUN =2 
BASE_DIMENSION_FOR_SENT =2 
BASE_DIMENSION_FOR_PREP_PHRASE= 2
MAXPARAMS = 300
BATCH_SIZE = 30
EPOCHS_TRAIN_MODEL1 = 100
EPOCHS_MODEL3_OOV_MODEL = 100
LEARNING_RATE = 3e-2
SEED = 0
DATA_BASE_FOLDER= "data"


#setting a flag for TESTING so that it is done only once.
#  Everything else is done on train and dev
TESTING = False



if(TYPE_OF_DATA_TO_USE== "uspantek"):
    TRAIN="uspantek_train.txt"
    DEV="uspantek_dev.txt"
    TEST="uspantek_test.txt"
    

if(TYPE_OF_DATA_TO_USE== "spanish"):
    TRAIN="spanish_train.txt"
    DEV="spanish_dev.txt"
    TEST="spanish_test.txt"
    

if(TYPE_OF_DATA_TO_USE== "msr_paraphrase_corpus"):
    TRAIN="msr_paraphrase_train.txt"
    DEV="msr_paraphrase_test.txt"
    TEST="msr_paraphrase_test.txt"
    type_of_data = "pair"

if(TYPE_OF_DATA_TO_USE== "food_it"):
    TRAIN="mc_train_data.txt"
    DEV="mc_dev_data.txt"
    TEST="mc_test_data.txt"
    


wandb.init(    
    project="qnlp_nov2024_expts",    
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": arch,
    "BASE_DIMENSION_FOR_NOUN".lower(): BASE_DIMENSION_FOR_NOUN ,
    "BASE_DIMENSION_FOR_SENT".lower():BASE_DIMENSION_FOR_SENT ,
    "MAXPARAMS".lower() :MAXPARAMS,
    "BATCH_SIZE".lower():BATCH_SIZE,
    "EPOCHS".lower() :EPOCHS_TRAIN_MODEL1,
    "LEARNING_RATE".lower() : LEARNING_RATE,
    "SEED".lower() : SEED ,
    "DATA_BASE_FOLDER".lower():DATA_BASE_FOLDER,
    "EPOCHS_DEV".lower():EPOCHS_MODEL3_OOV_MODEL,
    "TYPE_OF_DATA_TO_USE".lower():TYPE_OF_DATA_TO_USE,
    "embedding_model_to_use".lower():embedding_model_to_use
    })


sig = torch.sigmoid

def accuracy(y_hat, y):
        assert type(y_hat)== type(y)
        # half due to double-counting
        #todo/confirm what does he mean by double counting
        return torch.sum(torch.eq(torch.round(sig(y_hat)), y))/len(y)/2  

eval_metrics = {"acc": accuracy}
spacy_tokeniser = SpacyTokeniser()

if TYPE_OF_DATA_TO_USE in ["uspantek","spanish"]:
    spanish_tokeniser=spacy.load("es_core_news_sm")
    spacy_tokeniser.tokeniser = spanish_tokeniser
else:
    english_tokenizer = spacy.load("en_core_web_sm")
    spacy_tokeniser.tokeniser =english_tokenizer


import os


"""go through all the circuits in training data, 
    and pick the one which has highest type value
    i.e the symbol which has the highest vector dimension (if using a classical anstaz)
      or number of 
    qbits(if using quantum ansatz)
    """
def get_max_word_param_length(input_circuits):
        lengths=[]
        for d in input_circuits:
            for symb in d.free_symbols:
                x =  symb.name.split('_', 1)[1]
                y = x.split('__')[0]
                lengths.append(int(y))
        return lengths


            
"""
given a word from training and dev vocab, get the corresponding embedding using fast text. 

:param vocab: vocabulary
:return: returns a dictionary of each word and its corresponding embedding
""" 
def get_vocab_emb_dict(vocab):            
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
def create_vocab_from_circuits(circuits):
    vocab=set()
    if(ansatz_to_use==SpiderAnsatz):  
        for d in circuits:
            for symb in d.free_symbols:                  
                cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(symb.name)                
                vocab.add(cleaned_wrd_with_type)
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
def generate_initial_parameterisation(train_circuits, val_circuits, embedding_model, qnlp_model):   
    
    train_vocab=create_vocab_from_circuits(train_circuits)
    val_vocab=create_vocab_from_circuits(val_circuits)
    print(len(val_vocab.union(train_vocab)), len(train_vocab), len(val_vocab))    
    print(f"OOV word count: i.e out of {len(val_vocab)} words in the testing vocab there are  {len(val_vocab - train_vocab)} words that are not found in training. So they are OOV")
    oov_words=val_vocab - train_vocab
    print(f"list of OOV words are {oov_words}")     

    #calculate all out of vocabulary word count. Note: aldea is a word while aldea_0__s is a symbol
    oov_symbols={symb.name for d in val_circuits for symb in d.free_symbols} - {symb.name for d in train_circuits for symb in d.free_symbols}
    n_oov_symbs = len(oov_symbols)
    print(f'OOV symbol count: {n_oov_symbs} / {len({symb.name for d in val_circuits for symb in d.free_symbols})}')
    print(f"the symbols that are in symbol count but not in word count are:{oov_symbols-oov_words}")

   
    max_word_param_length=0
    if(ansatz_to_use==SpiderAnsatz):
        max_word_param_length_train = max(get_max_word_param_length(train_circuits))
        max_word_param_length_test = max(get_max_word_param_length(val_circuits))
        max_word_param_length = max(max_word_param_length_train, max_word_param_length_test) + 1

        """ max param length should include a factor from dimension
          for example if bakes is n.r@s, and n=2 and s=2, the parameter 
          length must be 4. """
        max_word_param_length = max_word_param_length * max (BASE_DIMENSION_FOR_SENT,BASE_DIMENSION_FOR_NOUN)

    assert max_word_param_length!=0
    
    if(ansatz_to_use==SpiderAnsatz):               
        #for each word in train and test vocab get its embedding from fasttext
        train_vocab_embeddings = get_vocab_emb_dict(train_vocab)
        val_vocab_embeddings = get_vocab_emb_dict(val_vocab)            
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
        if(ansatz_to_use==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            
            
            if cleaned_wrd_with_type in train_vocab_embeddings:
                if model_to_use == PytorchModel:                                                        
                    # dimension of initial param vector is decided by the actual dimension assigned in qnlp.weights for that word
                    list_of_params_for_this_word=[]
                    for i in range(len(weight)):
                        assert len(train_vocab_embeddings[cleaned_wrd_with_type]) > i
                        val= train_vocab_embeddings[cleaned_wrd_with_type][int(i)]
                        list_of_params_for_this_word.append(val)
                    tup= torch.tensor (list_of_params_for_this_word, requires_grad=True) #initializing with first two values of the embedding
                    initial_param_vector.append(tup)
                else:
                    initial_param_vector.append(train_vocab_embeddings[cleaned_wrd_with_type][int(idx)])
            else:                            
                print(f"ERROR: found that this word {cleaned_wrd_with_type} was OOV/not in fasttext emb")
             
    
    # assert len(qnlp_model.weights) == len(initial_param_vector)
    #also assert dimension of every single symbol/weight matches that of initial_para_vector
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
    dict1 = {symbol: param for symbol, param in zip(trained_qnlp_model.symbols, trained_qnlp_model.weights)}
    dict2 = {wrd: np.zeros(max_word_param_length) for wrd in train_vocab_embeddings}
    
    cleaned_wrd_with_type=""
    for symbol, trained_weights in dict1.items():
        if(ansatz_to_use==SpiderAnsatz):  
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
    
    build_model(keras_tuner.HyperParameters())  
    
    tuner = keras_tuner.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=3,
        executions_per_trial=2,
        overwrite=True,
        directory="tuning_model",
        project_name="oov_model3",
    )
    
    tuner.search(NN_train_X, np.array(NN_train_Y),validation_split=0.2, verbose=1, epochs=EPOCHS_MODEL3_OOV_MODEL)
    print(tuner.search_space_summary())
    models = tuner.get_best_models(num_models=2)
    

    best_model = models[0]
    best_model.summary()
    
    

    return best_model,dict2


def call_existing_code(lr,activation_oov):
    assert MAX_PARAM_LENGTH > 0
    OOV_NN_model = keras.Sequential([ layers.Dense(int((MAX_PARAM_LENGTH + MAXPARAMS) / 2), activation=activation_oov),
      layers.Dense( MAX_PARAM_LENGTH, activation= activation_oov),
    ])

    OOV_NN_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="mean_absolute_error",
        metrics=["accuracy"],
    )
    return OOV_NN_model


def build_model(hp):
    # units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation_oov =hp.Choice("activation", ["relu", "tanh","sigmoid","selu","softplus", "softmax","elu","exponential","leaky_relu","relu6","silu","hard_silu","gelu","hard_sigmoid","linear","mish","log_softmax"])
    loss_fn_oov =hp.Choice("loss", ["categorical_crossentropy", "binary_crossentropy","binary_focal_crossentropy","kl_divergence","softplus", "sparse_categorical_crossentropy","poisson","mean_squared_error","hinge"])
    # dropout = hp.Boolean("dropout")
    
    lr = hp.Float("lr", min_value=1e-6, max_value=1e-1, sampling="linear")
    # call existing model-building code with the hyperparameter values.
    model = call_existing_code(lr=lr, activation_oov=activation_oov)
    return model

def evaluate_val_set(pred_model, val_circuits, val_labels, trained_weights, val_vocab_embeddings, max_word_param_length, OOV_strategy='random', OOV_model=None):   
    pred_parameter_map = {}

    #Use the words from train wherever possible, else use DNN prediction
    for wrd, embedding in val_vocab_embeddings.items():
        if OOV_strategy == 'model':
            pred_parameter_map[wrd] = trained_weights.get(wrd, OOV_model.predict(np.array([embedding]), verbose=0)[0])
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
        if(ansatz_to_use==SpiderAnsatz):  
            cleaned_wrd_just_plain_text,cleaned_wrd_with_type =  clean_wrd_for_spider_ansatz(sym.name)
            rest = sym.name.split('_', 1)[1]
            idx = rest.split('__')[0]      
            if cleaned_wrd_with_type in pred_parameter_map:
                if model_to_use == PytorchModel:
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
    l= loss_pyTorch(preds, torch.tensor(val_labels))
    a=accuracy(preds, torch.tensor(val_labels))

    return l, a

def read_data(filename):         
            labels, sentences = [], []
            with open(filename) as f:
                for line in f:           
                    t = float(line[0])            
                    labels.append([t, 1-t])            
                    sentences.append(line[1:].strip())
            return labels, sentences

#back to the main thread after all functions are defined.

#read the base data, i.e plain text english.
train_labels, train_data = read_data(os.path.join(DATA_BASE_FOLDER,TRAIN))
val_labels, val_data = read_data(os.path.join(DATA_BASE_FOLDER,DEV))
test_labels, test_data = read_data(os.path.join(DATA_BASE_FOLDER,TEST))




def convert_to_diagrams(list_sents,labels):
    list_target = []
    labels_target = []
    sent_count_longer_than_32=0
    for sent, label in tqdm(zip(list_sents, labels),desc="reading sent"):                        
        tokenized = spacy_tokeniser.tokenise_sentence(sent)                
        if( ansatz_to_use==SpiderAnsatz ):
            if len(tokenized)> 32:
                print(f"no of tokens in this sentence is {len(tokenized)}")
                sent_count_longer_than_32+=1
                continue
        spiders_diagram = parser_to_use_obj.sentence2diagram(sentence=sent)
        list_target.append(spiders_diagram)
        labels_target.append(label)
    
    print(f"sent_count_longer_than_32={sent_count_longer_than_32}")
    print("no. of items processed= ", len(list_target))
    return list_target, labels_target

#convert the plain text input to ZX diagrams
train_diagrams = parser_to_use_obj.sentences2diagrams(train_data)
val_diagrams = parser_to_use_obj.sentences2diagrams(val_data)
test_diagrams = parser_to_use_obj.sentences2diagrams(test_data)

train_X = []
val_X = []

print(f"count of train, test, val elements respectively are: ")
print({len(train_diagrams)}, {len(test_diagrams)}, {len(val_diagrams)})
assert len(train_diagrams)== len(train_labels)
assert len(val_diagrams)== len(val_labels)
assert len(test_diagrams)== len(test_labels)

def run_experiment(MAX_WORD_PARAM_LEN,nlayers=1, seed=SEED):
    if ansatz_to_use in [IQPAnsatz,Sim15Ansatz, Sim14Ansatz]:
        ansatz = ansatz_to_use({AtomicType.NOUN: BASE_DIMENSION_FOR_NOUN,
                    AtomicType.SENTENCE: BASE_DIMENSION_FOR_SENT,
                    AtomicType.PREPOSITIONAL_PHRASE: BASE_DIMENSION_FOR_PREP_PHRASE} ,n_layers= nlayers,n_single_qubit_params =3)    
    else:
        ansatz = ansatz_to_use({AtomicType.NOUN: Dim(BASE_DIMENSION_FOR_NOUN),
                    AtomicType.SENTENCE: Dim(BASE_DIMENSION_FOR_SENT)}  )    
    
   
        #use the anstaz to create circuits from diagrams
        train_circuits =  [ansatz(diagram) for diagram in train_diagrams]
        val_circuits =  [ansatz(diagram) for diagram in val_diagrams]
        test_circuits = [ansatz(diagram) for diagram in test_diagrams]        
   
    print("length of each circuit in train is:")
    print([len(x) for x in train_circuits])

    if(model_to_use==TketModel):
        backend = AerBackend()
        backend_config = {
                    'backend': backend,
                    'compilation': backend.default_compilation_pass(2),
                    'shots': 8192
                }
        qnlp_model= TketModel.from_diagrams(train_circuits, backend_config=backend_config)
    else:
        qnlp_model = model_to_use.from_diagrams(train_circuits+val_circuits )

    train_dataset = Dataset(
                train_circuits,
                train_labels,
                batch_size=BATCH_SIZE)

    val_dataset = Dataset(val_circuits, val_labels, shuffle=False)

    print(len(train_labels), len(train_circuits))
    
    print(len(train_circuits), len(val_circuits), len(test_circuits))
    assert len(train_circuits)== len(train_labels)
    assert len(val_circuits)== len(val_labels)
    assert len(test_circuits)== len(test_labels)


    if(trainer_to_use==QuantumTrainer):
        trainer = QuantumTrainer(
        model=qnlp_model,
        loss_function=BinaryCrossEntropyLoss(),
        epochs=EPOCHS_TRAIN_MODEL1,
        optimizer=SPSAOptimizer,
        optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.001*EPOCHS_TRAIN_MODEL1},
        evaluate_functions=eval_metrics,
        evaluate_on_train=True,
        verbose='text',
        log_dir='RelPron/logs',
        seed=SEED
        )
    else:
        trainer = trainer_to_use(
            model=qnlp_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=LEARNING_RATE,            
            epochs=EPOCHS_TRAIN_MODEL1,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=SEED)
    

    train_embeddings, val_embeddings, max_w_param_length, oov_word_count = generate_initial_parameterisation(
        train_circuits, val_circuits, embedding_model, qnlp_model)

    global MAX_PARAM_LENGTH
    MAX_PARAM_LENGTH = max_w_param_length
    trainer.fit(train_dataset,val_dataset=val_dataset, eval_interval=1, log_interval=1)
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

   
    NN_model,trained_wts = generate_OOV_parameterising_model(qnlp_model, train_embeddings, max_w_param_length)
    prediction_model = model_to_use.from_diagrams(val_circuits)

    trainer = trainer_to_use(
            model=prediction_model,
            loss_function=torch.nn.BCEWithLogitsLoss(),
            optimizer=torch.optim.AdamW,
            learning_rate=LEARNING_RATE,
            epochs=EPOCHS_TRAIN_MODEL1,
            evaluate_functions=eval_metrics,
            evaluate_on_train=True,
            verbose='text',
            seed=SEED)

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

    
    res =  {
            'NN': (smart_loss, smart_acc)
            
            
           }
    
    return smart_loss.item(), smart_acc.item()



"""####final push-which calls run_experiment function above
todo: why is he setting random seed, that tooin tensor flow
- especially since am using a pytorch model.
"""
import tensorflow as tf
compr_results = {}

#ideally should be more than 1 seed. But  commenting out due to lack of ram in laptop
tf_seeds = [2]

for tf_seed in tf_seeds:
    tf.random.set_seed(tf_seed)
    this_seed_results = []    
    for nl in [3]:
        this_seed_results.append([run_experiment(nl, tf_seed)])
    compr_results[tf_seed] = this_seed_results

print(f"\nvalue of all evaluation metrics across all seeds is :")

for k,v in compr_results.items():
    print(f"\n{k}: {v}\n")
