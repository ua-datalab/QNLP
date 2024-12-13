import sys
import classify
import pytest
import time

# """
# some definitions
#  Classical 1= the combination of (Spider parser, spider ansatz, pytorch model, pytorchtrainer)
#  Classical 2= the combination of (bobCatParser , spider ansatz, pytorch model, pytorchtrainer)
#  Quantum1= (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)- this runs on a simulation f a quantum computer
#  Quantum2 = **Quantum2-simulation-actual quantum computer**=(penny lane model, bob cat parser, iqp ansatz, pytorchtrainer):
# """

""" 
python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10  --max_tokens_per_sent 10
"""
def test_sst2_classical1_no_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'sst2',
                             '--parser', 'Spider',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10'                             
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.6
        assert round(model4_loss,2)  <= 0.7
        assert round(model4_acc,2)  >= 0.5 
        assert round(model4_acc,2)  <= 0.65 


        # test_sst2_classical1_yes_expose_val
#python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --expose_model1_val_during_model_initialization --max_tokens_per_sent 10
def test_sst2_classical1_yes_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'sst2',
                             '--parser', 'Spider',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',
                             '--expose_model1_val_during_model_initialization',                             
                             '--max_tokens_per_sent', '10'                             
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.6
        assert round(model4_loss,2)  <= 0.7
        assert round(model4_acc,2)  >= 0.4 
        assert round(model4_acc,2)  <= 0.7 


# def test_food_it_classical1(monkeypatch):
#         monkeypatch.setattr(sys, 'argv', 
#                             ['classify.py', 
#                              '--dataset', 'food_it',
#                              '--parser', 'Spider'
#                              ])
#         smart_loss, smart_acc=classify.main()                         
#         assert round(smart_loss,2)  == 0.32
#         assert round(smart_acc,1)  >= 0.8 
#         assert round(smart_acc,1)  <= 0.9 





"""#for a combination of all classical 2 components+train=20,val=10,test=10 + not exposing val data during initialization of model 1
the actual command used to run this will be: 
python classify.py --dataset sst2 --parser BobCatParser --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10
"""

def test_sst2_classical2_no_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'sst2',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10'                             
                             ])
        try:

                model4_loss, model4_acc=classify.main()                         
        except Exception as ex:
                print(ex)
                assert type(ex) == RuntimeError


"""#for a combination of all classical 1 components+train=20,val=10,test=10 + yes exposing val data during initialization of model 1
the actual command used to run this will be: 
python classify.py --dataset sst2 --parser BobCatParser 
--ansatz SpiderAnsatz --model14type PytorchModel 
--trainer PytorchTrainer --epochs_train_model1 7 
--no_of_training_data_points_to_use 20 
--no_of_val_data_points_to_use 10 --max_tokens_per_sent 10 
 --expose_model1_val_during_model_initialization
"""
   
def test_sst2_classical2_yes_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'sst2',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10' ,
                             '--expose_model1_val_during_model_initialization'                                                       
                             ])
        try:

                model4_loss, model4_acc=classify.main()                         
        except Exception as ex:
                print(ex)
                assert type(ex) == RuntimeError
                





# python classify.py --dataset uspantek --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10

def test_uspantek_classical1_no_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'Spider',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10'                             
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.6
        assert round(model4_loss,2)  <= 0.75
        assert round(model4_acc,2)  >= 0.3 
        assert round(model4_acc,2)  <= 0.5 



def test_uspantek_classical1_yes_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'Spider',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10'  ,
                             '--expose_model1_val_during_model_initialization'                                                                                  
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.6
        assert round(model4_loss,2)  <= 0.7
        assert round(model4_acc,2)  >= 0.4 
        assert round(model4_acc,2)  <= 0.6 


# python classify.py --dataset uspantek --parser BobCatParser --ansatz SpiderAnsatz --model14type PytorchModel 
# --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20
#  --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10

def test_uspantek_classical2_no_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10'                             
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.68
        assert round(model4_loss,2)  <= 0.75
        assert round(model4_acc,2)  >= 0.5 
        assert round(model4_acc,2)  <= 0.7 


# python classify.py --dataset uspantek --parser BobCatParser --ansatz SpiderAnsatz --model14type PytorchModel  --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20  --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10  --expose_model1_val_during_model_initialization


def test_uspantek_classical2_yes_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '7',
                             '--no_of_training_data_points_to_use', '20',
                             '--no_of_val_data_points_to_use', '10',                             
                             '--max_tokens_per_sent', '10' ,
                             '--expose_model1_val_during_model_initialization'                                                                                                              
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  >= 0.6
        assert round(model4_loss,2)  <= 0.75
        assert round(model4_acc,2)  >= 0.3 
        assert round(model4_acc,2)  <= 0.6 


# python classify.py --dataset uspantek --parser BobCatParser --ansatz IQPAnsatz --model14type TketModel --trainer QuantumTrainer --epochs_train_model1 30 --no_of_training_data_points_to_use 70 --no_of_val_data_points_to_use 30 --max_tokens_per_sent 10 


def test_uspantek_quantum1_no_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'IQPAnsatz',
                             '--model14type', 'TketModel',
                             '--trainer', 'QuantumTrainer',
                             '--epochs_train_model1', '30',
                             '--no_of_training_data_points_to_use', '70',
                             '--no_of_val_data_points_to_use', '30',                             
                             '--max_tokens_per_sent', '10'                              
                             ])
        try:
                model4_loss, model4_acc=classify.main()                         
        except Exception as ex:
                print(ex)
                assert type(ex) == KeyError
      

# python classify.py --dataset uspantek --parser BobCatParser --ansatz IQPAnsatz --model14type TketModel --trainer QuantumTrainer --epochs_train_model1 30 --no_of_training_data_points_to_use 70 --no_of_val_data_points_to_use 30 --max_tokens_per_sent 10 ----expose_model1_val_during_model_initialization


def test_uspantek_quantum1_yes_expose_val(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'uspantek',
                             '--parser', 'BobCatParser',
                             '--ansatz', 'IQPAnsatz',
                             '--model14type', 'TketModel',
                             '--trainer', 'QuantumTrainer',
                             '--epochs_train_model1', '30',
                             '--no_of_training_data_points_to_use', '70',
                             '--no_of_val_data_points_to_use', '30',                             
                             '--max_tokens_per_sent', '10' ,
                             '--expose_model1_val_during_model_initialization'                             
                             ])
        try:
                model4_loss, model4_acc=classify.main()                         
        except Exception as ex:
                print(ex)
                assert type(ex) == KeyError
      