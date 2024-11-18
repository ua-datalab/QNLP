# content of test_sysexit.py
import pytest
import OOV_classification_no_pair_sents
from OOV_classification_no_pair_sents import run_experiment, parser_to_use, ansatz_to_use, model_to_use, trainer_to_use, TYPE_OF_DATA_TO_USE, embedding_model_to_use
import os
import math
from lambeq import PytorchModel, NumpyModel, TketModel, PennyLaneModel
from lambeq import TensorAnsatz,SpiderAnsatz,Sim15Ansatz, IQPAnsatz,Sim14Ansatz
from lambeq import BobcatParser,spiders_reader
from lambeq import QuantumTrainer, PytorchTrainer



class TestClass:
    def test_run_expt_english_food_it_baseline(self):                                                    
                    assert parser_to_use == BobcatParser
                    assert model_to_use == PytorchModel
                    assert ansatz_to_use == SpiderAnsatz
                    assert trainer_to_use == PytorchTrainer

                    smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)                                        
                    assert round(smart_loss,2)  == 0.32
                    assert round(smart_acc,1)  >= 0.8 
                    assert round(smart_acc,1)  <= 0.9 
                    

    # def test_run_expt_spanish(self):
    #                 smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)                                        
    #                 assert round(smart_loss,2)  == 0.69 
    #                 assert round(smart_acc,2)  == 0.43

                    #these are the values, when using english embeddings. 
                    """2: [[(0.6931483745574951, 0.5166666507720947)]] see that little bump 
                    from 0.433 to 0.516 that is what blind faith did for you. that's how 
                    you make the patient respond. tada.
                    """ 
