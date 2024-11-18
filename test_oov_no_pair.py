# content of test_sysexit.py
import pytest
import OOV_classification_no_pair_sents
from OOV_classification_no_pair_sents import run_experiment
import os
import math

class TestClass:
    def test_run_expt_english(self):                                    
                    smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)                                        
                    assert round(smart_loss,2)  == 0.69 
                    assert round(smart_acc,2)  == 0.51
                    

    # def test_run_expt_spanish(self):
    #                 smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)                                        
    #                 assert round(smart_loss,2)  == 0.69 
    #                 assert round(smart_acc,2)  == 0.43

                    #these are the values, when using english embeddings. 
                    """2: [[(0.6931483745574951, 0.5166666507720947)]] see that little bump 
                    from 0.433 to 0.516 that is what blind faith did for you. that's how 
                    you make the patient respond. tada.
                    """ 
