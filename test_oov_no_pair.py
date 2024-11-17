# content of test_sysexit.py
import pytest
import OOV_classification_no_pair_sents
from OOV_classification_no_pair_sents import run_experiment


def test_run_expt():            
            smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)
            assert smart_loss >= 0.695 
            assert smart_acc >= 0.433

            #these are the values, when using english embeddings. 
            """2: [[(0.6931483745574951, 0.5166666507720947)]] see that little bump 
            from 0.433 to 0.516 that is what blind faith did for you. that's how 
            you make the patient respond. tada.
            """ 
