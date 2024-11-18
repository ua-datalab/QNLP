# content of test_sysexit.py
import pytest
import OOV_classification_no_pair_sents
from OOV_classification_no_pair_sents import run_experiment
from OOV_classification_no_pair_sents import arch
import os
import math


def test_run_expt():
            print(f"arch={arch}")
            smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)
            if(arch=="<class 'lambeq.ansatz.tensor.SpiderAnsatz'>+<lambeq.text2diagram.bobcat_parser.BobcatParser object at 0x10321ee70>+<class 'lambeq.training.pytorch_trainer.PytorchTrainer'>+<class 'lambeq.training.pytorch_model.PytorchModel'>+english"):
                print(f"smart_loss={smart_loss}")
                print(f"smart_acc={smart_acc}")
                assert round(smart_loss,2)  == 0.69 
                assert round(smart_acc,2)  == 0.43
            else:
                    print("*************error")
                

            #these are the values, when using english embeddings. 
            """2: [[(0.6931483745574951, 0.5166666507720947)]] see that little bump 
            from 0.433 to 0.516 that is what blind faith did for you. that's how 
            you make the patient respond. tada.
            """ 
test_run_expt()