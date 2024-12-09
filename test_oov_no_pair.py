import sys
import classify

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

def test_sst2_classical1(monkeypatch):        
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'sst2',
                             '--parser', 'Spider',
                             '--ansatz', 'SpiderAnsatz',
                             '--model14type', 'PytorchModel',
                             '--trainer', 'PytorchTrainer',
                             '--epochs_train_model1', '50'
                             ])
        model4_loss, model4_acc=classify.main()                         
        assert round(model4_loss,2)  == 0.32
        assert round(model4_acc,1)  >= 0.8 
        assert round(model4_acc,1)  <= 0.9 
