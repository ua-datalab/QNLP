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


#for a combination of train=20,val=10,test=10+exposing val data during initialization of model 1
def test_sst2_classical1(monkeypatch):        
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
        assert round(model4_acc,1)  >= 0.5 
        assert round(model4_acc,1)  <= 0.6 


#for a combination of train=20,val=10,test=10 +not exposing val data during initialization of model 1
def test_sst2_classical1_no_expose(monkeypatch):        
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
        assert round(model4_acc,1)  >= 0.5 
        assert round(model4_acc,1)  <= 0.6 


#for a combination of train=20,val=10,test=10 +not exposing val data during initialization of model 1
def test_sst2_classical2_no_expose(monkeypatch):        
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
        try:

                model4_loss, model4_acc=classify.main()                         
        except Exception as ex:
                print(ex)
                assert type(ex) == RuntimeError
                