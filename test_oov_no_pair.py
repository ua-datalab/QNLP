import sys
import classify

def test_food_it_classical1(monkeypatch):
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'food_it',
                             '--parser', 'Spider'
                             ])
        smart_loss, smart_acc=classify.main()                         
        assert round(smart_loss,2)  == 0.32
        assert round(smart_acc,1)  >= 0.8 
        assert round(smart_acc,1)  <= 0.9 

def test_food_it_classical2(monkeypatch):
        monkeypatch.setattr(sys, 'argv', 
                            ['classify.py', 
                             '--dataset', 'food_it',
                             '--parser', 'BobcatParser'
                             ])
        smart_loss, smart_acc=classify.main()                         
        assert round(smart_loss,2)  == 0.32
        assert round(smart_acc,1)  >= 0.8 
        assert round(smart_acc,1)  <= 0.9 
