# content of test_sysexit.py
from classify import parse_arguments

def test_parse_args(monkeypatch):
    """Test that arguments are parsed correctly."""
    mock_args = ["classify.py", "--dataset", "spanish"]
    monkeypatch.setattr("sys.argv", mock_args)
    args = parse_arguments()
    
    assert args.dataset == "spanish"



def test_perform_task_food_it(self):
    args = self.parse_arguments()
    assert args.dataset == "food_it"
    assert args.parser == BobcatParser
    assert args.model == PytorchModel
    assert args.ansatz == SpiderAnsatz
    assert args.trainer == PytorchTrainer
    smart_loss, smart_acc=perform_task(args)                         
    assert round(smart_loss,2)  > 0.32
    assert round(smart_loss,2)  < 0.5
    assert round(smart_acc,1)  >= 0.8 
    assert round(smart_acc,1)  <= 0.9 

  
  
