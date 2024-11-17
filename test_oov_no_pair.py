# content of test_sysexit.py
import pytest
import OOV_classification_no_pair_sents
from OOV_classification_no_pair_sents import run_experiment

class TestClass:
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")

    def test_mytest():
        with pytest.raises(SystemExit):
            
            smart_loss, smart_acc= run_experiment(nlayers=3, seed=2)
            assert smart_loss >= 0.695 
            assert smart_acc >= 0.433
