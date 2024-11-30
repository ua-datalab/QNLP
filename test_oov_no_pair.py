# content of test_sysexit.py

from lambeq import PytorchModel, NumpyModel, TketModel, PennyLaneModel
from lambeq import TensorAnsatz,SpiderAnsatz,Sim15Ansatz, IQPAnsatz,Sim14Ansatz
from lambeq import BobcatParser,spiders_reader
from lambeq import QuantumTrainer, PytorchTrainer
from classify import main, parse_arguments, perform_task
import argparse



class TestClass:


    
    def parse_arguments(self):
        parser = argparse.ArgumentParser(description="Description of your script.")
        parser.add_argument('--dataset', type=str, required=False, default="food_it" ,help="type of dataset-choose from uspantek,spanish,food_it,msr_paraphrase_corpus")
        parser.add_argument('--parser', type=CCGParser, required=False, default=BobcatParser, help="type of parser to use: [tree_reader,bobCatParser, spiders_reader,depCCGParser]")
        parser.add_argument('--ansatz', type=BaseAnsatz, required=False, default=SpiderAnsatz, help="type of ansatz to use: [IQPAnsatz,SpiderAnsatz,Sim14Ansatz, Sim15Ansatz,TensorAnsatz ]")
        parser.add_argument('--model', type=Model, required=False, default=PytorchModel , help="type of model to use: [numpy, pytorch,TketModel]")
        parser.add_argument('--trainer', type=Trainer, required=False, default=PytorchTrainer, help="type of trainer to use: [PytorchTrainer, QuantumTrainer]")
        parser.add_argument('--max_param_length_global', type=int, required=False, default=0, help="a global value which will be later replaced by the actual max param length")
        parser.add_argument('--do_model3_tuning', type=bool, required=False, default=False, help="only to be used during training, when a first pass of code works and you need to tune up for parameters")
        parser.add_argument('--base_dimension_for_noun', type=int, default=2, required=False, help="")
        parser.add_argument('--base_dimension_for_sent', type=int, default=2, required=False, help="")
        parser.add_argument('--base_dimension_for_prep_phrase', type=int, default=2, required=False, help="")
        parser.add_argument('--maxparams', type=int, default=300, required=False, help="")
        parser.add_argument('--batch_size', type=int, default=30, required=False, help="")
        parser.add_argument('--epochs_train_model1', type=int, default=30, required=False, help="")
        parser.add_argument('--epochs_model3_oov_model', type=int, default=100, required=False, help="")
        parser.add_argument('--learning_rate_model1', type=float, default=3e-2, required=False, help="")
        parser.add_argument('--seed', type=int, default=0, required=False, help="")
        parser.add_argument('--data_base_folder', type=str, default="data", required=False, help="")
        parser.add_argument('--learning_rate_model3', type=float, default=3e-2, required=False, help="")     
        return parser.parse_args()



    def test_perform_task_food_it(self):
        args = parse_arguments()
        assert args.parser == BobcatParser
        assert args.model == PytorchModel
        assert args.ansatz == SpiderAnsatz
        assert args.trainer == PytorchTrainer
        smart_loss, smart_acc=perform_task(args)                         
        assert round(smart_loss,2)  == 0.32
        assert round(smart_acc,1)  >= 0.8 
        assert round(smart_acc,1)  <= 0.9 
                    

  
