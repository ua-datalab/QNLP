import warnings
warnings.filterwarnings('ignore')

from lambeq import AtomicType, BobcatParser, TensorAnsatz, SpiderAnsatz, IQPAnsatz,Sim14Ansatz, Sim15Ansatz
from lambeq.backend.tensor import Dim

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
C = AtomicType.CONJUNCTION

ansatz_to_use= SpiderAnsatz 

# Parse a sentence
parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram('Alice Loves Bob ')

# Apply a tensor ansatz
if(ansatz_to_use==Sim15Ansatz):
    ansatz=ansatz_to_use({AtomicType.NOUN: 4, AtomicType.SENTENCE: 2},
                   n_layers=1, n_single_qubit_params=3)
if(ansatz_to_use==SpiderAnsatz):
    ansatz = ansatz_to_use({AtomicType.NOUN: Dim(4),
                       AtomicType.SENTENCE: Dim(2)})
    

tensor_diagram = ansatz(diagram)
# tensor_diagram.draw(figsize=(12,5), fontsize=12)
print(f"when using bobcat parser and {ansatz_to_use} ")
print(tensor_diagram.free_symbols)
max_word_param_length= 9999
if(ansatz_to_use==Sim15Ansatz):
    all_symb=[]
    a=0
    for symb in tensor_diagram.free_symbols:
        all_symb.append(symb.name.rsplit('_', 1)[1])
    a=max(all_symb)

    max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for symb in tensor_diagram.free_symbols ),
                            max(int(symb.name.rsplit('_', 1)[1]) for symb in tensor_diagram.free_symbols)) + 1

if(ansatz_to_use==SpiderAnsatz):    
    all=[]
    for symb in tensor_diagram.free_symbols:
       x=  symb.name.split('_', 1)[1]
       y= x.split('__')[0]
       all.append(y)

    max_word_param_length = max(max(int(symb.name.split('_', 1)[1]) for symb in tensor_diagram.free_symbols),
                            max(int(symb.name.rsplit('_', 1)[1]) for symb in tensor_diagram.free_symbols)) + 1


print(f"value of max_word_param_length={max_word_param_length}")
    


import sys
sys.exit()
