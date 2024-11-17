import warnings
warnings.filterwarnings('ignore')

from lambeq import AtomicType, BobcatParser, TensorAnsatz, SpiderAnsatz, IQPAnsatz,Sim14Ansatz, Sim15Ansatz
from lambeq.backend.tensor import Dim
from discopy.quantum.gates import CX, Rx, H, Bra, Id


# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
C = AtomicType.CONJUNCTION

ansatz_to_use= Sim15Ansatz 

# Parse a sentence
parser = BobcatParser()
diagram = parser.sentence2diagram('Alice Loves Bob ')
train_diagrams=[diagram]


from lambeq import AtomicType, IQPAnsatz, RemoveCupsRewriter

ansatz = IQPAnsatz({AtomicType.NOUN: 1, AtomicType.SENTENCE: 0},
                   n_layers=1, n_single_qubit_params=3)
remove_cups = RemoveCupsRewriter()

train_circuits = [ansatz(remove_cups(diagram)) for diagram in train_diagrams]


train_circuits[0].draw(figsize=(9, 10))

from tqdm import tqdm
from lambeq import Rewriter, remove_cups

rewriter = Rewriter(['prepositional_phrase', 'determiner', 'coordination', 'connector', 'prepositional_phrase'])

train_X = []
test_X = []

for d in tqdm(train_diags_raw):
    train_X.append(remove_cups(rewriter(d).normal_form()))

N = AtomicType.NOUN
S = AtomicType.SENTENCE
P = AtomicType.PREPOSITIONAL_PHRASE
equality_comparator = (CX >> (H @ Rx(0.5)) >> (Bra(0) @ Id(1)))
ansatz = IQPAnsatz({N: 1, S: 1, P:1}, n_layers=1, n_single_qubit_params=3)
train_circs = ansatz(diagram) 
ansatz = SpiderAnsatz({AtomicType.NOUN: Dim(4),
                       AtomicType.SENTENCE: Dim(2)})
    

tensor_diagram = ansatz(diagram)
from discopy.quantum.gates import CX, Rx, H, Bra, Id

b= tensor_diagram>> equality_comparator
# tensor_diagram.draw(figsize=(12,5), fontsize=12)
print(f"when using bobcat parser and {ansatz_to_use} ")
print(tensor_diagram.free_symbols)
max_word_param_length= 9999
if(ansatz_to_use==Sim15Ansatz):
    c = ansatz()
# (    all_symb=[]
#     a=0
#     for symb in tensor_diagram.free_symbols:
#         all_symb.append(symb.name.rsplit('_', 1)[1])
#     a=max(all_symb)

#     max_word_param_length = max(max(int(symb.name.rsplit('_', 1)[1]) for symb in tensor_diagram.free_symbols ),
#                             max(int(symb.name.rsplit('_', 1)[1]) for symb in tensor_diagram.free_symbols)) + 1

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
