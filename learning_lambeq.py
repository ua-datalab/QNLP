import warnings
warnings.filterwarnings('ignore')

from lambeq import AtomicType, BobcatParser, TensorAnsatz
from lambeq.backend.tensor import Dim

# Define atomic types
N = AtomicType.NOUN
S = AtomicType.SENTENCE
C = AtomicType.CONJUNCTION


# Parse a sentence
parser = BobcatParser(verbose='suppress')
diagram = parser.sentence2diagram('Alice Loves Bob ')

# Apply a tensor ansatz
ansatz = TensorAnsatz({N: Dim(1), S: Dim(2)})
tensor_diagram = ansatz(diagram)
tensor_diagram.draw(figsize=(12,5), fontsize=12)
print(tensor_diagram.free_symbols)
print([(s, s.size) for s in tensor_diagram.free_symbols])
