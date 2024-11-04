# -*- coding: utf-8 -*-
"""learning_lambeq_qnlp_quantum_model_toy_data

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Vr62XT2I2k0XO0bidYimVCMhMfdBqjr2
"""

#below code run only once per session
import os, sys
# from google.colab import drive
# drive.mount('/content/drive')

# #below code run only once-ever
# nb_path = '/content/notebooks'
# #os.symlink('/content/drive/My Drive/Colab Notebooks', nb_path)
# sys.path.insert(0,nb_path)

# !pip install lambeq --target=$nb_path lambeq
# !pip install --target=$nb_path PennyLane
# !pip install --target=$nb_path pytket-qiskit
# !pip install --target=$nb_path numpy

# !pip install lambeq
# !pip install pytket-qiskit
# !pip install  numpy
# !pip install "discopy>=1.1.0"

import os
import warnings
import numpy as np
import lambeq
from lambeq import BobcatParser
from lambeq import IQPAnsatz, AtomicType

warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLELISM"]="true"
BATCH_SIZE=30
EPOCHS=1000
SEED=33

# SENTENCES=["cat is on the mat","cat is not on the mat"]
SENTENCES=["alice loves bob","Alice has a deep affection for Bob and holds him in a special place in her heart"]
LABELS=[0,1]

train_data=SENTENCES
train_labels=[[1,0],[0,1]]

val_data=train_data
val_labels=train_labels

print(train_labels[:1])
print(len(train_labels))
print(len(train_data))

parser=BobcatParser(verbose='text')
diagrams=parser.sentences2diagrams(train_data)
# diagrams[0].draw()
# diagrams[1].draw()

# import warnings
# warnings.filterwarnings('ignore')

# from lambeq import AtomicType, BobcatParser, TensorAnsatz
# from lambeq.backend.tensor import Dim

# # Define atomic types
# N = AtomicType.NOUN
# S = AtomicType.SENTENCE

# # Parse a sentence
# parser = BobcatParser(verbose='progress')
# diagram = parser.sentence2diagram('black')
# diagram.draw()

train_diagrams_simplified = [diagram.normal_form() for diagram in diagrams if diagram is not None]
# train_diagrams_simplified[0].draw()
# train_diagrams_simplified[1].draw()

train_labels = [label for (diagram, label) in zip(train_diagrams_simplified, train_labels) if diagram is not None]

ansatz=IQPAnsatz({AtomicType.NOUN:1,AtomicType.SENTENCE:1},n_layers=1,n_single_qubit_params=3)
train_circuits=[ansatz(diagram) for diagram in train_diagrams_simplified]

# train_circuits[0].draw(draw_as_nodes=True,figsize=(200,500))
# train_circuits[1].draw(draw_as_nodes=True,figsize=(10,5))



from lambeq import TketModel

from lambeq import RemoveCupsRewriter
remove_cups = RemoveCupsRewriter()


train_circuits=[ansatz(remove_cups(diagram)) for diagram in train_diagrams_simplified]

# train_circuits[0].draw(figsize=(10,10))
# train_circuits[1].draw(figsize=(10,10))

def convert_lambeq_circuit_to_tket_to_qasm(circuit, filename):
    from pytket.circuit.display import render_circuit_jupyter
    tket_circuit = circuit.to_tk()
    render_circuit_jupyter(tket_circuit)
    from pytket.qasm import circuit_to_qasm_str
    # Convert the PyTKet circuit to QASM format
    qasm_output = circuit_to_qasm_str(tket_circuit)
    print(qasm_output)
    with open(filename, "w") as qasm_file:
        qasm_file.write(qasm_output)

convert_lambeq_circuit_to_tket_to_qasm(train_circuits[0],"alice1")
convert_lambeq_circuit_to_tket_to_qasm(train_circuits[1],"alice2")
import sys
sys.exit(1)
# from pytket.extensions.qiskit import tk_to_qiskit

# qiskit_circuit = tk_to_qiskit(tket_circuit)




from pytket.extensions.qiskit import AerBackend, IBMQBackend
from lambeq import TketModel
from pytket.extensions.qiskit import set_ibmq_config

set_ibmq_config(ibmq_api_token='3a3b06f905ca0402c001cba1c9f9ade39ef572e0c61ea6421d71b104103b0405c85cb7250572d925826236a093eee6561b3b4ebfc5827da1a45ab1999ce3f3a9')

backend=IBMQBackend('ibm_brisbane')
print(backend.available_devices())

backend_config={
    'backend':backend,
    'shots': 8192,
    'compilation': backend.default_compilation_pass(2),
}


model=TketModel.from_diagrams(train_circuits,backend_config=backend_config)

from lambeq import BinaryCrossEntropyLoss

# Using the builtin binary cross-entropy error from lambeq
bce = BinaryCrossEntropyLoss()

acc = lambda y_hat, y: np.sum(np.round(y_hat) == y) / len(y) / 2  # half due to double-counting
eval_metrics = {"acc": acc}

from lambeq import QuantumTrainer, SPSAOptimizer

trainer = QuantumTrainer(
    model,
    loss_function=bce,
    epochs=EPOCHS,
    optimizer=SPSAOptimizer,
    optim_hyperparams={'a': 0.05, 'c': 0.06, 'A':0.01*EPOCHS},
    evaluate_functions=eval_metrics,
    evaluate_on_train=True,
    verbose = 'text',
    seed=0
)

from lambeq import Dataset

train_dataset = Dataset(
            train_circuits,
            train_labels,
            batch_size=BATCH_SIZE)
#temporary hack
val_circuits=train_circuits
val_dataset = Dataset(val_circuits, val_labels, shuffle=False)
trainer.fit(train_dataset, val_dataset, evaluation_step=1, logging_step=100)