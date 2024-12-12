# QNLP
**Requirements:**
- Minimum 16 GB ram to load the fasttext model and lambeq models

**Steps for setup:**

- `conda create --name qnlp_temp7 python==3.11.10`
- `conda activate qnlp_temp7` 
- `./run_me_first.sh`


"""
some definitions
 Classical 1= the combination of (Spider parser, spider ansatz, pytorch model, pytorchtrainer)
 Classical 2= the combination of (bobCatParser , spider ansatz, pytorch model, pytorchtrainer)
 Quantum1= (IQPansatz+TKetmodel+Quantum Trainer+ bob cat parser)- this runs on a simulation f a quantum computer
 Quantum2 = **Quantum2-simulation-actual quantum computer**=(penny lane model, bob cat parser, iqp ansatz, pytorchtrainer):
"""


- Get embeddings files:
  - English:
    - `wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"`
    - `gunzip cc.en.300.bin.gz`
  - Spanish:
    - `wget -c "https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1" -O ./embeddings-l-model.bin`
- Run `pytest`
  - If you encounter any modulenotfound errors, use `pip install <module>` to add the needed libraries

**Steps for accessing and using data files:**
- Clone this reposory to access the data files (e.g. spanish_test.txt): [Private Repository](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data).
- Copy the folder qnlp_data to the QNLP repository: `cp -r usp_qnlp/qnlp-data QNLP/data`
- Create a new branch to test the code: `git checkout -b test_branch`


Once all requirements and the environment has been set up, you can begin training the QNLP model and testing it.

**Steps for training and testing the QNLP Model:**
- Run `python classify.py`

**Note:** the last line of ./run_me_first.sh will try to download a 5GB file. Alternately you can download spanish fasttext embeddings: go to [this](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file#fasttext-embeddings-from-suc) url and download manually the `.bin` file for spanish unannotated corpora to the same location where this code is.

## some sample command to run this code as of dec 10th 2024

- `pytest` (ensure nothing fails)

#### To run on dataset sst2 classical 1 combination (i.e SpiderReader, SpiderAnsatz, PytorchModel,PytorchTrainer+ yes expose val)

```bash
python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --expose_model1_val_during_model_initialization --max_tokens_per_sent 10
```



#### To run on dataset sst2 classical 2 combination (i.e BobCatParser, SpiderAnsatz, PytorchModel,PytorchTrainer+ no expose val)

```bash
python classify.py --dataset sst2 --parser BobCatParser --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10 
```

#### To run on dataset sst2 quantum1 combination 

```bash
python classify.py --dataset sst2 --parser BobCatParser --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 7 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10 
```

python classify.py --dataset sst2 --parser BobCatParser --ansatz IQPAnsatz --model14type TketModel --trainer QuantumTrainer --epochs_train_model1 30 --no_of_training_data_points_to_use 20 --no_of_val_data_points_to_use 10 --max_tokens_per_sent 10

## if you want to debug the code use:

```bash
python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 100 --no_of_training_data_points_to_use 23 --no_of_val_data_points_to_use 1000 --expose_model1_val_during_model_initialization --do_debug
```
Note: you will have to explicitly attach debugger. Refer projectplan.md

Also note, during development its a healthy habit to always run pytest before the actual code in any branch. This can be achieved using 
`./runner.sh`

---
### Details of args:


`--expose_model1_val_during_model_initialization`: Pass this (dont have to do ==True) if you want the model1 to evaluate its performance in val/dev data while training itself. 

`--do_debug`:  pass this only if you want debugging done. i.e you are planning to attaching a debugging process from an IDE like visual studio code.


