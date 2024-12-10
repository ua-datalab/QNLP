# QNLP
Requirements:
- minimum 16 GB ram to load the fasttext model and lambeq models
- steps to run this code
- download data files  (e.g. spanish_test.txt)from [this](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)  repo to the same location where this code is
- `conda create --name qnlp_temp7 python==3.11.10`
- `conda activate qnlp_temp7` 
- `./run_me_first.sh`

Note: the last line of ./run_me_first.sh will try to download a 5GB file. alternately you can download spanish fasttext embeddings: go to [this](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file#fasttext-embeddings-from-suc) url and download manually the .bin file for spanish unannotated corpora to the same location where this code is.

## command to run this code as of dec 10th 2024

- `pytest` (ensure nothing fails)
- ` python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 100 --no_of_training_data_points_to_use 23 --no_of_val_data_points_to_use 1000 --expose_model1_val_during_model_initialization`

## if you want to debug the code use:

- ` python classify.py --dataset sst2 --parser Spider --ansatz SpiderAnsatz --model14type PytorchModel --trainer PytorchTrainer --epochs_train_model1 100 --no_of_training_data_points_to_use 23 --no_of_val_data_points_to_use 1000 --expose_model1_val_during_model_initialization --do_debug`

Note: you will have to explicitly attach debugger. Refer projectplan.md

---
### Details of args:


`--expose_model1_val_during_model_initialization`: Pass this (dont have to do ==True) if you want the model1 to evaluate its performance in val/dev data while training itself. 

`--do_debug`:  pass this only if you want debugging done. i.e you are planning to attaching a debugging process from an IDE like visual studio code.


