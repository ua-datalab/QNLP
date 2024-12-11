# QNLP
**Requirements:**
- Minimum 16 GB ram to load the fasttext model and lambeq models

**Steps for setup:**
- download data files  (e.g. spanish_test.txt)from [this](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)  repo to the same location where this code is
- `conda create --name qnlp_temp7 python==3.11.10`
- `conda activate qnlp_temp7` 
- `./run_me_first.sh`


Once all requirements and the environment has been set up, you can begin training the QNLP model and testing it.

**Steps for training and testing the QNLP Model:**
- 

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


