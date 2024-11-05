# QNLP
Requirements:
- minimum 16 GB ram to load the fasttext model and lambeq models
## steps to run this code
- download data files  (e.g. spanish_test.txt)from [this](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)  repo to the same location where this code is
- `conda create --name qnlp_temp7 python==3.11.10`
- `conda activate qnlp_temp7` 
- `./run_me_first.sh`

Note: the last line of ./run_me_first.sh will try to download a 5GB file. alternately you can download spanish fasttext embeddings: go to [this](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file#fasttext-embeddings-from-suc) url and download manually the .bin file for spanish unannotated corpora to the same location where this code is.

- `python OOV.py`
