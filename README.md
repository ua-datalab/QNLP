# QNLP
Requirements:
- minimum 16 GB ram to load the fasttext model and lambeq models
- Python 3.11.10
- pip 24.3.1
steps to run this code
- download data files  (e.g. spanish_test.txt)from [this](https://github.com/bkeej/usp_qnlp/tree/main/qnlp-data)  repo to the same location where this code is
- conda create --name qnlp_temp7 python==3.11.10
- conda activate qnlp_temp7 
- ./run_me_first.sh
- `python OOV.py`

Note
    - alternately go to [this](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file#fasttext-embeddings-from-suc) url and download manually the .bin file for spanish unannotated corpora to the same location where this code is.
- `python v6_qnlp_uspantekan_experiments.py`
