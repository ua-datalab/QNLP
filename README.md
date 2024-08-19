# QNLP
Requirements:
- minimum 16 GB ram to load the fasttext model and lambeq models

steps to run this code
- pip install -r requirements.txt
# to install fasttext
- !wget -c https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1 -O ./embeddings-l-model.bin
- alternately go to [this](https://github.com/dccuchile/spanish-word-embeddings?tab=readme-ov-file#fasttext-embeddings-from-suc) url and download manually the .bin file for spanish unannotated corpora
- python v6_qnlp_uspantekan_experiments.py
