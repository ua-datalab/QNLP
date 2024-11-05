python3 -m pip install --upgrade pip
pip install scipy 
pip install numpy   
pip install spacy     
pip install tensorflow
pip install lambeq
pip install fasttext
python3 -m spacy download es_core_news_sm
python3 -m spacy download en_core_web_sm
pip install wget
wget -c "https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1" -O ./embeddings-l-model.bin