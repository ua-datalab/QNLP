python3 -m pip install --upgrade pip
pip install scipy 
pip install numpy   
pip install spacy     
pip install tensorflow
pip install pytket
pip install lambeq[extras] #if you are using zsh on mac osx use this instead: pip install 'lambeq[extras]'
pip install qiskit-ibm-runtime
pip install fasttext
pip install keras-tuner --upgrade
pip install pytest
pip install datasets
pip install huggingface-hub
pip install datasets
pip install lightning
pip install torchmetrics
pip install wandb
export TOKENIZERS_PARALLELISM=True
python3 -m spacy download es_core_news_sm
python3 -m spacy download en_core_web_sm
conda install -y anaconda::wget
pip install wget
#run this only once if you dont have the models downloaded 
#wget -c "https://zenodo.org/record/3234051/files/embeddings-l-model.bin?download=1" -O ./embeddings-l-model.bin
#wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz"
#gunzip cc.en.300.bin.gz 