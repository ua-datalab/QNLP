This is the code where we are loading uspantekan data into classical LLM base models
# Steps
- pip install -r requirments.txt

- install pytorch based on your os version/system config from the pytorch home page[page](https://pytorch.org/)
- (e.g: `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- create a wandb [account](https://wandb.ai/), and copy the online api key.
- `wandb login`
- `wandb online`
- The code can then be run as: `python fine_tune_fundamental_models_classification.py- --train_file [] --dev_file  [] --model_type  [] --epochs  [] --disable_wandb  []`

	e.g.: `python fine_tune_fundamental_models_classification.py --train_file ./data/Food_IT_classification/mc_train_data.txt --dev_file ./data/Food_IT_classification/mc_dev_data.txt --model_type roberta --epochs 100`
or 

	 `python fine_tune_fundamental_models_classification.py --train_file ./data/uspantekan/uspantekan_train.txt --dev_file ./data/uspantekan/uspantekan_dev.txt --model_type roberta --epochs 100 enable_wandb False`

- if you enabled wandb, wandb will ask for onlineapi key: copy from authorization [page](https://wandb.ai/authorize) and paste it on terminal
