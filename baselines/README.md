This is the code where we are loading uspantekan data into classical LLM base models
# Steps
- pip install -r requirments.txt

- install pytorch based on your os version(e.g: pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu)
- create a wandb account, and copy the online api key.
- wandb login
- wandb online
- The code can then be run as: python fine_tune_fundamental_models_classification.py- --train_file [] --dev_file  [] --model_type  [] --epochs  [] --disable_wandb  []

	e.g.: `python fine_tune_fundamental_models_classification.py --train_file ./data/Food_IT_classification/mc_train_data.txt --dev_file ./data/Food_IT_classification/mc_dev_data.txt --model_type roberta --epochs 100`

- wandb will ask for onlineapi key: paste it on terminal
