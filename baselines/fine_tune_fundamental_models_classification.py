import argparse
import pandas as pd
import wandb
import torch
import os
import subprocess
import numpy as np
from datetime import datetime
from tqdm import tqdm


from transformers import BertTokenizer, BertModel
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from sklearn import metrics

DISABLE_WANDB = False
MAX_LEN = 500
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
LEARNING_RATE = 1e-7
DROP_OUT_RATE = 0.3
PATIENCE = 20
NO_OF_CLASSES = 2
SAVED_MODEL_NAME="best_model_for_"+datetime.now().isoformat(timespec="minutes")+".pth"
SAVED_MODEL_PATH=os.path.join("./output/",SAVED_MODEL_NAME)

global_f1_validation = 0
global_validation_loss = 999999
precision_global = 0
dict_all_index_labels = {1 : "food", 0 : "IT"}



def replace_true_false(predictions):
    boolean_predictions=[]
    for class1,class2 in predictions:
        out_class1 = 0
        out_class2 = 0

        if class1==True:
            out_class1 = 1
        if class2==True:
            out_class2 = 1
        boolean_predictions.append([out_class1,out_class2])
    return boolean_predictions


def validation(epoch,model):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(validation_loader, 0),desc="validation loader"):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            validation_loss = loss_fn(outputs, targets)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets,validation_loss

#am reusing some old code- this code needs all classes in 1x2 form. i.e if the class label is 1 it is [1,0] and [0,1] for
def get_two_classes_mode(data):
    all_classes = []
    for val in data:
        if val ==1:
            all_classes.append([1,0])
        else:
            all_classes.append([0 ,1])
    return all_classes

def train(epoch,NO_OF_CLASSES,model):
    model.train()
    print(f"value of learning rate is {LEARNING_RATE} and the model used is {model_type}")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    for _, data in enumerate(training_loader, 0):
        ids = data['ids'].to(device, dtype=torch.long)
        mask = data['mask'].to(device, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
        targets = data['targets'].to(device, dtype=torch.float)
        outputs = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

class ModelWithNN(torch.nn.Module):
    def __init__(self,NO_OF_CLASSES,base_model):
        super(ModelWithNN, self).__init__()
        self.l1 = base_model
        self.l2 = torch.nn.Dropout(DROP_OUT_RATE)
        self.l3 = torch.nn.Linear(LAST_LAYER_INPUT_SIZE, 64)
        self.l4 = torch.nn.LayerNorm(64)
        self.l5 = torch.nn.Dropout(DROP_OUT_RATE)
        self.l6 = torch.nn.Linear(64,NO_OF_CLASSES)

    def forward(self, ids, mask, token_type_ids):
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output = self.l2(output_1['pooler_output'])
        output = self.l3(output)
        output = self.l4(output)
        output = self.l5(output)
        output = self.l6(output)

        return output

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.text
        self.targets= get_two_classes_mode(dataframe.labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


def print_return_per_label_metrics(gold_labels_boolean_tuples, pred_labels_boolean_tuples):

    # to calculate per label accuracy- increase counter for each true positive
    assert len(gold_labels_boolean_tuples) == len(pred_labels_boolean_tuples)
    label_counter_accuracy = {}
    label_counter_overall = {}

    avg_f1=0
    avg_precision=0
    avg_recall=0
    sum_f1=0
    sum_accuracy=0
    sum_precision=0
    sum_recall=0
    # have a dictionary inside a dictionary to keep track of TP,FN etc for each label
    # e.g.,{"words_location_TP:24}
    true_positive_true_negative_etc_per_label = {}

    #initializing the dictionaries with zeores
    for x in range(len(gold_labels_boolean_tuples[0])):
        label_string = dict_all_index_labels[x]
        label_counter_accuracy[label_string] = 0
        label_tp = label_string + "_TP"
        true_positive_true_negative_etc_per_label[label_tp] = 0
        label_tn = label_string + "_TN"
        true_positive_true_negative_etc_per_label[label_tn] = 0
        label_fp = label_string + "_FP"
        true_positive_true_negative_etc_per_label[label_fp] = 0
        label_fn = label_string + "_FN"
        true_positive_true_negative_etc_per_label[label_fn] = 0

    all_labels_string_value = []
    for gold_truple, pred_truple in zip(gold_labels_boolean_tuples, pred_labels_boolean_tuples):
        assert len(gold_truple)==len(pred_truple)
        for index,value in enumerate(gold_truple):
            #to calculate overall count of labels... should be same as len(gold)
            label_string=dict_all_index_labels[index]
            if label_string in label_counter_overall:
                current_count = label_counter_overall[label_string]
                label_counter_overall[label_string] = current_count + 1
            else:
                label_counter_overall[label_string] = 1

            if gold_truple[index] == pred_truple[index]:
                # calculate accuracy as long as both gold and pred match- irrespective of TP, FP etc
                if label_string in label_counter_accuracy:
                    current_count = label_counter_accuracy[label_string]
                    label_counter_accuracy[label_string] = current_count + 1
                else:
                    label_counter_accuracy[label_string] = 1

                #finding true positive
                if int(gold_truple[index]) ==1:
                    current_label_tp=label_string+"_TP"
                    if current_label_tp in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_tp]
                        true_positive_true_negative_etc_per_label[current_label_tp]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_tp]=1

                #true negative
                if int(gold_truple[index]) ==0:
                    current_label_tn = label_string + "_TN"
                    if current_label_tn in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_tn]
                        true_positive_true_negative_etc_per_label[current_label_tn]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_tn]=1

            #false negative
            else:
                if int(gold_truple[index]) == 1 and int(pred_truple[index])==0:
                    current_label_fn = label_string + "_FN"
                    if current_label_fn in true_positive_true_negative_etc_per_label:
                        old_value=true_positive_true_negative_etc_per_label[current_label_fn]
                        true_positive_true_negative_etc_per_label[current_label_fn]=old_value+1
                    else:
                        true_positive_true_negative_etc_per_label[current_label_fn]=1
                else:
                    if int(gold_truple[index]) == 0 and int(pred_truple[index]) == 1:
                        current_label_fp = label_string + "_FP"
                        if current_label_fp in true_positive_true_negative_etc_per_label:
                            old_value = true_positive_true_negative_etc_per_label[current_label_fp]
                            true_positive_true_negative_etc_per_label[current_label_fp] = old_value + 1
                        else:
                            true_positive_true_negative_etc_per_label[label_fp] = 1

    for label, v in label_counter_accuracy.items():
        total = label_counter_overall[label]

        print(f"------\nFor the  label {label}:")
        accuracy=v / total

        tp=true_positive_true_negative_etc_per_label[label+"_TP"]
        tn = true_positive_true_negative_etc_per_label[label + "_TN"]
        fp = true_positive_true_negative_etc_per_label[label + "_FP"]
        fn=true_positive_true_negative_etc_per_label[label+"_FN"]

        print(f"accuracy ={accuracy}")
        print(f"true positive:{tp}")
        print(f"true negative:{tn}")
        print(f"false positive:{fp}")
        print(f"false negative :{fn}")

        if (tp+fp)==0:
            precision=0
        else:
            precision =tp / (tp + fp)

        if (tp + fn) == 0:
            recall=0
        else:
            recall =tp / (tp + fn)
        if (precision+recall)==0:
            F1=0
        else:
            F1 = 2 * (precision * recall) / (precision + recall)

        print(f"precision={precision}")
        print(f"recall={recall}")
        print(f"F1={F1}")

        precision_label_name="precision"+"_"+label
        recall_label_name = "recall" + "_" + label
        f1_label_name = "f1" + "_" + label
        accuracy_label_name = "accuracy" + "_" + label
        wandb.log({precision_label_name: precision,'epoch': epoch})
        wandb.log({recall_label_name: recall, 'epoch': epoch})
        wandb.log({f1_label_name: F1, 'epoch': epoch})
        wandb.log({accuracy_label_name: accuracy, 'epoch': epoch})

        sum_accuracy = sum_accuracy+accuracy
        sum_f1=sum_f1+F1
        sum_precision = sum_precision + precision
        sum_recall = sum_recall + recall

    avg_f1=sum_f1/len(label_counter_accuracy.items())
    avg_accuracy=sum_accuracy/len(label_counter_accuracy.items())
    wandb.log({'average_precision': sum_precision/len(label_counter_accuracy.items()), 'epoch': epoch})
    wandb.log({'average_recall': sum_recall/len(label_counter_accuracy.items()), 'epoch': epoch})
    wandb.log({'average_accuracy': avg_accuracy, 'epoch': epoch})

    return avg_f1





argumentParser=argparse.ArgumentParser()
argumentParser.add_argument("--train_file",required=True)
argumentParser.add_argument("--dev_file",required=True)
argumentParser.add_argument("--model_type",required=True,help="#[bert-base-uncased, roberta-large]")
argumentParser.add_argument("--epochs",required=True)
argumentParser.add_argument("--disable_wandb",required=True)
args=argumentParser.parse_args()

train_file = args.train_file
dev_file = args.dev_file
model_type=  "roberta-large"
EPOCHS =  int(args.epochs)
DISABLE_WANDB = bool(args.disable_wandb)


device = 'cuda' if cuda.is_available() else 'cpu'
if (DISABLE_WANDB):
    subprocess.Popen('wandb offline', shell=True)


wandb.init(project="fine_tune_fundamental_model_food_it")

if "roberta" in model_type:
    tokenizer = RobertaTokenizer.from_pretrained(model_type)
    MODEL= RobertaModel.from_pretrained(model_type)
    LAST_LAYER_INPUT_SIZE = 1024  # will be 768 for roberta base, bert and 1024 for roberta large
else:
    if "bert" in model_type:
        tokenizer = BertTokenizer.from_pretrained(model_type)
        MODEL= BertModel.from_pretrained(model_type)
        LAST_LAYER_INPUT_SIZE = 768


train_dataset = pd.read_csv(train_file, sep="\t", on_bad_lines='skip',names=["labels","text"])
validation_dataset = pd.read_csv(dev_file, sep="\t", on_bad_lines='skip',names=["labels","text"])

print(f"total number of train datapoints={len(train_dataset)}")
print(f"total number of validation_dataset datapoints={len(validation_dataset)}")

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
validation_set = CustomDataset(validation_dataset, tokenizer, MAX_LEN)

train_params = {'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

validation_params = {'batch_size': VALID_BATCH_SIZE,
             'shuffle': True,
             'num_workers': 0
             }

training_loader = DataLoader(training_set, **train_params)
validation_loader = DataLoader(validation_set, **validation_params)

print(f"************found that the device is {device}\n")
patience_counter=0
overall_accuracy=0
accuracy_validation=0
for epoch in tqdm(range(EPOCHS),desc="epochs",total=EPOCHS):
    wandb.log({'patience_counter': patience_counter, 'epoch': epoch})
    if(patience_counter>PATIENCE):
        print(f"found that validation loss is not improving after hitting patience of {PATIENCE}. Quitting")
        break


    model = ModelWithNN(NO_OF_CLASSES, MODEL)
    model.to(device)
    train_loss=train(epoch, NO_OF_CLASSES, model)
    predictions_validation, gold_validation ,validation_loss = validation(epoch,model)


    if validation_loss<global_validation_loss:
        global_validation_loss=validation_loss
    else:
        patience_counter+=1



    wandb.log({'train_loss': train_loss,'epoch': epoch})
    wandb.log({'validation_loss': validation_loss,'epoch': epoch})
    predictions_validation = np.array(predictions_validation) >= 0.5
    predictions_validation = replace_true_false(predictions_validation)
    accuracy_validation_scikit_version = metrics.accuracy_score(gold_validation, predictions_validation)
    overall_accuracy=overall_accuracy+accuracy_validation
    # avg_accuracy_scikit_version=overall_accuracy/(epoch+1)
    # outputs_float = predictions_validation.astype(float)
    # accuracy_manual= print_return_per_label_metrics(predictions_validation)
    avg_f1_validation_this_epoch = print_return_per_label_metrics(gold_validation, predictions_validation)
    #
    # avg_f1_validation_this_epoch =avg_f1_validation_this_epoch
    # #rewrite the best model every time the f1 score improves
    # if avg_f1_validation_this_epoch > global_f1_validation:
    #     global_f1_validation = avg_f1_validation_this_epoch
    #     torch.save(model.state_dict(), SAVED_MODEL_PATH)
    #

    # print(f"avg F1:{avg_f1_validation_this_epoch}\n")
    print(f"accuracy in epoch {epoch} is {accuracy_validation_scikit_version}")
    wandb.log({'accuracy_validation_scikit_version': accuracy_validation_scikit_version})
    print(f"end of epoch {epoch}")
    print(f"---------------------------")