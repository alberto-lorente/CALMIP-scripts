

from dotenv import load_dotenv
import os
from huggingface_hub import whoami, HfFolder
from datetime import date, datetime
import re
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import html
import re
import emoji
from pipetools import pipe
from typing import Dict, List
import numpy as np
import torch
import random
import json
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.nn import CrossEntropyLoss, KLDivLoss
from copy import deepcopy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from datasets import DatasetDict, concatenate_datasets


def log_hf():
    load_dotenv("env_vars.env")
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    HfFolder.save_token(hf_token)
    return print(whoami()["name"])

def save_results_csv(df, experiment_name, model_id, cl_technique, result_type="specific"):

    cl_technique_clean = cl_technique.replace(" + ", "__")
    id_ = model_id.replace("/", "-") + "_" + cl_technique_clean + "_" + str(date.today())
    id_clean = experiment_name + id_.replace(" ", "_").replace(":", "-").replace(".","-") + result_type + ".csv"
    df.to_csv(id_clean, index=False)
    return print("Saved in path: ", id_clean)

def clean_cl_name(cl_name):

    regex = r'<(?:[\w\.]+)?\.([\w]+) object at'
    matches =   re.findall(regex, cl_name)
    clean_string = " + ".join(matches)
    return clean_string

def clean_metric_name(metric_name):

    reg = r"\s([a-z_1]+)\s"
    match_ = re.search(reg, metric_name)
    clean_str = match_.group().strip()

    return clean_str

def load_data_hf(hate_cat_filter=None):
    
    """
    Load all the hf-format datasets. 
    Use hate_cat_filter to filter by the category of hate: [general, 
                                                            misoginy, 
                                                            xenophobia, 
                                                            racism, 
                                                            sexism]
    """
    ds_folder_path = "Datasets"
    # print(ds_folder_path)
    # print(ds_folder_path)
    list_dirs_clean = os.listdir(ds_folder_path)
    # print(list_dirs_clean)
    print(list_dirs_clean)
    datasets = {ds: load_from_disk(os.path.join(ds_folder_path, ds)) for ds in list_dirs_clean}
    return datasets


# def load_data_hf():
#     ds_list = ['davidson', 'evalita', 'founta_hateful_57k', 'hateval-immigrant', 'hateval-women', 'ibereval', 'waseem-racism', 'waseem-sexism']
#     ds_repos = ["alberto-lorente/" + ds for ds in ds_list]

#     datasets = []
#     for ds_repo in ds_repos:
#         dataset = load_dataset(ds_repo)
#         datasets.append(dataset)
#     hf_datasets = dict(zip(ds_list, datasets))

#     return hf_datasets


def format_ds_for_torch(dataset_list, validate_format=False):
    """
    Set the correct format for the hf dataset list.
    Should be done after tokenizing the data and before applying the DataLoaders.
    """
    for ds in dataset_list:
        ds.set_format(type='torch')


    if validate_format==True:
        example_dataset = dataset_list[0]["train"]
        print(example_dataset.format)

def load_data_splits(dataset, tokenizer, cols_to_remove, batch_size=16, shuffle=False):
    """
    Used within get_dict_from_dataset_list
    """

    try:
        dataset = dataset.remove_columns(cols_to_remove)
    except:
        print("Columns already removed")
    dataloader_train    = DataLoader(dataset["train"], batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
    dataloader_validate = DataLoader(dataset["validation"], batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)
    dataloader_test     = DataLoader(dataset["test"], batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator)

    return [dataloader_train, dataloader_validate, dataloader_test]

def get_dict_from_dataset_list(datasets_list, batch_size, shuffle, cols_to_remove, tokenizer, dataset_names):

    """
    Returns a dictionary of format {source of the data:[list of loaders]}.

    First item in the list is the train split.
    second item in the list is the validation split.
    Third item in the list is the test split.
    """

    data_loaders = {}

    for idx, ds in enumerate(datasets_list):
        # source = ds["train"]["source"][0]
        source = dataset_names[idx]
        data_loaders[source] = load_data_splits(dataset=ds, tokenizer=tokenizer, cols_to_remove=cols_to_remove, batch_size=batch_size, shuffle=shuffle) # source is the name of the dataset

    return data_loaders


def process_emojis(batch: Dict[str, List]):
    batch["text"] = [emoji.demojize(html.unescape(t)) for t in batch["text"]]
    return batch


def process_user_mentions(batch: Dict[str, List]):
    batch["text"] = [re.sub(r"@[a-zA-Z0-9_]+", "[USER]", t) for t in batch["text"]]
    return batch


def process_URLs(batch: Dict[str, List]):
    batch["text"] = [re.sub(r"https?://\S+", "[URL]", t) for t in batch["text"]]
    return batch


preprocessing_pipeline = (pipe | process_emojis | process_user_mentions | process_URLs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
        

def filter_sample_ds(hf_datasets, tokenizer, batch_size=16, ks_array=False, preprocessing_pipeline=preprocessing_pipeline):

    def tokenize_function(batch: Dict[str, List]):
        return tokenizer(
            text=batch["text"],
            truncation=True,
            padding=True)

    cols_to_remove = ['text', 'source', 'hs_domain', 'split']
    preprocessed_ds = [ds.map(preprocessing_pipeline, batched=True) for name, ds in hf_datasets.items()]
    tokenized_datasets = [ds.map(tokenize_function, batched=True, remove_columns=cols_to_remove) for ds in preprocessed_ds]
    dataset_names = list(hf_datasets.keys())
    # filter_datasets = [ds.remove_columns(cols_to_remove) for ds in tokenized_datasets]
    filter_datasets = [ds.rename_column("label", "labels") for ds in tokenized_datasets]
    if ks_array:
        for idx, ds in enumerate(filter_datasets):
            ds.shuffle(seed=42)
            # if idx < len(ks_array):
            for split in ds:
                if split == "train":
                    total_samples = len(ds[split])
                    if total_samples < ks_array[idx]:
                        ds[split] = ds[split].take(total_samples)
                    else:
                        ds[split] = ds[split].take(ks_array[idx])

    format_ds_for_torch(filter_datasets, validate_format=False)
    # print(filter_datasets[0])
    data_loaders = get_dict_from_dataset_list(datasets_list=filter_datasets,
                                            tokenizer=tokenizer,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            cols_to_remove=cols_to_remove,
                                            dataset_names=dataset_names)

    return data_loaders


def custom_eval(test_loader,
                model,
                metrics=[f1_score, precision_score, recall_score, roc_auc_score],
                verbose=False):

    model.model.eval()

    y_outputs = torch.Tensor([]).to(device)
    y_labels = torch.Tensor([]).to(device)
    probabilities =  torch.Tensor([]).to(device)

    with torch.no_grad():
        for batch in test_loader:

            batch_inputs = {k:v.to(device) for k,v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)
            outputs = model.model(**batch_inputs, labels=labels)

            logits = outputs.logits
            probas = torch.nn.functional.softmax(logits, dim=1)
            preds = torch.max(probas, dim=1)

            y_pred = preds.indices
            y_probas = preds.values

            if verbose:
                print(logits)
                print()
                print(probas)
                print()
                print(y_pred)
                print()
                print(y_probas)
                print()

            # loss = outputs.loss
            y_outputs = torch.cat((y_outputs, y_pred))
            y_labels = torch.cat((y_labels, labels))
            probabilities = torch.cat((probabilities, y_probas))

    y_labels = y_labels.cpu().numpy()
    y_outputs = y_outputs.cpu().numpy()
    result = {clean_metric_name(str(score)): float(score(y_labels, y_outputs, average='macro')) for score in metrics}
    results_hate_class = {"HATE_" + clean_metric_name(str(score)): float(score(y_labels, y_outputs, labels=[1], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}
    results_nohate_class = {"NoHATE_" + clean_metric_name(str(score)): float(score(y_labels, y_outputs, labels=[0], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}

    result.update(results_hate_class)
    result.update(results_nohate_class)
    result["predictions"] = [int(pred_label) for pred_label in y_outputs]
    result["labels"] = [int(actual_y) for actual_y in y_labels]
    result["probabilities"] = [float(proba) for proba in probabilities.cpu().numpy()]
    # print("EVAL DONE")
    # print()


    # print(result)
    # print(logits)
    # print(predictions)

    return result



def test(model,
        model_id:str,
        test_loader,
        type_experiment,
        cl_technique:str,
        time:int,
        current_training_dataset:str,
        current_testing_dataset:str,
        training_order:list,
        trainable_params,
        epochs,
        lr,
        batch_size,
        num_samples,
        exp_setup,
        metrics=[f1_score, precision_score, recall_score, roc_auc_score]):


    log_test =                                 {}
    log_test["model"] =                        model_id
    log_test["type_experiment"] =              type_experiment
    log_test["n_trainable_params"] =           int(trainable_params)
    log_test["cl_technique"]  =                cl_technique
    log_test["time"] =                         int(time)
    log_test["dataset"] =                      current_testing_dataset
    log_test["curr_train"] =                   current_training_dataset
    log_test["curr_train_hate_type"] =         exp_setup["general_ds_categories"][current_training_dataset]
    log_test["curr_train_mix_type"] =          exp_setup["mixtures_ds"][current_training_dataset]
    log_test["hate_type_test"] =               exp_setup["general_ds_categories"][current_testing_dataset]
    log_test["mixture_test"] =                 exp_setup["mixtures_ds"][current_testing_dataset]
    log_test["n_epochs_per_experience"] =      int(epochs)
    log_test["learning_rate"] =                float(lr)
    log_test["batch_size"] =                   int(batch_size)
    log_test["num_samples"] =                  num_samples

    if current_testing_dataset == current_training_dataset:
        log_test["shots"] = "IN TRAINING"

    elif current_testing_dataset not in training_order: # if we have already passed all the training indexes, that means that we are doing the zero shots, which i left at the end
        log_test["shots"] = "ZERO SHOT"

    elif training_order.index(current_training_dataset)  < training_order.index(current_training_dataset):
        log_test["shots"] = "PASSED TRAINING"

    else:
        log_test["shots"] = "ZERO SHOT"

    # print(current_testing_dataset)
    try:
        test_metrics = custom_eval(test_loader, model, metrics=metrics)
    except Exception as e:
        print("TESTING FAILED")
        print(e)
        test_metrics = {clean_metric_name(str(score)): "FAULTY INFERENCE" for score in metrics}

    log_test.update(test_metrics)

    return log_test

import pandas as pd


def train(  model,
            model_id:str,
            train_loader,
            val_loader,
            epochs:int,
            loss_f,
            optimizer,
            cut_batch:bool,
            trainable_params,
            lr,
            batch_size,
            num_samples,
            type_experiment:str,
            cl_technique:str,
            time:int,
            training_order:list,
            testing_order:list,
            test_loaders:list,
            exp_setup:dict
            ):

    """
    Main training function. Trains, evaluates and tests. The function test is called inside. Inside the function test, the function custom_eval is called.
    Params:
        model                               -> The model object. It is expected to be a ModelForSeqClassification
        model_id:str,                       -> The identifier of the model. Used for logging purposes.
        current_training_dataloader:list,   -> The list of the current training dataset where each index corresponds to train/validate/test
        epochs:int,                         -> The number of epochs to train. Remember this is take from the epochs array
        loss_f,
        optimizer,
        cut_batch:bool,                     -> Little boolean to cut the training short for trying out the loops purposes
        trainable_params,
        lr,
        batch_size,
        num_samples,                        -> number of samples that were take for the training dataloader
        type_experiment:str,
        cl_technique:str,
        time:int,                           -> the current time step of the training
        training_order:list,                -> a list with the names of the datasets we are training on in order
        testing_order:list,                 -> a list with the names of the datasets we are going to test at the end of each training loop. the first part are the training datasets and the last ones are zero shot tests.
        test_loaders:list,                  -> a testing data loader that contains the test split of the datasets
        exp_setup:dict                      -> a dictionary with information such as which mixture/hate category each dataset belongs at etc.


    """

    train_losses = []
    validation_losses = []
    # validation_metrics = []

    for epoch in range(epochs):

        print(f"\tEpoch: {epoch + 1}")
        print()
        print("Training")
        model.model.train()
        total_loss = 0

        i = 1

        for batch in train_loader:

            optimizer.zero_grad()
            # print(batch)
            # print(len(batch))
            # print(type(batch))
            # print(batch.keys())
            print(f"___Batch {i} out of {len(train_loader)}___")
            # print(batch)

            # input_ids = batch["input_ids"].to(device)
            # # token_type_ids = batch["token_type_ids"].to(device)
            # attention_mask = batch["attention_mask"].to(device)
            # labels = batch["labels"].to(device)

            batch_inputs = {k:v.to(device) for k,v in batch.items() if k != "labels"}
            labels = batch["labels"].to(device)

            outputs = model.model(**batch_inputs, labels=labels)

            logits = outputs.logits
            loss = loss_f(logits, labels)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            train_losses.append(float(total_loss))

            i += 1

            if cut_batch:
                if i > 3:
                    break

        print(f'Training Loss: {total_loss/len(train_loader)}\n')
        print("_____________Validating___________")

        model.model.eval()
        val_loss = 0

        y_outputs = torch.Tensor([]).to(device)
        y_labels = torch.Tensor([]).to(device)

        with torch.no_grad():
            for batch in val_loader:

                batch_inputs = {k:v.to(device) for k,v in batch.items() if k != "labels"}
                labels = batch["labels"].to(device)

                outputs = model.model(**batch_inputs, labels=labels)

                logits = outputs.logits
                probas = torch.nn.functional.softmax(logits)
                y_pred = torch.max(probas, dim=1).indices

                loss = loss_f(logits, labels)
                val_loss += loss.item()

                # not used currently
                y_outputs = torch.cat((y_outputs, y_pred))
                y_labels = torch.cat((y_labels, labels))

        validation_losses.append(float(val_loss))

        print(f'Validation Loss: {val_loss/len(val_loader)}\n')
        print("---Epoch DONE---")

    print("-------Training Completed-------")
    print()

    #### checking some types so that they can be json-ified
    # print(type(train_losses))
    # print(type(validation_losses))

    train_val_log = {"model_id": model_id,
                    "current_ds_training": "all",
                    "epochs": int(epochs),
                    "time": int(time),
                    "num_samples": num_samples,
                    "training_details": list(train_losses),
                    "validation_details": list(validation_losses),
                    "type_experiment": type_experiment,
                    "cl_technique": cl_technique,
                    "train_order": " -> ".join(training_order)
                    # "validation_metrics":validation_metrics   ## COULD COMPUTE THIS WITH CUSTOM EVAL
                    }

    print("Testing")

    tests = [] # this is going to be a list of the log dictionaries
    for idx, current_testing_loader in enumerate(test_loaders):
        # print(testing_order[idx])
        # print(current_testing_loader)
        # print(type(current_testing_loader))

        # i wont need to compute cl stuff so i can pass model.model as the model instead of modifying the custome eval function
        log_test = test(model=model,
                        model_id=model_id,
                        test_loader=current_testing_loader, # laoder passed to the custm eval
                        type_experiment=type_experiment,
                        cl_technique=cl_technique,
                        time=time,
                        current_training_dataset=training_order[time], # name of the current ds train
                        current_testing_dataset=testing_order[idx],   # name of the current ds test
                        training_order=training_order,
                        trainable_params=trainable_params,
                        epochs=epochs,
                        lr=lr,
                        batch_size=batch_size,
                        num_samples=num_samples,
                        exp_setup=exp_setup)

        # print(log_test)

        tests.append(log_test)
    # print(test)
    # print(type(test))


    return model, train_val_log, tests


class CLTechniques:
    """Container for all continual learning techniques"""

    def __init__(self, model, device, technique="none",
                ewc_lambda=1000,
                mem_size=100,
                lwf_lambda=1,
                temperature=2,
                mas_lambda=1000):

        self.model = model
        self.device = device
        self.technique = technique.lower()

        # Initialize selected technique
        if self.technique == "ewc":
            self._init_ewc(ewc_lambda)
        elif self.technique == "agem":
            self._init_agem(mem_size)
        elif self.technique == "lwf":
            self._init_lwf(lwf_lambda, temperature)
        elif self.technique == "mas":
            self._init_mas(mas_lambda)

    def _init_ewc(self, ewc_lambda):
        """Elastic Weight Consolidation"""
        self.ewc_lambda = ewc_lambda
        self.params = {n: p.clone().detach()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad}
        self.fisher = {n: torch.zeros_like(p)
                    for n, p in self.model.named_parameters()
                    if p.requires_grad}

    def _init_agem(self, mem_size):
        """Average Gradient Episodic Memory"""
        self.mem_size = mem_size
        self.memory = []

    def _init_lwf(self, lwf_lambda, temperature):
        """Learning Without Forgetting"""
        self.lwf_lambda = lwf_lambda
        self.temperature = temperature
        self.old_model = None

    def _init_mas(self, mas_lambda):
        """Memory Aware Synapses"""
        self.mas_lambda = mas_lambda
        self.importance = {n: torch.zeros_like(p)
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}
        self.old_params = deepcopy(self.importance)

    def compute_regularization(self, inputs=None):
        """Compute CL regularization term"""
        if self.technique == "ewc":
            penalty = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    penalty += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
            return self.ewc_lambda * penalty

        elif self.technique == "lwf" and self.old_model:
            with torch.no_grad():
                logits = inputs['logits']
                actual_inputs = {k:v for k,v in inputs.items() if k != "logits"}
                old_outputs = self.old_model(**actual_inputs)
            return self.lwf_lambda * KLDivLoss(reduction='batchmean')(
                torch.log_softmax(logits/self.temperature, dim=1),
                torch.softmax(old_outputs.logits/self.temperature, dim=1)
            ) * (self.temperature ** 2)

        elif self.technique == "mas":
            penalty = 0
            for n, p in self.model.named_parameters():
                if p.requires_grad:
                    penalty += (self.importance[n] * (p - self.old_params[n]).pow(2)).sum()
            return self.mas_lambda * penalty

        return 0

    def pre_backward(self, inputs=None):
        """Operations before backward pass"""
        if self.technique == "agem" and self.memory:
            # Store current gradient
            self.model.zero_grad()
            for inputs_mem, labels_mem in self.memory:
                outputs = self.model(**inputs_mem, labels=labels_mem)
                outputs.loss.backward()
            self.ref_grad = [p.grad.clone() for p in self.model.parameters()]
            self.model.zero_grad()

    def post_backward(self):
        """Operations after backward pass"""
        if self.technique == "agem" and hasattr(self, 'ref_grad'):
            # Project gradients
            dot_product = sum(torch.sum(p.grad * g_ref)
                        for p, g_ref in zip(self.model.parameters(), self.ref_grad))
            ref_norm = sum(torch.sum(g_ref * g_ref) for g_ref in self.ref_grad)

            if dot_product < 0:  # Negative interference
                scale = dot_product / (ref_norm + 1e-8)
                for p, g_ref in zip(self.model.parameters(), self.ref_grad):
                    if p.grad is not None:
                        p.grad -= scale * g_ref

    def post_task_update(self, dataloader=None):
        """Update after each task"""
        if self.technique == "ewc":
            # Compute Fisher information
            self.model.eval()
            for batch in dataloader:
                self.model.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}
                labels = batch['labels'].to(self.device)

                outputs = self.model(**inputs, labels=labels)
                outputs.loss.backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.fisher[n] += p.grad.pow(2) / len(dataloader)

            # Update stored parameters
            self.params = {n: p.clone().detach()
                        for n, p in self.model.named_parameters()
                        if p.requires_grad}

        elif self.technique == "agem":
            # Update memory buffer
            self.memory = []
            for batch in dataloader:
                inputs = {k: v.to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}
                labels = batch['labels'].to(self.device)
                self.memory.append((inputs, labels))
                if len(self.memory) >= self.mem_size:
                    break

        elif self.technique == "lwf":
            # Save model snapshot
            self.old_model = deepcopy(self.model)
            self.old_model.eval()

        elif self.technique == "mas":
            # Update importance weights
            self.model.eval()
            for batch in dataloader:
                self.model.zero_grad()
                inputs = {k: v.to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}

                outputs = self.model(**inputs)
                torch.norm(outputs.logits, p=2, dim=1).mean().backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.importance[n] += p.grad.abs() / len(dataloader)

            # Update stored parameters
            self.old_params = {n: p.clone().detach()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad}

class AutoContinualLearner:
    def __init__(self, model_name, num_labels=2, device="cuda"):
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(self.device)
        self.cl = None

    def init_cl(self, technique, **kwargs):
        """Initialize continual learning technique"""
        self.cl = CLTechniques(self.model, self.device, technique, **kwargs)


with open("experiment_set_up_week_3.json", "r") as f:
    exp_setup = json.load(f)

seed = 42
set_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


load_dotenv("env_vars.env")
# hf_token = os.environ.get("HF_ACCESS_TOKEN")
# HfFolder.save_token(hf_token)
# print(whoami()["name"])


model_ids =     [
                "FacebookAI/roberta-base",
                "Xuhui/ToxDect-roberta-large",
                "diptanu/fBERT",
                "GroNLP/hateBERT"
                ]

type_experiment = "big_train_baseline"

loss_f = CrossEntropyLoss()
lr = 1e-5
batch_size = 64
cut_batch = False
epochs = 8
experiments_results = []


torch.cuda.ipc_collect()
torch.cuda.empty_cache()

hf_datasets = load_data_hf()

all_ds_train = [ds["train"] for ds in hf_datasets.values()]
all_ds_val = [ds["validation"] for ds in hf_datasets.values()]
all_ds_test = [ds["test"] for ds in hf_datasets.values()]

all_train = concatenate_datasets(all_ds_train)
all_val = concatenate_datasets(all_ds_val)

merged_data = DatasetDict({
    "train": all_train,
    "validation":all_val
    })


for model_id in model_ids:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoContinualLearner(model_id, num_labels=2, device=device)
    optimizer = AdamW(model.model.parameters(), lr=lr)
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad) 

    if model_id == "Xuhui/ToxDect-roberta-large":
        batch_size = 24
        lr = 5e-05

    def tokenize_function(batch: Dict[str, List]):
            return tokenizer(
                text=batch["text"],
                truncation=True,
                padding=True)


    cols_to_remove = ['text', 'source', 'hs_domain', 'split']
    preprocessed_ds = [ds.map(preprocessing_pipeline, batched=True) for name, ds in merged_data.items()]
    # doing the for above splits the ds in two, i could do it without it
    tokenized_datasets = [ds.map(tokenize_function, batched=True, remove_columns=cols_to_remove) for ds in preprocessed_ds]
    dataset_names = list(merged_data.keys())
    filter_datasets = [ds.rename_column("label", "labels") for ds in tokenized_datasets]
    format_ds_for_torch(filter_datasets, validate_format=False)


    tests = dict(zip(hf_datasets.keys(), all_ds_test))
    preprocessed_ds_test = [ds.map(preprocessing_pipeline, batched=True) for name, ds in tests.items()]
    tokenized_datasets_test = [ds.map(tokenize_function, batched=True, remove_columns=cols_to_remove) for ds in preprocessed_ds_test]
    filter_datasets_test = [ds.rename_column("label", "labels") for ds in tokenized_datasets_test]
    format_ds_for_torch(filter_datasets_test, validate_format=False)


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    dataloader_train    = DataLoader(filter_datasets[0], batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    dataloader_validate = DataLoader(filter_datasets[1], batch_size=batch_size, shuffle=True, collate_fn=data_collator)

    test_loaders = []
    for ds in filter_datasets_test:
        test_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
        test_loaders.append(test_loader)

    results, model =  train( model=model,
                            model_id=model_id,
                            train_loader=dataloader_train,
                            val_loader=dataloader_validate,
                            epochs=epochs,
                            loss_f=loss_f,
                            optimizer=optimizer,
                            cut_batch=cut_batch,
                            trainable_params=trainable_params,
                            lr=lr,
                            batch_size=batch_size,
                            num_samples="all",
                            type_experiment=type_experiment,
                            cl_technique="big baseline",
                            time=0,
                            training_order=["all_at_once"],
                            testing_order=list(hf_datasets.keys()),
                            test_loaders=test_loaders,
                            exp_setup=exp_setup
                            )

    experiment_json_name = "_".join([type_experiment, model_id.replace("/", "-"), "big_train_ft"]) + ".json"

    try:
        with open(experiment_json_name, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print("Result couldn't be saved.")
        print(e)

    results["json_name"] = experiment_json_name
    experiments_results.append(results)

print("----------------------------------------")
print(experiments_results)