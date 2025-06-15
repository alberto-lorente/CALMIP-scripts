import os
import json
from dotenv import load_dotenv
from copy import deepcopy

from typing import Dict, List

import itertools
import random

from datetime import date, datetime
import emoji
import re
import html

import pandas as pd
import numpy as np
from pipetools import pipe

from huggingface_hub import whoami, HfFolder
from datasets import load_dataset, load_from_disk
from transformers import DataCollatorWithPadding

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, KLDivLoss
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

#  UserWarning: CUDA initialization: The NVIDIA driver on your system is too old (found version 9010). 
# Please update your GPU driver by downloading and installing a new version from the URL: http://www.nvidia.com/Download/index.aspx 
# Alternatively, go to: https://pytorch.org to install a PyTorch version that has been compiled with your version of the CUDA driver. (Triggered internally at  ../c10/cuda/CUDAFunctions.cpp:112.)



# def log_hf():
#     load_dotenv("env_vars.env")
#     hf_token = os.environ.get("HF_ACCESS_TOKEN")
#     HfFolder.save_token(hf_token)
#     return print(whoami()["name"])

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

def downsample_hf_dataset_list(hf_datasets, total_n_samples=500):

    train_n = int(total_n_samples*0.8)
    val_n = int(total_n_samples*0.1)
    test_n = int(total_n_samples*0.1)

    samples_d = {"train": train_n,
                "validation": val_n,
                "test": test_n}
    for ds, data in hf_datasets.items():
        for split in data:
            data[split] = data[split].take(samples_d[split]) # take first n

    return hf_datasets

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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

preprocessing_pipeline = (pipe | process_emojis | process_user_mentions | process_URLs)
def filter_sample_ds(hf_datasets, tokenizer, order, batch_size=16, ks_array=False, preprocessing_pipeline=preprocessing_pipeline):

    def tokenize_function(batch: Dict[str, List]):
        return tokenizer(
            text=batch["text"],
            truncation=True,
            padding=True)

    cols_to_remove = ['text', 'source', 'hs_domain', 'split']
    preprocessed_ds = [ds.map(preprocessing_pipeline, batched=True) for name, ds in hf_datasets.items() if name in order]
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    def __init__(self, model_name, num_labels=2, device=device):
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
        hyper_param_str,
        metrics=[f1_score, precision_score, recall_score, roc_auc_score],
        ):


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
    log_test["hyper_param"] =                  hyper_param_str

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

def train(  model,
            model_id:str,
            current_training_dataloader:list,
            current_dataset_name,
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
            exp_setup:dict,
            hyper_param_str:str
            ):

    """
    Main training function. Trains, evaluates and tests. The function test is called inside. Inside the function test, the function custom_eval is called.
    Params:
        model                               -> The model object. It is expected to be a ModelForSeqClassification
        model_id:str,                       -> The identifier of the model. Used for logging purposes.
        current_training_dataloader:list,   -> The list of the current training dataset where each index corresponds to train/validate/test
        current_dataset_name,               -> The name of the current training dataset
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

    train_loader = current_training_dataloader[0]
    val_loader = current_training_dataloader[1]
    # test_loader = current_training_dataloader[2] # not used

    # overrides the optimizer in case the params change
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=lr)

    for epoch in range(epochs):

        if cl_technique == "zero_shot":
            break # uncomment for zero-shot perf

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
            # print(f"___Batch {i} out of {len(train_loader)}___")
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

            # check type of model
            if model.cl:
                batch_inputs['logits'] = outputs.logits  # needed for LwF
                loss += model.cl.compute_regularization(batch_inputs)
                model.cl.pre_backward(batch_inputs)


            loss.backward()

            # needed agem (A-GEM)
            if model.cl:
                model.cl.post_backward()


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

                # input_ids = batch["input_ids"].to(device)
                # # token_type_ids = batch["token_type_ids"].to(device)
                # attention_mask = batch["attention_mask"].to(device)
                # labels = batch["labels"].to(device)
                # outputs = model.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

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

        # print(f'Validation Loss: {val_loss/len(val_loader)}\n')
        print("---Epoch DONE---")

    print("-------Training Completed-------")
    print()

            # Post-task updates
    if model.cl:
        model.cl.post_task_update(train_loader)

    print("-----------POST TRAINING CL UPDATES COMPLETED---------")


    #### checking some types so that they can be json-ified
    # print(type(train_losses))
    # print(type(validation_losses))

    train_val_log = {"model_id": model_id,
                    "current_ds_training": current_dataset_name,
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
                        exp_setup=exp_setup,
                        hyper_param_str=hyper_param_str)

        # print(log_test)

        tests.append(log_test)
    # print(test)
    # print(type(test))


    return model, train_val_log, tests

def continual_learning( model,
                        tokenizer,
                        model_id,
                        training_order,
                        zero_testing_order,
                        hf_datasets,
                        epochs_array,
                        ks_array,
                        cl_technique,
                        type_experiment,
                        batch_size,
                        lr,
                        cut_batch,
                        loss_f,
                        optimizer,
                        exp_setup,
                        hyper_param_str):

  results = {
              "model": model_id,
              "training_order": training_order,
              "testing_order": training_order + zero_testing_order,
              "zero_testing":zero_testing_order,
              "epochs": epochs_array,
              "exp_setup": exp_setup,
              "cl_technique": cl_technique,
              "type_experiment": type_experiment,
              "batch_size": batch_size,
              "learning_rate": lr,
              "results": []
            }

  print("CONTINUAL LEARNING EXPERIMENT SET UP")
  for k, v in results.items():
    if k != "exp_setup":
      print(k, ":\t", v)
      print()

  # print()
  # print("DATA PREPARATION")
  # print()
  total_order = training_order + zero_testing_order
  data_ln = filter_sample_ds(hf_datasets, tokenizer, total_order, batch_size=batch_size, ks_array=False)
  data_loaders_train = {k:v for k,v in data_ln.items() if k in training_order}
  data_loaders_zero = {k:v for k,v in data_ln.items() if k in zero_testing_order}
  # print("Dataloaders")
  # print(data_loaders_train)
  # print(data_loaders_zero)
  # print("DL types:")
  # print(type(data_loaders_train)) # dict name_ds: list_dataloaders
  # print()

  testing_order = list(data_loaders_train.keys()) + list(data_loaders_zero.keys())
  print("Testing order")
  print(testing_order)
  # for the test_loaders i have to filter them to get a list of the second index of each loader
  test_ds = list(data_loaders_train.values()) + list(data_loaders_zero.values())
  test_loaders = [loader[2] for loader in test_ds]
  print("Test loaders")
  print(test_loaders)

  test_results_for_df = []
  for time, current_training_dataset in enumerate(data_loaders_train): # current tr_ds is a string!!
    print("------------------Starting Experience----------------")
    print(f"------------------TIME {time}------------------------")
    print()
    # torch.cuda.ipc_collect()
    # torch.cuda.empty_cache()
    epochs = epochs_array[time]
    if ks_array != None:
      num_samples = ks_array[time]
    else:
      num_samples = "all"
    current_dataset_name = training_order[time]
    print(current_training_dataset)
    # print(type(current_training_dataset))
    current_training_dataloader = data_loaders_train[current_training_dataset]
    print(current_training_dataloader) # dataloader with the three indices train/val/test
    print()
    print(f"Epochs in the current time: {epochs}\nNumber of training samples: {num_samples}\nCurrent Dataset: {current_dataset_name}")

    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad) # in case I increase/decrease the number of params

    ############################## GETTING THE EXP PARAMS, SERVING THE DATA, SAMPLING AND ALL OF THAT WORKS WELL ##################################

    # continue


    model, train_vals, tests = train(model=model,
                                    model_id=model_id,
                                    current_training_dataloader=current_training_dataloader,
                                    current_dataset_name=current_dataset_name,
                                    epochs=epochs,
                                    loss_f=loss_f,
                                    optimizer=optimizer,
                                    cut_batch=cut_batch,
                                    trainable_params=trainable_params,
                                    lr=lr,
                                    batch_size=batch_size,
                                    num_samples=num_samples,
                                    type_experiment=type_experiment,
                                    cl_technique=cl_technique,
                                    time=time,
                                    training_order=training_order,
                                    testing_order=testing_order,
                                    test_loaders=test_loaders,
                                    exp_setup=exp_setup,
                                    hyper_param_str=hyper_param_str
                                    )
    test_results_for_df.extend(tests)

    # print(len(test_results_for_df))
    # print(test_results_for_df[0])
    # print(test_results_for_df)

    time_results = {"time": time,
                    "train_val": train_vals,
                    "test":tests}

    results["results"].append(time_results)
    print("RESULTS FOR CURRENT EXPERIENCE DONE")
    print()
    print()
    print()

    if cl_technique == "zero_shot":
      break # uncomment for zero-shot

  print(test_results_for_df) # all the tests for each time are here
  df_log_test = pd.DataFrame(test_results_for_df, columns=test_results_for_df[0].keys())
  df_log_test.fillna(0, inplace=True)

  if cl_technique in ["zero_shot", "vainilla_finetune", "vainilla_finetuning"]:
    result_type = "baseline"
  else:
    result_type = "CL"

  save_results_csv(df=df_log_test,
                  model_id= model_id,
                  experiment_name=type_experiment + "_",
                  cl_technique=cl_technique,
                  result_type=result_type)

  print("RESULTS FOR CURRENT EXPERIENCE SAVED")

  return results, model

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  
## # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #  

with open("experiment_set_up_week_3.json", "r") as f:
    exp_setup = json.load(f)

seed = 42
set_seed(seed)

load_dotenv("env_vars.env")
# hf_token = os.environ.get("HF_ACCESS_TOKEN")
# HfFolder.save_token(hf_token)
# print(whoami()["name"])

model_ids =     [
                # "FacebookAI/roberta-base",
                # "Xuhui/ToxDect-roberta-large",
                "diptanu/fBERT"
                # "GroNLP/hateBERT"
                ]

data_set_up =   [
        {       "type":"alternate_mis_raci",
                "train_test": ["evalita", "waseem-racism", "ibereval", "hateval-immigrant"],
                "zero": ["davidson", "founta_hateful_57k", "hateval-women", "waseem-sexism"],
                "epochs_array": [8, 8, 8, 8],
                "training_ks":  None,

        }                
]

cl_techniques = ["agem"]


experiments = list(itertools.product(model_ids,
                                    data_set_up,
                                    ))

# experiments_zero = list(itertools.product(model_ids,
#                                     data_set_up_zero_shot,
#                                     ))

experiments_cl = list(itertools.product(model_ids,
                                    data_set_up,
                                    cl_techniques
                                    ))

cl_hyperparams = {
        "ewc": {"ewc_lambda":1000},
        "agem": {"mem_size":100},
        "lwf": {"lwf_lambda":1,
                "temperature":2},
        "mas": {"mas_lambda":1000}
        }

filter_cl_techniques = ['agem']

hf_datasets = load_data_hf()

loss_f = CrossEntropyLoss()
lr = 1e-5
batch_size = 32
filter_cl_techniques = [cl_techniques[:-2]]
cut_batch = False

experiments_results = []

for experiment in experiments_cl:

    # torch.cuda.ipc_collect()
    # torch.cuda.empty_cache()

    model_id = experiment[0]
    cl_technique = experiment[-1]
    # print(cl_technique)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoContinualLearner(model_id, num_labels=2, device=device)

    cl_params = cl_hyperparams[cl_technique]
    hyper_param_str = "=".join([str(k) + "=" + str(v) for k, v in cl_params.items()])
    model.init_cl(cl_technique, **cl_params)

    optimizer = AdamW(model.model.parameters(), lr=lr)

    training_order = experiment[1]["train_test"]
    zero_testing_order = experiment[1]["zero"]
    epochs_array = experiment[1]["epochs_array"]
    ks_array = experiment[1]["training_ks"]
    type_experiment = experiment[1]["type"]

    results, model =  continual_learning(model,
                            tokenizer,
                            model_id,
                            training_order,
                            zero_testing_order,
                            hf_datasets, # loaded this var earlier in the notebook
                            epochs_array,
                            ks_array,
                            cl_technique,
                            type_experiment,
                            batch_size,
                            lr,
                            cut_batch,
                            loss_f,
                            optimizer,
                            exp_setup,
                            hyper_param_str) # loaded exp_setup earlier

    experiment_json_name = "_".join([type_experiment, model_id.replace("/", "-"), cl_technique, hyper_param_str]) + ".json"

    try:
        with open(experiment_json_name, "w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        print("Result couldn't be saved.")
        print(e)
    print(results)

    results["json_name"] = experiment_json_name
    experiments_results.append(results)
    
    