import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
import os 
import sys 
import warnings 
import random
from pprint import pprint as pp
from dotenv import load_dotenv
from datetime import date
import re
from tqdm.auto import tqdm
import emoji
import json
from huggingface_hub import whoami, HfFolder
from copy import deepcopy
from datetime import timedelta


import gc

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Softmax, KLDivLoss
from torch.utils.data.dataloader import DataLoader

from transformers import BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import set_seed, Seq2SeqTrainer, LlamaTokenizer

from datasets import Dataset, DatasetDict

from peft import LoraConfig, get_peft_model

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


# for some reason the data collator is outputing dimensions batch_size x 1 x seq_length, need to squeeze THE INPUTS, NOT THE LOGITS when needed
def squeeze_notneeded_dimension(x):
    # print("X SHAPE")
    # print(x.shape)
    x = x.squeeze(1) if x.dim() == 3 and x.size(1) == 1 else x
    # print("X NEW SHAPE")
    # print(x.shape)
    return x

def log_hf():
    
    load_dotenv("env_vars.env")
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    HfFolder.save_token(hf_token)
    return print(whoami()["name"])

def setup():
    try:
        dist.init_process_group("nccl", timeout=timedelta(seconds=18000))
        # dist.init_process_group("gloo")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank
    except Exception as e:
        print(e)
        print("DISTR TRAINING ALREADY INITIALIZED")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank


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

def translate_class_to_label(class_):

    translation_dict = {"not_hate": "NOT HATEFUL",
                        "explicit_hate": "HATEFUL",
                        "implicit_hate": "HATEFUL"}

    translated_label = translation_dict[class_]

    return translated_label

def format_message(formatted_prompt, label=True):
    if label:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": formatted_prompt},
            {"role": "assistant", "content": label}
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": formatted_prompt}
        ]
    return messages

base_prompt = """You are a social media content moderator.
INSTRUCTION: The following is a social media message that needs to be classified with the label HATEFUL or NOT HATEFUL.
MESSAGE: {}
OUTPUT AND FORMAT: your output should be just the label."""

def format_prompt(text, base_prompt=base_prompt):

    formatted_prompt = base_prompt.format(text)
    
    return formatted_prompt

loss_fn = CrossEntropyLoss(ignore_index=-100) # ignore the left pad tokens
def loss_f(logits, labels):

    flat_logits = logits.view(-1, logits.size(-1))
    flat_labels = labels.view(-1)

    loss = loss_fn(flat_logits, flat_labels)
    
    return loss

def translate_prediction_to_label(text):
    if "NOT HATEFUL" in text:
        text_clean = text.replace("NOT HATEFUL", "")
        if "HATEFUL" in text_clean or "HATEFUAL" in text_clean:
            return 2
        else:
            return 0
    elif "NOT_HATEFUL" in text:
        text_clean = text.replace("NOT_HATEFUL", "")
        if "HATEFUL" in text_clean or "HATEFUAL" in text_clean:
            return 2
        else: 
            return 0
    elif "HATEFUL" in text:
        text_clean = text.replace("HATEFUL", "")
        if "NOT_HATEFUL" in text_clean or "NOT HATEFUL" in text_clean:
            return 2
        else:
            return 1
    else:
        return 2

# to test_model we pass the whole dataset dictionary, not just the split
def test_model(model, tokenizer, base_prompt, ds, device, mode=None, verbose=False):

    print("_________________________________")
    print("Testing the model")

    predictions_test = []
    labels_test = []
    predicted_strings = []
    labels_strings = []
    full_generation = []

    model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float32):
        with torch.no_grad():
            print("TESTING DS")
            print(ds)
            print()
            print(len(ds), "len ds")
            # for i, test_item in enumerate(ds["test"]):
            for i, test_item in enumerate(ds):
                print("Type of the ds")
                print(type(ds))
                print("Test item type")
                print(type(test_item))
                print("full test item")
                print(test_item)
                # print("TESTING ITEM")
                # print(test_item)
                # print(i)
                target_label = test_item["label"]
                labels_strings.append(target_label)
                # print("TARGET LABEL")
                # print(target_label)
                if target_label == "NOT HATEFUL":
                    target_label = 0
                elif target_label == "HATEFUL":
                    target_label = 1
                
                labels_test.append(target_label)

                formatted_prompt = test_item["formatted_prompt"]
                # prompt_plus_messages = base_prompt.format(clean_post)
                # print("FORMATTED PROMPT")
                # print(formatted_prompt)
                print(len(formatted_prompt), "len formatted_prompt")

                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": formatted_prompt}
                ]
                chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                print("CHAT TEMPLATE COMPUTED")
                print(chat_template)
                input_dict = tokenizer(chat_template, return_tensors="pt", add_special_tokens=False)
                input_dict = {k: squeeze_notneeded_dimension(v).to(device) for k, v in input_dict.items()}
                input_ids_tokenized = input_dict["input_ids"]
                attention_mask = input_dict["attention_mask"]
                
                # print("TOKENIZED CHAT TEMPLATE COMPUTED")
                # print(input_ids_tokenized)
                # print(type(input_ids_tokenized))
                # if input_ids_tokenized.shape[0] == 1:
                #     print("wrong size")
                #     input_ids_tokenized = input_ids_tokenized.squeeze(0)
                #     attention_mask = attention_mask.squeeze(0)
                # print("NEW SHAPE")
                # print(input_ids_tokenized.shape)
                # print(attention_mask.shape)
                # ######################
                # print("----------------right beforeoutput---------------------------------------")
                # # print(model)
                # # print(model.module)
                # # print(dir(model))
                # # print(dir(model.module))
                # # print(help(model.module.generate))
                # print(model.module.generate(input_ids=input_ids_tokenized, 
                #                                 attention_mask=attention_mask, 
                #                                 top_p=0.9, 
                #                                 temperature=0.6, 
                #                                 max_new_tokens=10,
                #                                 return_dict_in_generate=False))
                # print("----------------right after output---------------------------------------")
                output = model.module.model.generate(input_ids=input_ids_tokenized, 
                                                attention_mask=attention_mask, 
                                                top_p=0.9, 
                                                temperature=0.6, 
                                                max_new_tokens=10,
                                                return_dict_in_generate=False)
                        
                # pred = tokenizer.batch_decode(output, skip_special_tokens=True)
                # print("OUTPUT COMPUTED")
                # print(output)
                # print(type(output))
                seq = output[0]
                # print(tokenizer.decode(seq, skip_special_tokens=True).strip())
                pred = tokenizer.decode(seq[input_ids_tokenized.shape[1]:], skip_special_tokens=True)

                full_generation.append(seq)
                predicted_strings.append(pred)
                print("PRED COMPUTED")
                print(pred)
                print()
                pred_label = translate_prediction_to_label(pred)
                # print("PRED LABEL COMPUTED")
                # print(pred_label)
                predictions_test.append(pred_label)

                if mode != None:
                    break

                if verbose:
                    # print("Text: ", pred)
                    print()
                    # print("Chat Template: ", chat_template)
                    # print()
                    # print("Tokenized Chat Template: ", input_ids_tokenized)
                    # print()
                    # print("Output: ", output)
                    # print()
                    # print("Prediction: ", pred)
                    # print("____________________________________________________")
                    # print("Checking the predictions")
                    # print("Number of prediction batches")
                    # print(len(predictions_test))
                    # print("Number of predictions in each batch")
                    # print(len(predictions_test[0]))
                    # print()
                    # print("--------------------------------------------------")
                    # print("CHECKING GENERATION")
                    # print()
                    # print(input_ids_tokenized)
                    # print()
                    # print("List predictions")
                    # print(predictions_test)
                    # print()
                    # print("List labels")
                    # print(labels_test)

        result = get_scores_from_preds(predictions_test, labels_test)
        result["predicted_strings"] = predicted_strings
        result["labels_strings"] = labels_strings
        result["full_generation"] = full_generation

    return result

def get_scores_from_preds(predictions_test, labels_test, metrics=[f1_score, precision_score, recall_score]):
    
    y_labels = labels_test
    y_outputs = predictions_test

    result = {clean_metric_name(str(score)): float(score(y_labels, y_outputs, average='macro')) for score in metrics}
    results_hate_class = {"HATE_" + clean_metric_name(str(score)): float(score(y_labels, y_outputs, labels=[1], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}
    results_nohate_class = {"NoHATE_" + clean_metric_name(str(score)): float(score(y_labels, y_outputs, labels=[0], average='macro')) for score in metrics if score  in [f1_score, precision_score, recall_score]}

    result.update(results_hate_class)
    result.update(results_nohate_class)
    result["predictions"] = [int(pred_label) for pred_label in y_outputs]
    result["labels"] = [int(actual_y) for actual_y in y_labels]

    return result

def log_test(model,
        model_id:str,
        test_ds,
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
        mode=None,
        tokenizer=None,
        base_prompt=None,
        device=None,
        ):


    log_test =                                 {}
    log_test["model"] =                        model_id
    log_test["type_experiment"] =              type_experiment
    log_test["n_trainable_params"] =           int(trainable_params)
    log_test["cl_technique"]  =                cl_technique
    log_test["time"] =                         int(time)
    log_test["dataset"] =                      current_testing_dataset
    log_test["curr_train"] =                   current_training_dataset
    # log_test["curr_train_hate_type"] =         exp_setup["general_ds_categories"][current_training_dataset]
    # log_test["curr_train_mix_type"] =          exp_setup["mixtures_ds"][current_training_dataset]
    # log_test["hate_type_test"] =               exp_setup["general_ds_categories"][current_testing_dataset]
    # log_test["mixture_test"] =                 exp_setup["mixtures_ds"][current_testing_dataset]
    log_test["n_epochs_per_experience"] =      int(epochs)
    log_test["learning_rate"] =                float(lr)
    log_test["batch_size"] =                   int(batch_size)
    log_test["num_samples"] =                  num_samples
    log_test["hyper_param"] =                  hyper_param_str

    
    if current_testing_dataset == current_training_dataset:
        log_test["shots"] = "IN TRAINING"

    elif current_testing_dataset not in training_order: # if we have already passed all the training indexes, that means that we are doing the zero shots, which i left at the end
        log_test["shots"] = "ZERO SHOT"

    else:
        log_test["shots"] = "PASSED TRAINING"
    # print("LOG TEST BEFORE CALCULATING THE SCORES")
    # print(log_test)
    # print(current_testing_dataset)
    try:
        print("COOL UNTIL TEST MODEL-----------")
        test_metrics = test_model(model=model, tokenizer=tokenizer, base_prompt=base_prompt, ds=test_ds, mode=mode, verbose=False, device=device)
        print("Testing for " + current_testing_dataset + " completed")
    except Exception as e:
        print("TESTING FAILED")
        print(e)
        test_metrics = {clean_metric_name(str(score)): "FAULTY INFERENCE" for score in metrics}

    log_test.update(test_metrics)

    for k, v in log_test.items():
        if type(v) not in [int, float, str, list]:
            print("Wrong dictionary format")
            print(k)
            print(v)
    print("Test log completed")
    return log_test

def zero_shot_test(model,
        model_id:str,
        test_datasets:list,
        training_order:list,
        trainable_params,
        metrics=[f1_score, precision_score, recall_score],
        mode=None,
        tokenizer=None,
        base_prompt=None,
        device=None,):

    zero_log_tests = []

    for test_data in test_datasets:
        test_data_name = str(list(test_data.keys())[0])
        test_loader = list(test_data.values())[0]
        log_test = log_test(model,
                            model_id,
                            test_loader,
                            type_experiment="ZERO SHOT",
                            cl_technique="ZERO SHOT",
                            time="NA",
                            current_training_dataset=None,
                            current_testing_dataset=test_data_name,
                            training_order=training_order,
                            trainable_params=trainable_params,
                            epochs="NA",
                            lr="NA",
                            batch_size="NA",
                            num_samples="NA",
                            exp_setup="NA",
                            hyper_param_str="NA",
                            metrics=metrics,
                            mode=mode,
                            tokenizer=tokenizer,
                            base_prompt=base_prompt,
                            device=device,
                            )
                            
        zero_log_tests.append(log_test)

    return zero_log_tests 

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
                actual_inputs = {k:squeeze_notneeded_dimension(v).to(self.device) for k,v in inputs.items() if k != "logits"}
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

                inputs_mem = {k:squeeze_notneeded_dimension(v).to(self.device) for k,v in inputs_mem.items()}
                labels_mem = squeeze_notneeded_dimension(labels_mem).to(self.device)
                outputs = self.model(**inputs_mem)
                logits = outputs.logits
                print("MEMORY BATCH SHAPES:")
                print(f"  logits: {logits.shape}")
                print(f"  labels: {labels_mem.shape}")
                print(f"  input_ids: {inputs_mem['input_ids'].shape}")
                print(f"  attention_mask: {inputs_mem['attention_mask'].shape}")
                print()
                B, T, V = logits.shape
                assert labels_mem.shape == (B, T), \
                    f"Mismatch logits: {logits.shape}, labels: {labels_mem.shape}"

                loss = loss_f(logits, labels_mem)
                loss.backward()

            self.ref_grad = [p.grad.clone() for p in self.model.parameters() if p.requires_grad] # i think this should be with req grad
            self.model.zero_grad()

    def post_backward(self):
        """Operations after backward pass"""
        if self.technique == "agem" and hasattr(self, 'ref_grad'):
            # Project gradients
            dot_product = sum(torch.sum(p.grad * g_ref)
                        for p, g_ref in zip(self.model.parameters(), self.ref_grad) if p.requires_grad)
            ref_norm = sum(torch.sum(g_ref * g_ref) for g_ref in self.ref_grad)

            if dot_product < 0:  # Negative interference
                scale = dot_product / (ref_norm + 1e-8)
                for p, g_ref in zip(self.model.parameters(), self.ref_grad):
                    if p.grad is not None and p.requires_grad:
                        p.grad -= scale * g_ref

    def post_task_update(self, dataloader=None):
        """Update after each task"""
        if self.technique == "ewc":
            # Compute Fisher information
            self.model.eval()
            for batch in dataloader:
                self.model.zero_grad()
                inputs = {k:squeeze_notneeded_dimension(v).to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}
                labels = squeeze_notneeded_dimension(batch['labels']).to(self.device)

                # outputs = self.model(**inputs, labels=labels)
                outputs = self.model(**inputs)

                loss = loss_f(outputs.logits, labels)
                loss.backward()

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
                inputs = {k:squeeze_notneeded_dimension(v).to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}
                labels = squeeze_notneeded_dimension(batch['labels']).to(self.device)
                print("STORING TO MEMORY:")
                print({k: v.shape for k, v in inputs.items()}, "labels:", labels.shape)

                self.memory.append((inputs, labels))
                if len(self.memory) >= self.mem_size:
                    break
                for i, (mem_inputs, mem_labels) in enumerate(self.memory):
                    print(f"Memory example {i}:")
                    for k, v in mem_inputs.items():
                        print(f"{k}: {v.shape} | dtype: {v.dtype}")
                    print(f"labels: {mem_labels.shape} | dtype: {mem_labels.dtype}")
                print(f"Total memory stored: {len(self.memory)} examples")


        elif self.technique == "lwf":
            # Save model snapshot
            self.old_model = deepcopy(self.model)
            self.old_model.eval()

        elif self.technique == "mas":
            # Update importance weights
            self.model.eval()
            for batch in dataloader:
                self.model.zero_grad()
                inputs = {k:squeeze_notneeded_dimension(v).to(self.device) for k, v in batch.items()
                        if k in ['input_ids', 'attention_mask']}

                outputs = self.model(**inputs)
                # does this need a loss?????????????
                torch.norm(outputs.logits, p=2, dim=1).mean().backward()

                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        self.importance[n] += p.grad.abs() / len(dataloader)

            # Update stored parameters
            self.old_params = {n: p.clone().detach()
                            for n, p in self.model.named_parameters()
                            if p.requires_grad}

class AutoContinualLearner(nn.Module):
    def __init__(self, model_name, device, quantization_config, torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch_dtype
        )
        self.n_initial_params = sum(t.numel() for t in self.model.parameters())
        self.n_trainable_params_initial = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        self.cl = None

    def init_cl(self, technique, lora_config, **kwargs):
        """Init the continual learning technique AND peft version of the model"""
        self.model = get_peft_model(self.model, lora_config).to(self.device)
        self.n_params_lora = sum(t.numel() for t in self.model.parameters())
        self.n_trainable_params_lora = sum(t.numel() for t in self.model.parameters() if t.requires_grad)
        self.model.print_trainable_parameters()
        self.cl = CLTechniques(self.model, self.device, technique, **kwargs)
    def forward(self, **kwargs):
        return self.model(**kwargs)


with open("llm_experiments_set_up.json", "r") as f:
    exp_setup = json.load(f)

warnings.filterwarnings("ignore") 
load_dotenv("env_vars.env")

set_seed(42)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def main(
    type_experiment:str,
    cl_technique:str,
    training_order:list,
    testing_order:list,
    model_id = "Models/Qwen2.5-0.5B",
    exp_setup = exp_setup,
    mode = None,
    dataset_path="df_from_exp_to_imp.csv",
    metrics=[f1_score, precision_score, recall_score, roc_auc_score]
        ):

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    ########################################################## DATA WORK
    print("_________________________________")
    print("Preapring the Tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(model_id + "/Tokenizer")
    if tokenizer.pad_token is None and "Llama" in model_id: tokenizer.pad_token = '<|finetune_right_pad_id|>'
    elif tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    tokenizer.chat_template = open(model_id + "/Tokenizer/chat_template.jinja").read()

    print("_________________________________")
    print("Loading and filtering the Data")

    df = pd.read_csv(dataset_path)

    #### Attaching the prompt to the clean post
    df["formatted_prompt"] = df["clean_post"].apply(format_prompt)
    df["label"] = df["class"].apply(translate_class_to_label)

    # ### Turning the Df into a DatasetDict

    times_array = list(df["time"].unique())
    datasets = []
    dataset_names = list(df["task"].unique())

    datasets_test = []
    for i, task in enumerate(testing_order):
        split_df = df[(df["split"] == "test") & (df["task"] == task)]
        hf_split = Dataset.from_pandas(split_df)
        datasets_test.append({testing_order[i]: hf_split})


    hf_datasets = []

    for i, dataset in enumerate(datasets):
        print("dataset")
    
        hf_ds = DatasetDict({dataset[0]["split"][0]: dataset[0], 
                            dataset[1]["split"][0]: dataset[1],
                            dataset[2]["split"][0]: dataset[2]})
        hf_ds_name = training_order[i]
        hf_datasets.append({hf_ds_name: hf_ds})

    for i, ds in enumerate(hf_datasets):
        for task_name, hf_data in ds.items():
            for split in hf_data:
                if split == "test":
                    data_loaders[i]["test"] = hf_data[split]

    print("DATA LOADERS AT THE END OF PROCESSING")
    pp(data_loaders)
    print()
    print("TEST DATA AT THE END OF PROCESSING")
    pp(datasets_test)
    print()

    # ### So far, created the prompt, did the messages with the prompt and answer in place. Applied to chat template and tokenized 

    ########################3#################### MODEL WORK

    print("_________________________________")
    print("Loading the model and model config with LoRA and 4-bit quantization nf4")

    bnb_config = BitsAndBytesConfig(  
                                    load_in_4bit= True,
                                    bnb_4bit_quant_type= "nf4",
                                    bnb_4bit_compute_dtype= torch.bfloat16,
                                    bnb_4bit_use_double_quant= True,
                                )


    hyper_param_str = "="

    model = AutoContinualLearner(model_id + "/Model", device, bnb_config)
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                # find_unused_parameters=True
                )

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("_________________________________")

    model, test_results, train_results = zero_shot_test(model,
                                                        model_id,
                                                        datasets_test,
                                                        training_order,
                                                        n_trainable_params,
                                                        metrics=metrics,
                                                        mode=mode,
                                                        tokenizer=tokenizer,
                                                        base_prompt=base_prompt,
                                                        device=device)

    experiment_json_name = ""
    if local_rank==0:
        print("_________________________________")
        print("Saving the results")
        pp(test_results)
        pp(train_results)
        experiment_json_name = "_".join([type_experiment, model_id.replace("/", "-"), cl_technique, hyper_param_str]) + ".json"
        try:
            with open(experiment_json_name, "w") as f:
                json.dump(test_results, f, indent=4)
            print("Results saved successfully")
        except Exception as e:
            print("Result couldn't be saved.")
            print(e)
        try:
            with open("train_log-" + experiment_json_name, "w") as f:
                json.dump(train_results, f, indent=4)
            print("Train Log saved successfully")
        except Exception as e:
            print("Train Log couldn't be saved.")
            print(e)

    # if local_rank == 0:
    #     print("_________________________________")
    #     print("Saving the model and Tokenizer")
    #     model_name = model_id.split("/")[-1]
    #     model.module.model.save_pretrained(f"alberto-lorente/{experiment_json_name}/model_test")
    #     tokenizer.save_pretrained(f"alberto-lorente/{experiment_json_name}/tokenizer_test")

    print("RUN SUCCESSFULLY")
    print()
    print()
    print("_________________________________")
    print()
    print(experiment_json_name.replace(".json", "") + " DONE")
    print()
    print("_________________________________")
    print()

if __name__ == "__main__":

    mode=None
    batch_size=2
    models = [  "Models/SmolLM2-360M-Instruct", 
                # "Models/Llama-3.2-1B-Instruct", 
                # "Models/Qwen2.5-0.5B-Instruct",
                "Models/TinyLlama-1.1b-Chat-v1.0", ]

    cl_technique = ["vainilla_finetune", "ewc", "agem", "mas"]

    main(
        type_experiment="ZERO-SHOT",
        cl_technique="ZERO-SHOT",
        model_id = "Models/SmolLM2-360M-Instruct",
        training_order=[],
        testing_order=["explicit_hs", "implicit_hs"],
        exp_setup = exp_setup,
        mode = mode,
        dataset_path="df_from_exp_to_imp.csv",
        )