#!/usr/bin/env python
# coding: utf-8

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

import gc

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data.dataloader import DataLoader

from transformers import BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM 
from transformers import set_seed, Seq2SeqTrainer, LlamaTokenizer

from datasets import Dataset, DatasetDict

from peft import LoraConfig, get_peft_model

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


def log_hf():
    
    load_dotenv("env_vars.env")
    hf_token = os.environ.get("HF_ACCESS_TOKEN")
    HfFolder.save_token(hf_token)
    return print(whoami()["name"])

def setup():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

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

def get_probability_distribution(logits):
    probability_dist = Softmax(dim=-1)(logits)
    return probability_dist

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
    else:
        return 1

# to test_model we pass the whole dataset dictionary, not just the split
def test_model(model, tokenizer, base_prompt, ds, mode=None, verbose=False):

    print("_________________________________")
    print("Testing the model")

    predictions_test = []
    labels_test = []

    model.eval()
    with torch.no_grad():
        
        for i, test_item in enumerate(tqdm(ds["test"])):

            target_label = test_item["label"]

            if target_label == "NOT HATEFUL":
                target_label = 0
            elif target_label == "HATEFUL":
                target_label = 1
            
            labels_test.append(target_label)
            clean_post = test_item["clean_post"]
            prompt_plus_messages = base_prompt.format(clean_post)

            messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt_plus_messages}
            ]
            chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True).rstrip()
            input_ids_tokenized = tokenizer(chat_template, return_tensors="pt", add_special_tokens=False).to(device)
            
            ######################
            output = model.module.generate(**input_ids_tokenized, top_p=90, temperature=0.6)
            # pred = tokenizer.batch_decode(output, skip_special_tokens=True)
            print(output)
            pred = tokenizer.decode(output[0][input_ids_tokenized['input_ids'].shape[1]:], skip_special_tokens=True).strip()
            pred_label = translate_prediction_to_label(pred)
            predictions_test.append(pred_label)

            if mode != None:
                break

            if verbose:
                print("Text: ", pred)
                print()
                print("Chat Template: ", chat_template)
                print()
                print("Tokenized Chat Template: ", input_ids_tokenized)
                print()
                print("Output: ", output)
                print()
                print("Prediction: ", pred)
                print("____________________________________________________")
                print("Checking the predictions")
                print("Number of prediction batches")
                print(len(predictions_test))
                print("Number of predictions in each batch")
                print(len(predictions_test[0]))
                print()
                print("--------------------------------------------------")
                print("CHECKING GENERATION")
                print()
                print(input_ids_tokenized)
                print()
                print("List predictions")
                print(predictions_test)
                print()
                print("List labels")
                print(labels_test)


    result = get_scores_from_preds(predictions_test, labels_test)

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

    elif training_order.index(current_training_dataset)  < training_order.index(current_training_dataset):
        log_test["shots"] = "PASSED TRAINING"

    else:
        log_test["shots"] = "ZERO SHOT"

    # print(current_testing_dataset)
    try:
        test_metrics = test_model(model, tokenizer, base_prompt, test_ds, mode=None, verbose=False)
    except Exception as e:
        print("TESTING FAILED")
        print(e)
        test_metrics = {clean_metric_name(str(score)): "FAULTY INFERENCE" for score in metrics}

    log_test.update(test_metrics)

    return log_test

def validate_model(model, validation_loader, device, world_size, local_rank, mode=None):

    model.eval()
    with torch.no_grad():

        print("_________________________________")
        print("Validating the model")
        print()
        torch.cuda.empty_cache()
        gc.collect()

        val_losses = [] # val loss for each batch

        for i, batch in enumerate(validation_loader):
            
            # batch.to(device)
            batch = {k:torch.squeeze(v).to(device) for k,v in batch.items()}

            output = model(**batch)
            logits = output.logits
            val_loss = loss_f(logits, batch["labels"])

            val_losses.append(val_loss.detach().item())

            if mode != None:
                break
        
        val_loss_epoch = sum(val_losses)/len(val_losses)
        val_loss_tensor = torch.tensor(val_loss_epoch, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss_tensor /= world_size

        if local_rank == 0:
            print(f"Validation Loss: {val_loss_tensor.item()}")

    return val_loss_tensor.item()

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
    model_id = "Models/Qwen2.5-0.5B",
    batch_size = 4,
    n_epochs = 2,
    lr = 1e-5,
    lora_r = 8,
    exp_setup = exp_setup,
        ):

    local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()

    ########################################################## DATA WORK
    print("_________________________________")
    print("Preapring the Data")

    df = pd.read_csv("df_from_exp_to_imp.csv")

    tokenizer = AutoTokenizer.from_pretrained(model_id + "/Tokenizer")
    if tokenizer.pad_token is None and "Llama" in model_id: tokenizer.pad_token = '<|finetune_right_pad_id|>'
    if tokenizer.pad_token is None and "Qwen" in model_id: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = open(model_id + "/Tokenizer/chat_template.jinja").read()

    # print(tokenizer.chat_template)
    # print(tokenizer.apply_chat_template("Hello World", tokenize=False, add_generation_prompt=True, return_tensors="pt"))

    def preprocess_and_tokenize(clean_post, label, base_prompt=base_prompt, max_length=512):
        
        prompt_plus_messages = base_prompt.format(clean_post)
        messages = [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": prompt_plus_messages},
                {"role": "assistant", "content": label.strip("\n")}
            ]

        chat_template = tokenizer.apply_chat_template(messages, tokenize=False, continue_final_message=False, add_special_tokens=False).rstrip()
        input_ids_tokenized = tokenizer(chat_template, return_tensors="pt", add_special_tokens=False, padding="max_length", max_length=max_length)["input_ids"]

        # getting the normal text just to know how much we need to add to the left as -100 and right as pad token
        input_ids_shape = tokenizer(chat_template, return_tensors="pt", add_special_tokens=False, padding=False)["input_ids"]

        # getting the label target to only predict the actual label and ignore the prompt
        labels_tokenized = tokenizer(label + tokenizer.eos_token, add_special_tokens=True, return_tensors="pt")["input_ids"]
        shape = input_ids_shape.shape[1] - labels_tokenized.shape[1]
        zeros = torch.zeros((1, shape), dtype=labels_tokenized.dtype, device=labels_tokenized.device)
        zeros.fill_(-100) # for the cross entropy loss
        labels_left_padded = torch.cat([zeros, labels_tokenized], dim=1)

        eos_n = input_ids_tokenized.shape[1] - labels_left_padded.shape[1]
        eos_n_tensor = torch.zeros((1, eos_n), dtype=labels_tokenized.dtype, device=labels_tokenized.device)
        eos_n_tensor.fill_(tokenizer.encode(tokenizer.pad_token, add_special_tokens=False)[0])
        labels_padded = torch.cat([labels_left_padded, eos_n_tensor], dim=1)

        # print(labels_padded.shape == input_ids_tokenized.shape)

        # shifting because we dont predict the first token
        input_ids_tokenized_left_shifted = input_ids_tokenized[:, :-1]
        labels_tokenized_right_shifted = labels_padded[:, 1:]

        attention_mask = input_ids_tokenized_left_shifted != tokenizer.pad_token_id
        
        return {
            "input_ids": input_ids_tokenized_left_shifted,
            "labels": labels_tokenized_right_shifted,
            "attention_mask": attention_mask
        }

    #### Attaching the prompt to the clean post
    df["formatted_prompt"] = df["clean_post"].apply(format_prompt)
    df["label"] = df["class"].apply(translate_class_to_label)

    # ### Turning the Df into a DatasetDict

    t_1 = []
    t_2 = []

    for split in df["split"].unique():

        split_df_1 = df[(df["split"] == split) & (df["time"] == 1)]
        split_df_2 = df[(df["split"] == split) & (df["time"] == 2)]

        hf_split_1 = Dataset.from_pandas(split_df_1)
        hf_split_2 = Dataset.from_pandas(split_df_2)
        
        t_1.append(hf_split_1)
        t_2.append(hf_split_2)

    hf_time_1 = DatasetDict({t_1[0]["split"][0]: t_1[0], 
                            t_1[1]["split"][0]: t_1[1],
                            t_1[2]["split"][0]: t_1[2]})

    hf_time_2 = DatasetDict({t_2[0]["split"][0]: t_2[0], 
                            t_2[1]["split"][0]: t_2[1],
                            t_2[2]["split"][0]: t_2[2]})


    ########################################################## TOKENIZER WORK

    hf_time_1 = hf_time_1.map(preprocess_and_tokenize, input_columns=["clean_post", "label"], batched=False)
    hf_time_2 = hf_time_2.map(preprocess_and_tokenize, input_columns=["clean_post", "label"], batched=False)

    n_samples = len(hf_time_1["train"])

    hf_time_1.set_format("torch")
    hf_time_2.set_format("torch")

    cols_to_remove = ["clean_post", "post", "class", "implicit_class", "extra_implicit_class", "target", "implied_statement", "split", "time", "formatted_prompt", "label", "__index_level_0__"]

    for split in hf_time_1:
        if split != "test":
            hf_time_1[split] = hf_time_1[split].remove_columns(cols_to_remove)
            hf_time_2[split] = hf_time_2[split].remove_columns(cols_to_remove)
        # hf_time_1[split] = hf_time_1[split].remove_columns(cols_to_remove)
        # hf_time_2[split] = hf_time_2[split].remove_columns(cols_to_remove)


    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    sampler_train_1 = DistributedSampler(hf_time_1["train"], num_replicas=world_size, rank=local_rank, shuffle=False)
    sampler_train_2 = DistributedSampler(hf_time_2["train"], num_replicas=world_size, rank=local_rank, shuffle=False)

    sampler_validation_1 = DistributedSampler(hf_time_1["validation"], num_replicas=world_size, rank=local_rank, shuffle=False)
    sampler_validation_2 = DistributedSampler(hf_time_2["validation"], num_replicas=world_size, rank=local_rank, shuffle=False)

    sampler_test_1 = DistributedSampler(hf_time_1["test"], num_replicas=world_size, rank=local_rank, shuffle=False)
    sampler_test_2 = DistributedSampler(hf_time_2["test"], num_replicas=world_size, rank=local_rank, shuffle=False)


    hf_time_1_train_loader = DataLoader(hf_time_1["train"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_train_1)
    hf_time_1_validation_loader = DataLoader(hf_time_1["validation"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_validation_1)
    hf_time_1_test_loader = DataLoader(hf_time_1["test"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_test_1)

    hf_time_2_train_loader = DataLoader(hf_time_2["train"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_train_2)
    hf_time_2_validation_loader = DataLoader(hf_time_2["validation"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_validation_2)
    hf_time_2_test_loader = DataLoader(hf_time_2["test"], collate_fn=data_collator, batch_size=batch_size, sampler=sampler_test_2)

    # ### So far, created the prompt, did the messages with the prompt and answer in place. Applied to chat template and tokenized 

    ########################3#################### MODEL WORK

    print("_________________________________")
    print("Loading the model and model config")

    bnb_config = BitsAndBytesConfig(  
                                    load_in_4bit= True,
                                    bnb_4bit_quant_type= "nf4",
                                    bnb_4bit_compute_dtype= torch.bfloat16,
                                    bnb_4bit_use_double_quant= True,
                                )

    model = AutoModelForCausalLM.from_pretrained(model_id + "/Model",
                                                torch_dtype=torch.bfloat16,
                                                # device_map="auto",
                                                quantization_config=bnb_config
                                                )

    # to deal with the fact that we dont make the first token prediction??


    model_size_before = sum(t.numel() for t in model.parameters())
    print("Model Size before LoRA", model_size_before)
    # print(model)
    # print()

    lora_alpha = lora_r*2
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=0.1,
        bias="none",
    )

    model = get_peft_model(model, config)
    print("Model After LoRA")
    model.print_trainable_parameters()


    # so that i can use 2 gpus
    model.to(device)
    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                # find_unused_parameters=True
                )
    # print(model)
    # print()
    if cl_technique in ["ewc", "agem", "lwf", "mas"]:
        cl_hyperparams = {
        "ewc": {"ewc_lambda":1500},
        "agem": {"mem_size":150},
        "lwf": {"lwf_lambda":1,
                "temperature":2},
        "mas": {"mas_lambda":1500}
        }

        cl_params = cl_hyperparams[cl_technique]
        hyper_param_str = "=".join([str(k) + "-" + str(v) for k, v in cl_params.items()])

    else:
        cl_params = "N/A"
        hyper_param_str = "N/A"

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    
    print("_________________________________")
    print("Training the model")
    print()

    # for task in tasks/dataset - train, eval
    for epoch in tqdm(range(n_epochs)):

        torch.cuda.empty_cache()
        gc.collect()
        model.train()

        global_training_losses = []
        global_validation_losses = []

        print("Epoch: ", epoch)
        losses = []

        for i, batch in enumerate(hf_time_1_train_loader):
            # if i > 0:
            #     continue

            torch.cuda.empty_cache()
            gc.collect()

            print("\tBatch: ", i)
            batch = {k:torch.squeeze(v).to(device) for k,v in batch.items()}

            # print(batch["input_ids"].shape)
            # print(batch["attention_mask"].shape)
            # print(batch["labels"].shape)

            output = model(**batch)
            logits = output.logits
            # print("Shape Logits")
            # print(logits.shape)
            # print("Shape Labels")
            # print(batch["labels"].shape)
            loss = loss_f(logits, batch["labels"])

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            losses.append(loss.detach().item())

        epoch_loss = sum(losses) / len(losses) # loss on current device

        epoch_loss_tensor = torch.tensor(epoch_loss, device=device)

        dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM) # loss on all devices

        epoch_loss_tensor /= world_size # avg

        if local_rank == 0:
            # print("-------------------------EXAMPLE BATCH, OUTPUT AND SO ON----------------")
            # print(batch["input_ids"])
            # print(batch["labels"])
            # print(batch["attention_mask"])
            # print("LOSS: ", loss)
            # print("OUTPUT: ", output)
            # print()
            # print("----------------------------------------------------------------")
            print(f"Epoch {epoch} Loss: {epoch_loss_tensor.item()}")
            global_training_losses.append(epoch_loss_tensor.item())

        val_loss = validate_model(model, hf_time_1_validation_loader, device, world_size, local_rank)
        global_validation_losses.append(val_loss)

    print()

    if local_rank == 0:
        print("---------------------TRAINING ENDED---------------")
        print("Final Training Losses:", global_training_losses)
        print("Final Validation Losses:", global_validation_losses)

    if local_rank == 0:
        test_result = log_test(model=model,
                            model_id=model_id,
                            test_ds=hf_time_1,
                            type_experiment=type_experiment,
                            cl_technique=cl_technique,
                            time=0,
                            current_training_dataset="",
                            current_testing_dataset="",
                            training_order=[],
                            trainable_params=n_trainable_params,
                            epochs=epochs,
                            lr=lr,
                            batch_size=batch_size,
                            num_samples=n_samples,
                            exp_setup=exp_setup,
                            hyper_param_str=hyper_param_str,
                            metrics=[f1_score, precision_score, recall_score, roc_auc_score])
        print(test_result)


    if local_rank==0:
        experiment_json_name = "_".join([type_experiment, model_id.replace("/", "-"), cl_technique, hyper_param_str]) + ".json"
        try:
            with open(experiment_json_name, "w") as f:
                json.dump(log_test, f, indent=4)
        except Exception as e:
            print("Result couldn't be saved.")
            print(e)

    if local_rank == 0:
        print("_________________________________")
        print("Saving the model and Tokenizer")
        model_name = model_id.split("/")[-1]
        model.module.save_pretrained(f"alberto-lorente/{model_name}/model_test")
        tokenizer.save_pretrained(f"alberto-lorente/{model_name}/tokenizer_test")

    print("RUN SUCCESSFULLY")

if __name__ == "__main__":
    main(
        type_experiment="from_expl_to_impl",
        cl_technique="vainilla_finetune",
        model_id = "Models/Qwen2.5-0.5B",
        batch_size = 4,
        n_epochs = 8,
        lr = 1e-5,
        lora_r = 8,
        exp_setup = exp_setup,
        )