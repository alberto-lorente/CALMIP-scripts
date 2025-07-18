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


def log_failed_batch(batch):
    
    print("FAILED UNSQUEEZED BATCH")
    # if "failed_batches.json" not in list(os.listdir()):
    #     with open("failed_batches.json", "w") as f:
    #         json.dump([], f)
    # with open("failed_batches.json", "r") as f:
    #     failed_batches = json.load(f)
    # batch = {k:list(v.detach().cpu().numpy()) for k,v in batch.items()}
    # failed_batches.append(batch)
    # with open("failed_batches.json", "w") as f:
    #     json.dump(failed_batches, f)

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

                full_generation.append(pred) # seq is a torch tensor
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

def validate_model(model, validation_loader, device, world_size, local_rank, mode=None):

    model.eval()
    with torch.cuda.amp.autocast(dtype=torch.float32):
        with torch.no_grad():

            print("_________________________________")
            print("Validating the model")
            print()
            torch.cuda.empty_cache()
            gc.collect()

            val_losses = [] # val loss for each batch

            for i, batch in enumerate(validation_loader):
                
                # batch.to(device)
                batch_unsqueezed = batch
                print("\tBatch: ", i)
                batch = {k:squeeze_notneeded_dimension(v).to(device) for k,v in batch.items()}

                # print("Squeezed Batch")
                # for k, v in batch.items():
                #     print(f"{k}: {v.shape}")

                # print("Unsqueezed Batch")
                # for k, v in batch_unsqueezed.items():
                #     print(f"{k}: {v.shape}")

                try:
                    output = model.module.model(**batch)
                    # print(output)
                    logits = output.logits
                    # print("Shape Logits")
                    # print(logits.shape)
                    # print("Shape Labels")
                    # print(batch["labels"].shape)
                    val_loss = loss_f(logits, batch["labels"])
                    val_losses.append(val_loss.detach().item())

                except Exception as e:
                    
                    try:
                        print("Switching Batch Size to unsqueezed")
                        output = model(**batch_unsqueezed)
                        logits = output.logits
                        val_loss = loss_f(logits, batch_unsqueezed["labels"])

                        val_losses.append(val_loss.detach().item())

                        print()
                        print(e)
                    except:
                        log_failed_batch(batch)

                if mode != None:
                    break
            
            val_loss_epoch = sum(val_losses)/len(val_losses)
            val_loss_tensor = torch.tensor(val_loss_epoch, device=device)
            dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
            val_loss_tensor /= world_size

            if local_rank == 0:
                print("_________________________________")
                print("Validation completed")
                print()
                print(f"Validation Loss: {val_loss_tensor.item()}")
                print("_________________________________")

    return val_loss_tensor.item()

def train(  model, 
            model_id, 
            tokenizer,
            hyper_param_str,
            base_prompt,
            n_epochs, 
            train_loader, 
            validation_loader, 
            test_datasets, # actually will have to be the full datasets since we are not using dataloaders but the whole ds and then doing df["test"]
            type_experiment, 
            cl_technique, 
            time,
            current_training_dataset,
            current_testing_dataset,
            training_order,
            n_trainable_params,
            lr, 
            batch_size, 
            n_samples, 
            exp_setup, 
            loss_f, 
            optimizer, 
            device, 
            world_size, 
            local_rank,
            metrics=[f1_score, precision_score, recall_score, roc_auc_score], 
            mode=None):

    print("_________________________________")
    print("Training the model")
    print()

    global_training_losses = []
    global_validation_losses = []
    model.train()
    with torch.amp.autocast('cuda', dtype=torch.float32):

    # for task in tasks/dataset - train, eval
        for epoch in tqdm(range(n_epochs)):

            torch.cuda.empty_cache()
            gc.collect()

            epoch_validation_losses = []
            train_losses = []

            print("Epoch: ", epoch)

            for i, batch in enumerate(train_loader):
                # if i > 0:
                #     continue

                torch.cuda.empty_cache()
                gc.collect()
                batch_unsqueezed = batch
                print("\tBatch: ", i)
                batch = {k:squeeze_notneeded_dimension(v).to(device) for k,v in batch.items()}

                for k, v in batch.items():
                    if v.shape[0] != batch_size:
                        print(f"{k}: {v.shape}")
                        print()
                        continue

                try:
                    output = model.module.model(**batch)
                    # print(output)
                    logits = output.logits
                    print("Shape Logits")
                    print(logits.shape)
                    # print("Shape Labels")
                    # print(batch["labels"].shape)
                    loss = loss_f(logits, batch["labels"])
                    if i == 0:
                        print("Checking that the model type is the continual learner to do the cls")
                        print(type(model.module))
                        print(type(model.module.cl))
                        print(model.module.cl)
                        # print(dir(model.module))
                    if model.module.cl:
                        batch['logits'] = logits  # needed for LwF
                        print("Works before computing regularization")
                        loss += model.module.cl.compute_regularization(batch)
                        print("Works after computing regularization")
                        model.module.cl.pre_backward(batch)
                        print("Works after pre_backward")
                    # print("CL regularization and backward computed")

                    loss.backward()


                    # needed agem (A-GEM)
                    if model.module.cl:
                        print("Works before post_backward")
                        model.module.cl.post_backward()
                        print("Works after post_backward")
                    # print("CL post backward computed")

                    optimizer.step()
                    optimizer.zero_grad()

                    train_losses.append(loss.detach().item())

                except Exception as e:

                    print()
                    print(e)

                    print("-------------CHECKING SIZES-----------")
                    print("Squeezed Batch")
                    for k, v in batch.items():
                        print(f"{k}: {v.shape}")

                    print("Unsqueezed Batch")
                    for k, v in batch_unsqueezed.items():
                        print(f"{k}: {v.shape}")

                    try:
                        print("TRYING UNSQUEEZED BATCH")
                        output = model.module.model(**batch_unsqueezed)
                        # print(output)
                        logits = output.logits
                        # print("Shape Logits")
                        # print(logits.shape)
                        # print("Shape Labels")
                        # print(batch_unsqueezed["labels"].shape)
                        loss = loss_f(logits, batch_unsqueezed["labels"])
                        # print(dir(model.module))
                        if model.module.cl:
                            batch_unsqueezed['logits'] = logits  # needed for LwF
                            loss += model.module.cl.compute_regularization(batch_unsqueezed)
                            model.module.cl.pre_backward(batch_unsqueezed)
                        print("CL regularization and backward computed")

                        loss.backward()


                        # needed agem (A-GEM)
                        if model.module.cl:
                            model.module.cl.post_backward()
                        print("CL post backward computed")

                        optimizer.step()
                        optimizer.zero_grad()

                        train_losses.append(loss.detach().item())

                    except:
                        log_failed_batch(batch)


                if mode != None:
                    break

            if len(train_losses) == 0:
                print("NO TRAIN LOSSES")
                continue

            epoch_loss = sum(train_losses) / len(train_losses) # loss on current device

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
                print(f"Epoch Loss: {epoch_loss_tensor.item()}")
                global_training_losses.append(epoch_loss_tensor.item())

            val_loss = validate_model(model, validation_loader, device, world_size, local_rank, mode=mode)
            epoch_validation_losses.append(val_loss)
            global_validation_losses.append(val_loss)

        print()

        if local_rank == 0:
            print("---------------------TRAINING ENDED---------------")
            print("Final Training Losses:", global_training_losses)
            print("Final Validation Losses:", global_validation_losses)

        if model.module.cl:
            # batch = {k:torch.squeeze(v, dim=1).to(device) for k,v in batch.items()}
            model.module.cl.post_task_update(train_loader)

        print("-----------POST TRAINING CL UPDATES COMPLETED---------")


    tests_results = []
    train_val_log = {}
    if local_rank == 0:
        # tests_results = []
        print(test_datasets)
        for idx, test_ds in enumerate(test_datasets):
            test_data_name = str(list(test_ds.keys())[0])
            test_loader = list(test_ds.values())[0]
            print("Test Loader type:")
            print(type(test_loader))
            print("TESTING - " + test_data_name)
            print(test_loader)
            # print("-------------------------------------------------------")
            # print("fails here")
            test_result = log_test(model=model,
                            model_id=model_id,
                            test_ds=test_loader,
                            type_experiment=type_experiment,
                            cl_technique=cl_technique,
                            time=time,
                            current_training_dataset=training_order[time],
                            current_testing_dataset=test_data_name,
                            training_order=training_order,
                            trainable_params=n_trainable_params,
                            epochs=n_epochs,
                            lr=lr,
                            batch_size=batch_size,
                            num_samples=n_samples,
                            exp_setup=exp_setup,
                            hyper_param_str=hyper_param_str,
                            metrics=metrics,
                            mode=mode,
                            tokenizer=tokenizer,
                            base_prompt=base_prompt,
                            device=device)
            tests_results.append(test_result)
            # print(test_result)

        train_val_log = {"model_id": model_id,
                    "current_ds_training": training_order[time],
                    "epochs": int(n_epochs),
                    "time": int(time),
                    "num_samples_in_curr_ds": n_samples,
                    "training_losses_details": list(global_training_losses),
                    "validation_losses_details": list(global_validation_losses),
                    "type_experiment": type_experiment,
                    "cl_technique": cl_technique,
                    "train_order": " -> ".join(training_order)
                    }



        # print(tests_results)
        # print(train_val_log)
        # print(model)

    return model, tests_results, train_val_log 

def continual_training(model,
                        model_id,
                        tokenizer,
                        n_trainable_params,
                        base_prompt,
                        training_order,
                        testing_order,
                        hf_datasets,
                        testing_datasets,
                        epochs_array,
                        ks_array,
                        n_samples_per_ds,
                        cl_technique,
                        type_experiment,
                        hyper_param_str,
                        batch_size,
                        lr,
                        loss_f,
                        optimizer,
                        device,
                        world_size,
                        local_rank,
                        exp_setup,
                        metrics=[f1_score, precision_score, recall_score, roc_auc_score],
                        mode=None):

    zero_testing_order = [dataset for dataset in testing_order if dataset not in training_order]

    results = {
                "model": model_id,
                "training_order": training_order,
                "testing_order": testing_order,
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

    data_loaders_train = []
    data_loaders_val = []
    test_datasets = []
    for time, ds in enumerate(training_order):
        # print("training order")
        # print(training_order)
        # print("ds")
        # print(ds)
        # print("hf_datasets")
        # print(hf_datasets[time].keys())
        data_loaders_train.append(hf_datasets[time][ds]["train"])
        data_loaders_val.append(hf_datasets[time][ds]["validation"])
        test_datasets.append({"test":hf_datasets[time]["test"]})
    # print("TEST DATASETS BEFORE STARTING THE EXPERIENCE")
    # print(test_datasets)

    test_results = []
    train_results = []

    for time, current_training_dataset in enumerate(data_loaders_train): # current tr_ds is a string!!
        print("------------------Starting Experience----------------")
        print(f"------------------TIME {time}------------------------")
        print()
        # torch.cuda.ipc_collect()
        # torch.cuda.empty_cache()
        n_epochs = epochs_array[time]
        if ks_array != None:
            n_samples = ks_array[time]
        else:
            n_samples = n_samples_per_ds[time]

        current_dataset_name = training_order[time]
        current_testing_dataset = testing_order[time]

        print(f"Epochs in the current time: {n_epochs}\nNumber of training samples: {n_samples}\nCurrent Dataset: {current_dataset_name}")

        ############################## GETTING THE EXP PARAMS, SERVING THE DATA, SAMPLING AND ALL OF THAT WORKS WELL ##################################

        # continue


        model, tests, train_vals = train(   model=model,
                                            model_id=model_id,
                                            tokenizer=tokenizer,
                                            base_prompt=base_prompt,
                                            n_epochs=n_epochs, 
                                            train_loader=data_loaders_train[time], 
                                            validation_loader=data_loaders_val[time], 
                                            test_datasets=test_datasets, # actually will have to be the full datasets since we are not using dataloaders but the whole ds and then doing df["test"]
                                            type_experiment=type_experiment, 
                                            cl_technique=cl_technique, 
                                            time=time,
                                            current_training_dataset=current_training_dataset,
                                            current_testing_dataset=testing_order[time],
                                            training_order=training_order,
                                            n_trainable_params=n_trainable_params,
                                            lr=lr, 
                                            batch_size=batch_size, 
                                            n_samples=n_samples, 
                                            exp_setup=exp_setup, 
                                            hyper_param_str=hyper_param_str, 
                                            loss_f=loss_f, 
                                            optimizer=optimizer, 
                                            device=device, 
                                            world_size=world_size, 
                                            local_rank=local_rank,
                                            metrics=metrics, 
                                            mode=mode)
        if local_rank == 0:
            test_results.append(tests)
            train_results.append(train_vals)

        # print("RESULTS FOR CURRENT EXPERIENCE DONE")
        # print(tests)
        # print("TEST RESULTS")
        # print(test_results)
        # print("TRAIN RESULTS")
        # print(train_results)

        # if mode != None:
        #     break

    return model, test_results, train_results

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
                # print("MEMORY BATCH SHAPES:")
                # print(f"  logits: {logits.shape}")
                # print(f"  labels: {labels_mem.shape}")
                # print(f"  input_ids: {inputs_mem['input_ids'].shape}")
                # print(f"  attention_mask: {inputs_mem['attention_mask'].shape}")
                # print()
                B, T, V = logits.shape
                assert labels_mem.shape == (B, T), \
                    f"Mismatch logits: {logits.shape}, labels: {labels_mem.shape}"

                loss = loss_f(logits, labels_mem)
                loss.backward()

            self.ref_grad =  {
                                n: p.grad.clone()
                                for n, p in self.model.named_parameters()
                                if p.requires_grad and p.grad is not None
                            } # added p.grad is not none
            self.model.zero_grad()

    def post_backward(self):
        """Operations after backward pass"""
        if self.technique == "agem" and hasattr(self, 'ref_grad'):
            # Project gradients
            dot_product = sum(torch.sum(p.grad * self.ref_grad[n])
                        for n,p in self.model.named_parameters()
                        if p.requires_grad and p.grad is not None and n in self.ref_grad) # added p.grad is not none
            ref_norm = sum(torch.sum(g * g) for g in self.ref_grad.values())

            # ref_norm = sum(torch.sum(g_ref * g_ref) for g_ref in self.ref_grad if g_ref is not None) # added g_ref is not none

            if dot_product < 0:  # Negative interference
                scale = dot_product / (ref_norm + 1e-8)
                for n, p in self.model.named_parameters():
                    if p.grad is not None and p.requires_grad and n in self.ref_grad:
                        p.grad -= scale * self.ref_grad[n]

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
                # added one by one to the memory instead of the batches, check if the error was coming from this
                batch_size = inputs["input_ids"].shape[0]
                for i in range(batch_size):
                    sample_inputs = {
                        k: v[i:i+1] 
                        for k, v in inputs.items()
                    }
                    sample_labels = labels[i:i+1]
                    self.memory.append((sample_inputs, sample_labels))
                    if len(self.memory) >= self.mem_size:
                        break


                # self.memory.append((inputs, labels))
                # for i, (mem_inputs, mem_labels) in enumerate(self.memory):
                #     print(f"Memory example {i}:")
                #     for k, v in mem_inputs.items():
                #         print(f"{k}: {v.shape} | dtype: {v.dtype}")
                #     print(f"labels: {mem_labels.shape} | dtype: {mem_labels.dtype}")
                # print(f"Total memory stored: {len(self.memory)} examples")


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
    batch_size = 4,
    n_epochs = 2,
    lr = 1e-5,
    lora_r = 8,
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

    print("----------Preparing the Data-----------------")

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
    # print("times_array")
    # print(times_array)
    # print("dataset_names")
    # print(dataset_names)
    # print("Actual training_order")
    # print(training_order)

    for task in training_order:
        # print("Task")
        # print(task)
        time_ds = []
        for split in df["split"].unique():

            split_df = df[(df["split"] == split) & (df["task"] == task)]
            hf_split = Dataset.from_pandas(split_df)
            time_ds.append(hf_split)
        datasets.append(time_ds)

    datasets_test = []
    for i, task in enumerate(testing_order):
        # print("Task")
        # print(task)
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

    hf_datasets_processed = []
    for ds in hf_datasets:
        ds_dict = {}
        for task_name, hf_data in ds.items():
            ds_dict[task_name] = {}
            for split in hf_data:
                ds_dict[task_name][split] = hf_data[split].map(preprocess_and_tokenize, input_columns=["clean_post", "label"], batched=False)
        hf_datasets_processed.append(ds_dict)

    n_samples_per_ds = [
        len(hf_time["train"])
        for hf_data in hf_datasets
        for task_name, hf_time in hf_data.items() 
    ]

    # print("hf_datasets_processed")
    # print(hf_datasets_processed)
    # print()
    for ds in hf_datasets_processed:
        for task_name, hf_data in ds.items():
            for split in hf_data:
                hf_data[split].set_format("torch")

    cols_to_remove = ["clean_post", "post", "class", "implicit_class", "extra_implicit_class", 
                    "target", "implied_statement", "split", "time", "task",
                    "formatted_prompt", "label", "__index_level_0__"]


    hf_datasets_no_cols = []
    for ds in hf_datasets_processed:
        ds_dict = {}
        for task_name, hf_data in ds.items():
            ds_dict[task_name] = {}
            for split in hf_data:
                if split != "test":
                    ds_dict[task_name][split] = hf_data[split].remove_columns(cols_to_remove)
        hf_datasets_no_cols.append(ds_dict)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    distributed_samplers = []
    for ds in hf_datasets_no_cols:
        ds_dict = {}
        # print("ds:")
        # print(ds)
        for task_name, hf_data in ds.items():
            # print("task_name:")
            # print(task_name)
            # print("hf_data:")
            # print(hf_data)
            ds_dict[task_name] = {}
            for split in hf_data:
                # print("split:")
                # print(split)
                if split != "test":
                   distr_sampler = DistributedSampler(hf_data[split], num_replicas=world_size, rank=local_rank, shuffle=False)
                   ds_dict[task_name][split] = distr_sampler

        distributed_samplers.append(ds_dict)

    # print("distributed_samplers")
    # print(distributed_samplers)
    # print()

    data_loaders = []
    for i, distr_sampler in enumerate(distributed_samplers):
        ds_name = list(distr_sampler.keys())[0]
        ds_dict = {}
        ds_dict[ds_name] = {}
        for split, distributed_sampler in distr_sampler[ds_name].items():
            data_loader = DataLoader(hf_datasets_no_cols[i][ds_name][split], collate_fn=data_collator, batch_size=batch_size, sampler=distributed_sampler)
            ds_dict[ds_name][split] = data_loader
        data_loaders.append(ds_dict)
    
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

    # model = AutoModelForCausalLM.from_pretrained(model_id + "/Model",
    #                                             torch_dtype=torch.bfloat16,
    #                                             # device_map="auto",
    #                                             quantization_config=bnb_config
    #                                             )

    # to deal with the fact that we dont make the first token prediction??


    # model_size_before = sum(t.numel() for t in model.parameters())
    # print("Model Size before LoRA", model_size_before)
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

    if cl_technique in ["ewc", "agem", "lwf", "mas"]:
        cl_hyperparams = {
        "ewc": {"ewc_lambda":1500},
        "agem": {"mem_size":100},
        "lwf": {"lwf_lambda":1,
                "temperature":2},
        "mas": {"mas_lambda":1000}
        }

        cl_params = cl_hyperparams[cl_technique]
        hyper_param_str = "=".join([str(k) + "-" + str(v) for k, v in cl_params.items()])

        model = AutoContinualLearner(model_id + "/Model", device, bnb_config)
        model.init_cl(technique=cl_technique, lora_config=config, **cl_params)

    else:
        cl_params = {"NA": "NA"}
        hyper_param_str = "NA"

        model = AutoContinualLearner(model_id + "/Model", device, bnb_config)
        model.init_cl(technique=cl_technique, lora_config=config)

    model = DDP(model, 
                device_ids=[local_rank], 
                output_device=local_rank, 
                # find_unused_parameters=True
                )

    # print(model)
    # print(dir(model.module))
    # print(dir(model))
    # print(dir(model.module.model))
    # print(model)
    # print()

    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=lr)
    
    print("_________________________________")

    hf_datasets = data_loaders

    epochs_array = []

    for i in range(len(training_order)):
        epochs_array.append(n_epochs)

    ks_array = None

    model, test_results, train_results = continual_training(model=model,
                                                        model_id=model_id,
                                                        tokenizer=tokenizer,
                                                        n_trainable_params=n_trainable_params,
                                                        hyper_param_str=hyper_param_str,
                                                        base_prompt=base_prompt,
                                                        training_order=training_order,
                                                        testing_order=testing_order,
                                                        hf_datasets=hf_datasets,
                                                        testing_datasets=datasets_test,
                                                        epochs_array=epochs_array,
                                                        ks_array=ks_array,
                                                        n_samples_per_ds=n_samples_per_ds,
                                                        cl_technique=cl_technique,
                                                        type_experiment=type_experiment,
                                                        batch_size=batch_size,
                                                        lr=lr,
                                                        loss_f=loss_f,
                                                        optimizer=optimizer,
                                                        device=device,
                                                        world_size=world_size,
                                                        local_rank=local_rank,
                                                        exp_setup=exp_setup,
                                                        metrics=metrics,
                                                        mode=mode)

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
        type_experiment="from_expl_to_impl",
        cl_technique="vainilla_finetune",
        model_id = "Models/SmolLM2-360M-Instruct",
        training_order=["explicit_hs", "implicit_hs"],
        testing_order=["explicit_hs", "implicit_hs"],
        batch_size = batch_size,
        n_epochs = 8,
        lr = 1e-4,
        lora_r = 8,
        exp_setup = exp_setup,
        mode = mode,
        dataset_path="df_from_exp_to_imp.csv",
        )


    # for model_id in models:
    #     if model_id.strip("Models/") not in os.listdir("Models") :
    #         print(model_id.strip("Models/") + " NOT IN MODELS")
    #         print()
    #         print(os.listdir("Models"))
    #         continue
        
    #     main(
    #         type_experiment="from_expl_to_impl",
    #         cl_technique="vainilla_finetune",
    #         model_id = model_id,
    #         training_order=["explicit_hs", "implicit_hs"],
    #         testing_order=["explicit_hs", "implicit_hs"],
    #         batch_size = batch_size,
    #         n_epochs = 8,
    #         lr = 1e-4,
    #         lora_r = 8,
    #         exp_setup = exp_setup,
    #         mode = mode,
    #         dataset_path="df_from_exp_to_imp.csv",
    #         )

    # cl_techniques = ["ewc","agem", "mas"]
    # for cl_technique in cl_techniques:
    #     # "Models/Qwen2.5-0.5B", "Models/TinyLlama", 

    #     for model_id in models:
    #         if model_id.strip("Models/") not in os.listdir("Models") :
    #             print(model_id.strip("Models/") + " NOT IN MODELS")
    #             print()
    #             print(os.listdir("Models"))
    #             continue
    #         main(
    #             type_experiment="from_expl_to_impl",
    #             cl_technique=cl_technique,
    #             model_id = model_id,
    #             training_order=["explicit_hs", "implicit_hs"],
    #             testing_order=["explicit_hs", "implicit_hs"],
    #             batch_size = batch_size,
    #             n_epochs = 8,
    #             lr = 1e-4,
    #             lora_r = 8,
    #             exp_setup = exp_setup,
    #             mode = mode,
    #             dataset_path="df_from_exp_to_imp.csv",
    #             )   