from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer

# model_ids =     ["FacebookAI/roberta-base",
#                 "Xuhui/ToxDect-roberta-large",
#                 "diptanu/fBERT",
#                 "GroNLP/hateBERT"
#                 ]

# for model in model_ids:
#     ld_tokenizer = AutoTokenizer.from_pretrained(model)
#     ld_model = AutoModelForSequenceClassification.from_pretrained(model)
#     print(f"model: {model} loaded\n")

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

ld_tokenizer = AutoTokenizer.from_pretrained(model_id)
ld_model = AutoModelForCausalLM.from_pretrained(model_id)
print("Model Loaded")
