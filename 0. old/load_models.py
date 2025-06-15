from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_ids =     ["FacebookAI/roberta-base",
                "Xuhui/ToxDect-roberta-large",
                "diptanu/fBERT",
                "GroNLP/hateBERT"
                ]

for model in model_ids:
    ld_tokenizer = AutoTokenizer.from_pretrained(model)
    ld_model = AutoModelForSequenceClassification.from_pretrained(model)
    print(f"model: {model} loaded\n")