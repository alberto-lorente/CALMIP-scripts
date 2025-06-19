
model_ids =     [
                # "FacebookAI/roberta-base",
                # "Xuhui/ToxDect-roberta-large",
                "diptanu/fBERT"
                # "GroNLP/hateBERT"
                ]

data_set_up =   [
        
        {
                "type":"from_rac_to_mis",
                "train_test": ["hateval-immigrant", "waseem-racism", "ibereval"],
                "zero": ["davidson", "founta_hateful_57k", "evalita", "hateval-women", "waseem-sexism"],
                "epochs_array": [8, 8, 8],
                "training_ks": None,
        }
                ]

cl_techniques = ["mas"]


cl_hyperparams = {
        "ewc": {"ewc_lambda":1000},
        "agem": {"mem_size":100},
        "lwf": {"lwf_lambda":1,
                "temperature":2},
        "mas": {"mas_lambda":1000}
        }

################
# 
# data_set_up =   [
        {       "type":"alternate_mis_raci",
                "train_test": ["evalita", "waseem-racism", "ibereval", "hateval-immigrant"],
                "zero": ["davidson", "founta_hateful_57k", "hateval-women", "waseem-sexism"],
                "epochs_array": [8, 8, 8, 8],
                "training_ks":  None,

        }                
]

# cl_techniques = ["agem"]

###########################3

# data_set_up = {       
#     "type":"from_general_to_alternating_miso_raci",
#     "train_test": ["davidson", "founta_hateful_57k", "ibereval", "hateval-immigrant", "hateval-women", "waseem-racism"],
#     "zero": ["evalita", "waseem-sexism"],
#     "epochs_array": [8, 8, 8, 8, 8, 8],
#     "training_ks":  None
#     }
# cl_technique = "ewc"

    
    