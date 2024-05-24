# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    matthews_corrcoef
)
from transformers import (
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer
)

# Imports specific to the custom peft lora model
from peft import PeftModel

from sklearn.metrics import(
    matthews_corrcoef, 
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score
)
from peft import PeftModel
from transformers import DataCollatorForTokenClassification
from data import accelerator,tokenizer,train_dataset,test_dataset, compute_metrics
from train import SAVE_PATH


# Define paths to the LoRA and base models
base_model_path = "facebook/esm2_t6_8M_UR50D"
# "path/to/your/lora/model" Replace with the correct path to your LoRA model
peft_model_path = SAVE_PATH


# Load the base model
base_model = AutoModelForTokenClassification.from_pretrained(base_model_path)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, peft_model_path)
#model = PeftModel.from_pretrained(base_model, boft_model_path)
model = accelerator.prepare(model)  # Prepare the model using the accelerator

# Define label mappings
id2label = {0: "No binding site", 1: "Binding site"}
label2id = {v: k for k, v in id2label.items()}

# Create a data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

train_metrics = compute_metrics(train_dataset)
test_metrics = compute_metrics(test_dataset)

print(train_metrics, test_metrics)



