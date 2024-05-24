
import numpy as np
import torch
import torch.nn as nn
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    matthews_corrcoef
)
from transformers import AutoTokenizer, Trainer

from datasets import Dataset
from accelerate import Accelerator
# Imports specific to the custom peft lora model



# Helper Functions and Data Preparation
def truncate_labels(labels, max_length):
    """Truncate labels to the specified max_length."""
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    """Compute metrics for evaluation."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding (-100 labels)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute precision, recall, F1 score, and AUC
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    
    # Compute MCC
    mcc = matthews_corrcoef(labels, predictions) 
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc} 

def compute_loss(model, inputs):
    """Custom compute_loss function."""
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

# Load the data from pickle files (replace with your local paths)
with open("data/train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)

with open("data/test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)

with open("data/train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)

with open("data/test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
max_sequence_length = 1000

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)

# Directly truncate the entire list of labels
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)

train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

# Compute Class Weights
classes = [0, 1]  
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
accelerator = Accelerator()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss