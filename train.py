import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch.nn as nn
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM, 
    DataCollatorForTokenClassification,
    TrainingArguments,
)

# Imports specific to the custom peft lora model
from peft import (
    get_peft_model, 
    BOFTConfig,
    LoraConfig, 
    TaskType
)

from data import accelerator,tokenizer,compute_metrics,train_dataset,test_dataset, WeightedTrainer


timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
SAVE_PATH=os.path.join("lora_binding_sites", f"esm2_t6_8M_UR50D_lora_{timestamp}")


# Define Custom Trainer Class
def train_function_no_sweeps(train_dataset, test_dataset):
    
    # Set the LoRA config
    config = {
        "lora_alpha": 1, #try 0.5, 1, 2, ..., 16
        "lora_dropout": 0.2,
        "lr": 5.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 3,
        # "per_device_train_batch_size": 12,
        "per_device_train_batch_size": 3,
        "r": 2,
        "weight_decay": 0.2,
        # Add other hyperparameters as needed
    }
    # The base model you will train a LoRA on top of
    #model_checkpoint = "facebook/esm2_t12_35M_UR50D"  
    model_checkpoint = "facebook/esm2_t6_8M_UR50D" 
    # Define labels and model
    id2label = {0: "No binding site", 1: "Binding site"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    
    


    # Convert the model into a PeftModel
    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=config["r"], 
        lora_alpha=config["lora_alpha"], 
        target_modules=["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
        lora_dropout=config["lora_dropout"], 
        bias="none" # or "all" or "lora_only" 
    )
    model = get_peft_model(model, lora_config)

    for param in model.parameters():
        param.requires_grad = False
    
    num_unfreeze=1
    for i in range(num_unfreeze):
        for param in model.encoder.layer[-i-1].parameters():
            param.requires_grad=True
    
    # boft_config = BOFTConfig(
    #     boft_block_size=4,
    #     boft_n_butterfly_factor=2,
    #     target_modules=["query", "value", "key"],#"output.dense", "mlp.fc1", "mlp.fc2"],
    #     boft_dropout=0.1,
    #     bias="boft_only",
    #     modules_to_save=["classifier"],
    # )

    model = get_peft_model(model, lora_config)
    #model = get_peft_model(model, boft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)


    # Training setup
    training_args = TrainingArguments(
        #output_dir=f"esm2_t12_35M-lora-binding-sites_{timestamp}",
        output_dir=f"esm2_t6_8M_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1,
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=7,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb'
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and Save Model
    trainer.train()
    save_path = SAVE_PATH #os.path.join("lora_binding_sites", f"esm2_t6_8M_UR50D_lora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)


train_function_no_sweeps(train_dataset, test_dataset)