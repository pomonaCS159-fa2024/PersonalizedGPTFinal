from datasets import load_from_disk, Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import json
import os
import pandas as pd

# Load and prepare the dataset
def prepare_dataset(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return [{"text": f"{item['prompt']} {item['response']}"} for item in data]

# Save dataset as Hugging Face dataset
def save_to_dataset(data, save_path):
    df = pd.DataFrame(data)
    os.makedirs(save_path, exist_ok=True)
    dataset = Dataset.from_pandas(df)
    dataset.save_to_disk(save_path)

# File paths
dataset_path = "/Users/avga2021/Desktop/NLP_Final2/NLPData-3.json"  # Path to the uploaded JSON dataset
processed_data_path = "processed_data"  # Directory to save processed data

# Prepare and save the dataset
data = prepare_dataset(dataset_path)
save_to_dataset(data, processed_data_path)

# Load the processed dataset
dataset = load_from_disk(processed_data_path)

# Initialize tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Add a padding token if not present
tokenizer.pad_token = tokenizer.eos_token  # Use the eos_token as pad_token

# Tokenize the dataset
def tokenize_function(examples):
    tokenized_output = tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=512
    )
    # Add labels identical to input_ids for language modeling
    tokenized_output["labels"] = tokenized_output["input_ids"].copy()
    return tokenized_output

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Prepare dataset for training
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
tokenized_datasets.set_format("torch")

# Split the dataset into train and validation sets
split_datasets = tokenized_datasets.train_test_split(test_size=0.1)  # Use 10% for validation
train_dataset = split_datasets["train"]
eval_dataset = split_datasets["test"]

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",         # Directory to save model checkpoints
    evaluation_strategy="epoch",   # Evaluate at the end of each epoch
    learning_rate=2e-5,            # Learning rate
    per_device_train_batch_size=2, # Batch size
    num_train_epochs=3,            # Number of epochs
    weight_decay=0.01,             # Weight decay
    save_strategy="epoch",         # Save checkpoints at each epoch
    logging_dir='./logs',          # Logging directory
    logging_steps=10,              # Log every 10 steps
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset for validation
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_gpt2.1")
tokenizer.save_pretrained("./fine_tuned_gpt2.1")
