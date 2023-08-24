import pandas as pd
from transformers import AutoTokenizer, pipeline
from datasets import load_dataset
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
import datasets
import os
os.environ["WANDB_DISABLED"] = "true"

# df_train = pd.read_csv("../data/train_df.csv")
# df_test = pd.read_csv("../data/test_df.csv")
#
# ds_train = Dataset.from_pandas(df_train)
# ds_test = Dataset.from_pandas(df_test)
#
# dataset = datasets.DatasetDict({"train":ds_train,"test":ds_test})
#
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
# def tokenize_function(examples):
#     return tokenizer(examples["text"], padding="max_length", truncation=True)
#
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
#
# # small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
# # small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
#
# split_train_dataset = tokenized_datasets["train"].train_test_split(test_size=0.2)
#
# train_dataset = split_train_dataset["train"]
# eval_dataset = split_train_dataset["test"]
# test_dataset = tokenized_datasets["test"]
#
# print(train_dataset)
# print(eval_dataset)
# print(test_dataset)
#
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
# training_args = TrainingArguments(output_dir="test_trainer", report_to=None)
# metric = evaluate.load("accuracy")
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch", report_to=None, num_train_epochs= 10)
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics,)
#
# trainer.train()
# trainer.save_model("../models/training_model_bert_full_data")
#
# predictions = trainer.predict(test_dataset)

#Loading the model
model = AutoModelForSequenceClassification.from_pretrained("../../models/training_model_bert_full_data")
print("Model loaded successfully")

#Inference from bert model
classifier = pipeline(
    task="text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0)

print(classifier("My friend "))

# 1 is plural
# 0 is singular


