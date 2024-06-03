import json

import torch
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)

from data import PIIDataset, compute_metric_custom

print(
    f'Running on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}'
)

dataset_name = "ai4privacy/pii-masking-65k"  # 21_587 total size
inference_dataset_name = dataset_name
model_name = "facebook/opt-350m"  # "distilbert/distilbert-base-uncased"
model_dir = "pii_detector"

train_dataset_size = 17_000
eval_dataset_size = 4_587
max_len = 128
batch_size = 32

tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

dataset = PIIDataset(
    dataset_name, tokenizer, max_len=max_len, slice_end=train_dataset_size
)
index_to_label = dataset.index_to_label
label_to_index = dataset.label_to_index

eval_dataset = PIIDataset(
    dataset_name,
    tokenizer,
    max_len=max_len,
    slice_start=train_dataset_size,
    slice_end=train_dataset_size + eval_dataset_size,
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=dataset.num_labels,
    id2label=index_to_label,
    label2id=label_to_index,
)
data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    per_device_train_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir=model_dir + "/logs",
    logging_steps=10,
)

compute_metrics = compute_metric_custom(index_to_label)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

train_results = trainer.train()
print(train_results)

trainer.save_model(model_dir)
torch.save(model.state_dict(), model_dir + "/pytorch_model.bin")

eval_results = trainer.evaluate(eval_dataset)

eval_results["n_eval_samples"] = len(eval_dataset)
eval_results["eval_samples"] = eval_dataset.dataset["tokenised_unmasked_text"]

print(eval_results)

with open(model_dir + "/config.json") as f:
    updated_config = json.load(f)
    updated_config["model_name"] = model_name
    updated_config["dataset_name"] = inference_dataset_name
    updated_config["max_len"] = max_len
    json.dump(updated_config, f, sort_keys=True, indent=4)
with open(model_dir + "/eval_metrics.json", "w") as fp:
    json.dump(eval_results, fp, sort_keys=True, indent=4)
with open(model_dir + "/label_to_index.json", "w") as fp:
    json.dump(label_to_index, fp, sort_keys=True, indent=4)
with open(model_dir + "/index_to_label.json", "w") as fp:
    json.dump(index_to_label, fp, sort_keys=True, indent=4)
