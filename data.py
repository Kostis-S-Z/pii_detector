from typing import Dict

import numpy as np
import torch
from datasets import load_dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer


class PIIDataset(TorchDataset):
    """Use a TorchDataset wrapper to properly handle retrieving samples from this specific dataset"""

    def __init__(
        self,
        hf_dataset_id: str,
        tokenizer: PreTrainedTokenizer,
        max_len: int,
        samples_to_use: int = None,
        slice_start: int = 0,
        slice_end: int = -1,
    ):
        if samples_to_use:
            self.dataset = load_dataset(
                hf_dataset_id, split=f"train[:{samples_to_use}]"
            )
        else:
            self.dataset = load_dataset(
                hf_dataset_id, split=f"train[{slice_start}:{slice_end}]"
            )
        self.tokenizer = tokenizer
        self.max_len = max_len

        unique_labels = set()
        for sample in self.dataset["token_entity_labels"]:
            unique_labels.update(sample)

        self.label_to_index = {label: i for i, label in enumerate(unique_labels)}
        self.index_to_label = {i: label for i, label in enumerate(unique_labels)}
        self.num_labels = len(self.label_to_index)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        tokens = self.dataset["tokenised_unmasked_text"][index]
        token_labels = self.dataset["token_entity_labels"][index]

        encoding = self.tokenizer(
            tokens,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_attention_mask=True,
            is_split_into_words=True,
        )

        labels = torch.ones((self.max_len,), dtype=torch.long) * -100

        for i, word_label in enumerate(token_labels[: self.max_len]):
            labels[i] = self.label_to_index[word_label]

        sample = {
            "input_ids": encoding["input_ids"].view(-1),
            "attention_mask": encoding["attention_mask"].view(-1),
            "labels": labels,
        }
        return sample

    def __len__(self) -> int:
        return len(self.dataset)


def compute_metric_custom(index_to_label: Dict[int, str]):
    def compute_metric(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [[index_to_label[label]] for label in labels.flatten()]
        true_labels = [[index_to_label[label]] for label in predictions.flatten()]

        precision = precision_score(true_labels, true_predictions, scheme=IOB2)
        recall = recall_score(true_labels, true_predictions, scheme=IOB2)
        f1 = f1_score(true_labels, true_predictions, scheme=IOB2)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_labels": true_labels,
            "true_predictions": true_predictions,
        }

    return compute_metric
