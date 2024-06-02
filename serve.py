import json
import logging
from typing import Dict, List

import torch
from huggingface_hub._login import _login
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

from data import PIIDataset

logger = logging.getLogger(__name__)


class PIIDetector(BaseHandler):
    """
    Handles replies for a classifier by implementing the TorchServe
    basic handler class and using a model from HuggingFace
    :param basic TorchServe handler (https://github.com/pytorch/serve)
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.max_len = 128
        self.optimal_batch_size = 64
        self.index_to_label = None
        self.config = None
        self.initialized = False

    def initialize(self, context):
        """
        Mandatory initialization method called when the TorchServe server starts
        Loads the necessary components for later processing of the server requests
        :param context: required dict argument that contains two main attributes:
            - manifest: {
                'createdOn': 'DD/MM/YY HH:MM:SS',
                'runtime':'python',
                'model': {
                    'modelName': 'hf_id',
                    'handler': 'serve.py',
                    'modelVersion': 'X.X'
                },
                'archiverVersion': '0.6.0'
                'serializedFile': 'path/to/model.bin'
            }
            - system_properties: {
                'model_dir': '/home/model-server/tmp/models/<some_alphanumeric_str>',
                'gpu_id': None for CPU, 0 (or another int) for GPU,
                'batch_size': 1,
                'server_name': 'MMS',
                'server_version': '0.6.0',
                'limit_max_image_pixels': True
            }
        :raises KeyError whenever 'model_dir' is not found in the system properties
        """
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        with open(model_dir + "/index_to_label.json") as f:
            self.index_to_label = json.load(f)
        with open(model_dir + "/config.json") as f:
            self.config = json.load(f)
        if max_len := self.config.get("max_len"):
            self.max_len = max_len
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id"))
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.get("model_name"), add_prefix_space=True
        )

        # If a dataset was provided in the config, try and find the optimal batch size for the current GPU
        if dataset_name := self.config.get(
            "dataset_name"
        ) and self.device != torch.device("cpu"):
            # Need to log in to HF to download dataset
            _login(token=self.config.get("hf_token"), add_to_git_credential=False)
            dataset = PIIDataset(
                dataset_name, self.tokenizer, max_len=self.max_len, samples_to_use=100
            )
            self.optimal_batch_size = self.find_optimal_batch_size(
                dataset, starting_batch_size=2
            )

        logger.info(
            f"Initialize handler with:"
            f"\n\t Manifest: {json.dumps(context.manifest, sort_keys=True, indent=4)}"
            f"\n\t Properties: {json.dumps(properties, sort_keys=True, indent=4)}"
            f"\n\nInitialized model: {model_dir}"
            f"\nFound optimal batch size: {self.device}"
        )

        self.initialized = True  # mandatory flag for Torch BaseHandler

    def preprocess(self, texts: List[str]) -> Dict:
        """
        Method called whenever the Torch server gets a request. Expects a list of texts.
        :text: the raw texts as sent by the client
        :returns: the preprocessed and tokenized texts
        """
        logger.info(f"Raw texts: {texts}")

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

        return inputs

    def inference(self, inputs: Dict, **kwargs) -> List[List[Dict]]:
        """
        Mandatory method called automatically right after the `preprocess` method.
        Each text is split into tokens, each token is assigned a label.
        :param inputs: the result of the tokenizer
        :returns: the predictions in the shape of [number_of_texts, max_sequence_length, number_of_labels]
        """
        input_ids = inputs["input_ids"].to(self.device)

        outputs = []
        for i in range(0, len(input_ids), self.optimal_batch_size):
            batch_input_ids = input_ids[i : i + self.optimal_batch_size]

            with torch.no_grad():
                batch_output = self.model(batch_input_ids)[0]

            batch_predictions = torch.argmax(batch_output, dim=1).tolist()
            outputs.extend(batch_predictions)

        return outputs

    def postprocess(self, predictions: List[List[Dict]]) -> List[List[str]]:
        """
        Mandatory method called automatically right after the `inference` method.
        Process model outputs before sending it to the client to return the true labels, instead of the indices
        :param predictions: Each token prediction is the index of the label
        :returns: a list of lists of predicted labels for each token for each given sequence
        """
        text_labels = []
        for text_prediction in predictions:
            token_labels = []
            for token_prediction in text_prediction:
                token_labels.append(self.index_to_label.get(str(token_prediction)))
            text_labels.append(token_labels)

        return text_labels

    def find_optimal_batch_size(
        self, dataset: PIIDataset, starting_batch_size: int = 8_192
    ) -> int:
        """
        Brute force method to find the optimal batch size for the model for the currently selected GPU and the dataset
        provided by the config of the user. It will run inference with a very big batch size, expecting an OOM error
        and will gradually decrease it until the whole dataset can pass through.
        """
        logger.info(
            f'Trying to find optimal batch size for dataset: {self.config.get("dataset_name")}'
            f"and GPU: {torch.cuda.get_device_name(0)}"
        )

        batch_size = starting_batch_size
        optimal_batch_size = 0

        while batch_size > 2 and not optimal_batch_size:
            logger.info(f"Trying batch size: {batch_size}")
            try:
                data_loader = DataLoader(dataset, batch_size=batch_size)

                for batch in data_loader:
                    inputs = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    self.model(inputs, attention_mask=attention_mask)

                optimal_batch_size = batch_size
                logger.info(f"Optimal batch size is {optimal_batch_size}")

            except RuntimeError as e:
                if "out of memory" in str(e) or "OOM" in str(e):
                    logger.info(f"Batch size: {batch_size} too big for current GPU")
                    batch_size = int(batch_size / 2)
                    pass
                else:
                    raise e

        # Clear the GPU memory
        torch.cuda.empty_cache()

        return optimal_batch_size
