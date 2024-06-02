### TL;DR How-to

0. Create a venv and install the dependencies

```
python3 -m venv ./venv
source venv/bin/activate
pip install -r requirements.txt
```

1. Run train_detector.py locally or on a server to generate the fine-tuned model (make sure venv is active). 

```
python3 train_detector.py
```

2. Add your HF access token to config.json in the model directory
```
"hf_token": "hf_XXX"
```

3. Build image with Dockerfile. Supply as build arg DEVICE=cpu or DEVICE=gpu depending on the device you want to run it on.

_**Note**: When initializing the TorchServe handler on a GPU, it will run a search for the optimal batch size on an inference dataset. By default, this is the same dataset as used for training the model. To change this, either update the `train_detector.py` script before training or manually update the field `dataset_name` with a HF dataset ID in the config.json in the `pii_detector` directory with the dataset of your choice. Note however that the dataset you provide needs to follow the same format as the one used for training._ 

4. Deploy image to cloud of your choice for inference

5. Test server with
```
curl -X POST http://localhost:7080/predictions/pii_detector -d '["My name is Bob"]'
```


