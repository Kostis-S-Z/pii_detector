ARG DEVICE=gpu

FROM pytorch/torchserve:latest-${DEVICE}

RUN pip3 install transformers datasets seqeval

ARG MODEL_NAME="pii_detector"

COPY serve.py /home/model-server/
COPY data.py /home/model-server/
COPY ./${MODEL_NAME}/config.json /home/model-server/
COPY ./${MODEL_NAME}/pytorch_model.bin /home/model-server/
COPY ./${MODEL_NAME}/index_to_label.json /home/model-server/

# create torchserve configuration file
USER root
RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
RUN printf "\ninference_address=http://0.0.0.0:7080" >> /home/model-server/config.properties
RUN printf "\nmanagement_address=http://0.0.0.0:7081" >> /home/model-server/config.properties

# expose health and prediction listener ports from the image
EXPOSE 7080
EXPOSE 7081

# create model archive file packaging model artifacts and dependencies
RUN torch-model-archiver -f \
  --model-name=pii_detector \
  --version=1.0 \
  --serialized-file=/home/model-server/pytorch_model.bin \
  --handler=/home/model-server/serve.py \
  --extra-files "/home/model-server/data.py,/home/model-server/index_to_label.json,/home/model-server/config.json" \
  --export-path=/home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ts-config=/home/model-server/config.properties", \
     "--models pii_detector.mar", \
     "--model-store /home/model-server/model-store"]