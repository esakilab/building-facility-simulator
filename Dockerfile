FROM python:3.9

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.21.0
RUN pip3 install torch torchvision torchaudio tensorboard
RUN pip3 install pydantic==1.9.0

# COPY ./data/json /work/data/json

WORKDIR /work

COPY ./simulator /work/simulator
COPY ./rl /work/rl
COPY ./distributed_platform /work/distributed_platform
COPY ./*.py /work/
