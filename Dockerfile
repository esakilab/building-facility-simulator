FROM python:3.9

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.21.0
RUN pip3 install torch torchvision torchaudio tensorboard

COPY ./input_xmls /work/input_xmls

COPY ./simulator /work/simulator
COPY ./rl /work/rl
COPY ./distributed_platform /work/distributed_platform
COPY ./*.py /work

WORKDIR /work