FROM python:3.9

RUN pip3 install --upgrade pip
RUN pip3 install numpy==1.21.0
RUN pip3 install pydantic==1.9.0
RUN pip3 install tensorboard==2.11.0 torch==1.11.0 torchaudio==0.11.0 torchvision==0.12.0

COPY . /work

WORKDIR /work
