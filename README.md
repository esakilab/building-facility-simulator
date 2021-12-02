# building-facility-simulator

2050年代のビル設備を想定した、ビルの運転シミュレータ

仕様: https://docs.google.com/document/d/1mGKpOzYdUQ-Lv8eJkvpL64GG0WTpLpSUq-EXHNt34hA/edit?usp=sharing

```
$ path/to/python main.py
```

## requirements
```
python>=3.9.0
```

1. conda, pyenvなどのお好みのpython環境を用意
2. まず以下を実行
```
pip3 install numpy=1.21.0

```
3. そのあとLinuxなら
```
pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
Mac/Windowsなら
```
pip3 install torch torchvision torchaudio
```
を実行

4. 以下のコマンドを実行することで実行可能
```
python3 main_rl.py
```
## GPUを使う上での注意
* GPUサーバー(username@gpu.hongo.wide.ad.jp)にssh公開鍵を登録しssh接続を可能とする.
* pyenvやcondaなどでPython3.9.0以上をインストールし以下のコマンドからPytorchをインストール. (https://pytorch.org/)
```
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```
* tensorboardを使う時は以下を入力
```
pip3 install tensorboard
```
* GPUはcuda:0とcuda:1の2枚あり以下のコマンドを用いて使用状況を確認する.(使われているcudaでコードを実行させるとout of memoryで落ちます)
```
nvidia-smi
```
* federated learning用のコードfed_avg.pyの実行は以下の通り (ビルの数3 cuda:0を使用する時の例)
```
python3 fed_avg.py --builging_num 3 --cuda_name cuda:0
```
* またxmlデータは個々人でサーバに保存すると無駄があるので現状 '/home/kfujita/data' 以下にあります. 
* 1つのcudaで2つくらいの実験は同時に走らせられるのでtmuxなどを用いて複数ターミナルから同時に実行可能です.