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
* 今のところ標準ライブラリのみで動いています

## reinforcement learning
1. conda, pyenvなどのお好みのpython環境を用意
2. まず以下を実行
```
pip install numpy=1.21.0

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
