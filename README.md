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
python3 main.py
```

## クラスタでの実行
グローバルモデルを管理するグローバルサーバ（1台）と、ローカルモデルの実行を行うローカルサーバ（1台以上）
を用意することで、
準備段階として、
1. グローバルサーバから各ローカルサーバにssh接続できるようにする
1. グローバルサーバ上で本レポジトリをcloneし、`distributed_platform/local-server-hostnames`に全ローカルサーバでのユーザ名とホスト名をsshコマンドと同じ形式（`${user_name}@${host_name}`）で書き込む
1. グローバルサーバで以下のコマンドを実行し、各ローカルサーバにdockerとdocker-composeをインストールする
```bash
$ ./install_docker_to_local_servers.sh
```

を完了した上で、グローバルサーバ上で
```
$ docker-compose up -d global
$ ./start_local_servers.sh
$ docker-compose logs -f
```
とすると、全てのサーバを立ち上げた上でグローバルサーバのログを見ることができる。

各ローカルサーバで実行されているモデルたちの状態は、`logs/`に蓄積され、tensorboard上で確認することが可能である。