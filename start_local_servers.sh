#!/bin/bash

# TODO: ここをconfigurableにする
GLOBAL_HOSTNAME=10.17.104.108

while read -u 10 local_server; do
    ssh $local_server "
        mkdir -p bfs;
    " && \
    find . -mindepth 1 -maxdepth 1 ! -path './data' ! -path './logs' ! -path './.*' -print \
        | xargs -i scp -r {} ${local_server}:~/bfs \
    && \
    ssh $local_server "
        cd ~/bfs && \
        mkdir -p data/json && \
        GLOBAL_HOSTNAME=${GLOBAL_HOSTNAME} docker-compose up --build -d local
    " &
done 10< distributed_platform/local-server-hostnames
wait
