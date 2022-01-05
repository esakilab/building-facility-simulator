#!/bin/bash

INSTALL_SCRIPT=`cat distributed_platform/install_docker.sh`

echo $INSTALL_SCRIPT

while read -u 10 local_server; do
    ssh $local_server "
        ${INSTALL_SCRIPT}
    " &
done 10< distributed_platform/local-server-hostnames
wait

