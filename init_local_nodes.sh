#!/bin/bash

while read -u 10 local_server; do

    # Initial SSH connection
    expect -c "
        set timeout 5
        spawn /usr/bin/ssh -t ${local_server}
        expect {
            \"yes/no\" {
                send \"yes\r\"
                exp_continue
            }
            \"New password: \" {
                send \"password\r\"
                exp_continue
            }
            \"Retype new password: \" {
                send \"password\r\"
            }
            \"\$\" {}
        }
        exit 0
    "
    
    # install docker
    ssh $local_server '
        set -o errexit
        set -o nounset

        IFS=$(printf '\n\t')

        # Docker
        sudo apt remove --yes docker docker-engine docker.io containerd runc || true
        sudo apt update
        sudo apt --yes --no-install-recommends install apt-transport-https ca-certificates
        wget --quiet --output-document=- https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository --yes "deb [arch=$(dpkg --print-architecture)] https://download.docker.com/linux/ubuntu $(lsb_release --codename --short) stable"
        sudo apt update
        sudo apt --yes --no-install-recommends install docker-ce docker-ce-cli containerd.io
        sudo usermod --append --groups docker "$USER"
        sudo systemctl enable docker
        printf "\nDocker installed successfully\n\n"
    ' &
done 10< distributed_platform/local-server-hostnames
wait

