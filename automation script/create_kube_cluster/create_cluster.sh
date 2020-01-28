#!/bin/bash
# create kube cluster automatically
# command: create_cluster network

arg_num=$#
if [ $arg_num -eq 0 ]; then
    network=flannel
elif [$arg_num -eq 1]; then
    network=$1
else
    echo "command: create_cluster [network]"
    return
fi

create(){
    echo start creating...
    echo "type of network: $network"
    echo "delete old cluster..."
    sudo swapoff -a 
    sudo kubeadm reset -f && sudo rm -r /etc/cni/net.d && rm $HOME/.kube/config

    if [ $network == 'flannel' ]; then
        sudo kubeadm init --pod-network-cidr=10.244.0.0/16 

        while true; do
            if test -e /etc/kubernetes/admin.conf; then
                sleep 5
                mkdir -p $HOME/.kube 
                sudo cp /etc/kubernetes/admin.conf $HOME/.kube/config
                sudo chown -R 1000:1000 $HOME/.kube/config 
                kubectl apply -f flannel.yaml
                break
            fi
            sleep 5

        done
        
    else
        echo type of nework $network is still not implemented
    fi
}

create
