#!/usr/bin/env bash

#  The /dev/shm line keeps pytorch dataloader from crashing.
#  It should be done via --shm-size param in docker run, but AWS Batch doesn't let us touch that

COMMAND="
    source activate RDB;
    mkdir RDB_code;
    sudo mount -o remount,size=50G /dev/shm ;
    sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <your_AWS_EFS_name>:/ ./RDB_code ;
    cd RDB_code;
    $1
    "

echo $COMMAND

nvidia-docker run --privileged -it --mount type=bind,src=<data_root>,dst=/root/RDB_data <the_id_of_the_docker_image_that_you_build_from_docker/whole_project/Dockerfile> /bin/bash -c "$COMMAND"
