import base64
import os
import pickle
import subprocess
import sys
from pprint import pprint

import boto3

local_docker_image_id = '<the_id_of_the_docker_image_that_you_build_from_docker/whole_project/Dockerfile>'


def run_script_with_kwargs(script_name, kwargs, session_name, locale='local_tmux', n_gpu=0, n_cpu=1, mb_memory=1024):
    enc_kwargs = base64.b64encode(pickle.dumps(kwargs)).decode()
    container_cmd = f'source activate RDB; \
                      export CUDA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES; \
                      mkdir RDB_code; \
                      sudo mount -o remount,size=100G /dev/shm ;\
                      sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport <your_AWS_EFS_name>:/ ./RDB_code ; \
                      cd RDB_code; \
                      python -m {script_name} {enc_kwargs};'
    if locale == 'local_tmux':
        subprocess.run("tmux new-session -d -s {}".format(session_name), shell=True)
        subprocess.run("tmux send-keys -t {} 'sudo {} -m {} {}' C-m ".format(session_name,
                                                                             sys.executable,
                                                                             script_name,
                                                                             base64.b64encode(
                                                                                 pickle.dumps(kwargs)).decode()),
                       shell=True,
                       env=os.environ.copy())
        subprocess.run("tmux send-keys -t {} 'exit' C-m ".format(session_name), shell=True)

    elif locale == 'local_docker':
        subprocess.run("tmux new-session -d -s {}".format(session_name), shell=True)
        subprocess.run(
            "tmux send-keys -t {} 'nvidia-docker run --privileged {} /bin/bash -c \" {} \" ' C-m ".format(session_name,
                                                                                                          local_docker_image_id,
                                                                                                          container_cmd),
            shell=True,
            env=os.environ.copy())

    elif locale == 'local_tmux':
        client = boto3.client('batch')
        command = ['/bin/bash', '-c', 'Ref::container_cmd']
        job = dict(
            jobName=session_name,
            jobQueue='<your_AWS_Batch_job_queue_name>',
            jobDefinition='<your_AWS_Batch_job_definition_name>',
            containerOverrides=dict(
                vcpus=n_cpu,
                memory=mb_memory,
                command=command,
                resourceRequirements=[dict(value=str(n_gpu), type='GPU')] if n_gpu else []
            ),
            parameters=dict(container_cmd=container_cmd)
        )
        response = client.submit_job(**job)
        pprint(response)

    else:
        raise ValueError('locale not recognized')
