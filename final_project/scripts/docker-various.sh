# for running docker with GPU support
docker run -it --gpus all -v /home/ec2-user/jupyter:/root/jupyter 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.5.0-gpu-py36-cu101-ubuntu16.04 /bin/bash



# update docker on host to work with nvidia gpu's (appears to be required)
# https://github.com/NVIDIA/nvidia-docker/issues/1243
# -------------------
# setup nvidia-docker
# https://github.com/NVIDIA/nvidia-docker/README.md
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) &&\
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.repo | sudo tee /etc/yum.repos.d/nvidia-docker.repo &&\
sudo yum install -y nvidia-container-toolkit &&\
sudo systemctl restart docker