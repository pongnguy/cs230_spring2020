# -------
# Installing apex and some libraries to the main enviroment.
# Could not figure out how to install in conda environment.
# Goal was to get apex installed fully and working.
# Should link to python interpreter outside of conda env for the PyCharm remote interpreter.
#
# -----

# Use the run configuration
# this will create new image with tag vastai/pytorch:alfred
FROM vastai/pytorch:latest

# output the OS version
RUN cat /etc/os-release &&\
echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# create environment
#RUN conda create --name cs230_spring2020
#RUN activate cs230_spring2020
# fix conda to show up in the shell
#RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# input environment from AWS server
# -------------------
#CMD ["cd /tmp_workspace/cs230_spring2020/final_project/"]
#RUN ["/bin/bash", "-c", "cd /workspace/cs230_spring2020/final_project/"]
#RUN cd "/workspace/cs230_spring2020/final_project/"
#CMD ["conda create --file aws_neuron_pytorch_p36.yml --name cs230_spring2020_aws"]
# import environment from aws server (minus apex=0.1 which throws a constraints error)
# volume binding does not exist yet, need to copy in file from build context
COPY "aws_neuron_pytorch_p36.yml" "/tmp_workspace/cs230_spring2020/final_project/"
# correct syntax "conda env create" instead of "conda create "
# added "|| true" since it still seems to work even though it throws up an error
RUN conda env create --file "/tmp_workspace/cs230_spring2020/final_project/aws_neuron_pytorch_p36.yml" --name cs230_spring2020_aws || true

#RUN echo "source activate cs230_spring2020_aws" > ~/.bashrc
#ENV PATH /opt/conda/envs/cs230_spring2020_aws/bin:$PATH
#RUN /bin/bash -c "echo '. /opt/conda/etc/profile.d/conda.sh' >> ~/.bashrc" &&\
#conda activate
#SHELL ["conda", "run", "-n", "cs230_spring2020_aws", "/bin/bash", "-c"]
#SHELL ["conda", "run", "-n", "cs230_spring2020_aws"]
#SHELL ["conda", "run",  "-n cs230_spring2020_aws"]
#SHELL ["/bin/bash", "-c"]
#RUN activate cs230_spring2020_aws &&\
#conda install pip

#RUN activate cs230_spring2020_aws




#RUN /bin/bash -c "source activate cs230_spring2020_aws && pip install transforms"


# this is from pytorch website (vast.ai docker image uses CUDA 10.0)
#RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
# installing pytorch from source
# https://github.com/pytorch/pytorch#from-source
# -----------------
#RUN conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi
#RUN conda install -c pytorch magma-cuda100
#RUN cd /pytorch
# updating repo
#RUN git pull
#RUN git submodule sync
#RUN git submodule update --init --recursive
#RUN export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#RUN python setup.py install

# install pytorch into the environment
#RUN conda activate cs230_spring2020_aws &&\
RUN conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch


# install various libraries
# -------------------
#RUN activate cs230_spring2020_aws &&\
#RUN conda activate cs230_spring2020_aws &&\
RUN pip install transformers &&\
conda install pandas &&\
pip install sklearn
#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex
#RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# Setting up the ssh server for remote debugging (how does vast.ai do their ssh server??)
# ----------------------
RUN apt-get update &&\
apt-get install sudo &&\
# instal ssh server
apt-get install -y openssh-server

#ENTRYPOINT ["/workspace/cs230_spring2020/final_project/Docker/docker-entrypoint.sh"]
RUN /bin/bash -c "echo 'root:cs230stanford' | chpasswd"
RUN /bin/bash -c "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config"
## SSH login fix. Otherwise user is kicked off after login
RUN /bin/bash -c "sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd"
# disallow privilege separation
RUN /bin/bash -c "sed -i 's/UsePrivilegeSeparation yes/UsePrivilegeSeparation no/' /etc/ssh/sshd_config"
#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile


WORKDIR /tmp_workspace/
#RUN cd /workspace/apex/
# updating repo
RUN git clone https://github.com/NVIDIA/apex
#RUN git pull
#RUN git submodule sync
#RUN git submodule update --init --recursive
WORKDIR /tmp_workspace/apex
# rollback to earlier version
# https://github.com/NVIDIA/apex/issues/802
#RUN conda activate cs230_spring2020_aws &&\

#RUN git checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0 &&\
#pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' ./
#RUN pip install -v --no-cache-dir ./N


# 2020-05-29 to here

#RUN conda activate cs230_spring2020_aws &&\
RUN pip install -U transformers==2.1.1 &&\
apt install pciutils &&\
#apt install module-init-tools
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.1.85-1_amd64.deb
dpkg --install cuda-repo-ubuntu1604_9.1.85-1_amd64.deb &&\
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub &&\
apt install cuda

#RUN wget http://us.download.nvidia.com/tesla/396.37/nvidia-diag-driver-local-repo-ubuntu1710-396.37_1.0-1_amd64.deb &&\
#sudo dpkg -i nvidia-diag-driver-local-repo-ubuntu1710-396.37_1.0-1_amd64.deb
#RUN apt-get install module-init-tools
#RUN wget http://us.download.nvidia.com/tesla/440.64.00/NVIDIA-Linux-x86_64-440.64.00.run
#RUN wget https://developer.download.nvidia.com/compute/cuda/10.0/secure/Prod/local_installers/cuda_10.0.130_410.48_linux.run?foJfy8jPc58xZLwVZLFPqxUF4br8_XlQcjM4o49r-NPykAYmJ-TatFLfwOFeI33DKAGTHC-Df62J7zpe67bugUCwaJqLcVA2ZxHEATu1Kcz4Fm8kQnn-I6miTvz33luM3BIHycYQhKsC6Mz3D9eTa2sARO6bhBCfYoQWh_vzjTnYoOP9mg14Exv2fMc



# indicates ports where container listens for connections
EXPOSE 22

# this line ends up having no effect since the service status is ephemeral (i.e. needs to get started again when the container is created)
# i tried putting this in an entrypoint, but then the container would always exit (i.e. even though sshd was running in the background, the entrypoint command would exit and then the docker container would exit as well)
#RUN /bin/bash -c "service ssh start"

# looks like this only works at build time, not if you run the container later
# note: if the attached console does not show anything it because another call never detached (i.e. like the one below)
CMD ["/usr/sbin/sshd", "-D"]