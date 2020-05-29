# Use the run configuration
# this will create new image with tag vastai/pytorch:alfred
FROM vastai/pytorch:latest

# output the OS version
RUN cat /etc/os-release

# create environment
RUN conda create --name cs230_spring2020
RUN activate cs230_spring2020
# fix conda to show up in the shell
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# install various libraries
# -------------------
RUN pip install transformers
RUN conda install pandas
#RUN conda install -c conda-forge openssh
RUN pip install sklearn
RUN git clone https://github.com/NVIDIA/apex
RUN cd apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# Setting up the ssh server for remote debugging (how does vast.ai do their ssh server??)
# ----------------------
RUN apt-get update

RUN apt-get install sudo

RUN apt-get install -y openssh-server

#ENTRYPOINT ["/workspace/cs230_spring2020/final_project/Docker/docker-entrypoint.sh"]
RUN /bin/bash -c "echo 'root:cs230stanford' | chpasswd"
RUN /bin/bash -c "sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config"
## SSH login fix. Otherwise user is kicked off after login
RUN /bin/bash -c "sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd"
#ENV NOTVISIBLE "in users profile"
#RUN echo "export VISIBLE=now" >> /etc/profile

# this line ends up having no effect since the service status is ephemeral (i.e. needs to get started again when the container is created)
# i tried putting this in an entrypoint, but then the container would always exit (i.e. even though sshd was running in the background, the entrypoint command would exit and then the docker container would exit as well)
RUN /bin/bash -c "service ssh start"

# this command is for information purposes only (i.e. needs to be called out in the docker run command)
EXPOSE 22