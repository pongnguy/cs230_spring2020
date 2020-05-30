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


#RUN echo "source activate cs230_spring2020_aws" > ~/.bashrc
#ENV PATH /opt/conda/envs/cs230_spring2020_aws/bin:$PATH


#RUN echo "source activate cs230_spring2020_aws" &&
RUN /opt/conda/bin/pip install transformers
#RUN /bin/bash -c "source activate cs230_spring2020_aws && pip install transforms"

#RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc &&\
#RUN source activate cs230_spring2020_aws &&\

#RUN conda run -n cs230_spring2020_aws /bin/bash -c "conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch"

#RUN /bin/bash -c "source activate base &&\
#conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch"

RUN git clone https://github.com/NVIDIA/apex &&\
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex


# indicates ports where container listens for connections
EXPOSE 22

# this line ends up having no effect since the service status is ephemeral (i.e. needs to get started again when the container is created)
# i tried putting this in an entrypoint, but then the container would always exit (i.e. even though sshd was running in the background, the entrypoint command would exit and then the docker container would exit as well)
#RUN /bin/bash -c "service ssh start"

# looks like this only works at build time, not if you run the container later
# note: if the attached console does not show anything it because another call never detached (i.e. like the one below)
CMD ["/usr/sbin/sshd", "-D"]