# -------
# Installing apex and some libraries to the main enviroment.
# Could not figure out how to install in conda environment.
# Goal was to get apex installed fully and working.
# Should link to python interpreter outside of conda env for the PyCharm remote interpreter.
#
# -----

# Use the run configuration
# this will create new image with tag vastai/pytorch:alfred
# this is Ubuntu 16.04
FROM vastai/pytorch:latest


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

# indicates ports where container listens for connections
EXPOSE 22

# installs in conda base environment (even without activate/source or calling pip from the specific directory)
WORKDIR /tmp_workspace/
RUN git clone https://github.com/NVIDIA/apex &&\
# execute git as if inside apex
git -C ./apex checkout f3a960f80244cf9e80558ab30f7f7e8cbf03c0a0
#RUN /bin/bash -c "source activate base && pip install -v --no-cache-dir --global-option='--cpp_ext' --global-option='--cuda_ext' ./apex"
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex

# upgrade anaconda (adds conda run)
# downgrade python back to original version (causes apex to show up again)
RUN conda update conda &&\
conda install python=3.7.1

RUN pip install transformers==2.1.1 &&\
conda install pandas &&\
pip install sklearn


# looks like this only works at build time, not if you run the container later
# note: if the attached console does not show anything it because another call never detached (i.e. like the one below)
CMD ["/usr/sbin/sshd", "-D"]