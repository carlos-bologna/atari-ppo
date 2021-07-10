# Use nvidia/cuda image
FROM nvidia/cuda:11.4.0-devel-ubuntu20.04

# Set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Update OS packages
RUN apt-get update

# Install packages 
# wget: to install Anaconda
# git: to clone GitHub project repository
# libglib2.0-0: to make cv2 Python lib to work
RUN apt-get install -y wget git libglib2.0-0  && \
    apt-get clean

# Install Anaconda prerequisites
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 && \
    apt-get clean

# Install Anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh -O ~/anaconda.sh && \
    /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

# Add Anaconda into path
ENV PATH /opt/conda/bin:$PATH

# Install all needed Python libs into conda virtual environment
COPY ./requirements.yaml /tmp/requirements.yaml
RUN conda update conda \
    && conda env create -f /tmp/requirements.yaml

# Activate Conda environment
RUN echo "conda activate atari-ppo" >> ~/.bashrc

# Add Conda env into path
ENV PATH /opt/conda/envs/atari-ppo/bin:$PATH

# Set "home" dir as work dir
WORKDIR /home

# Clone Git repo (into home dir)
RUN git clone https://github.com/carlos-bologna/atari-ppo.git

# Set project directory as work dir
WORKDIR /home/atari-ppo

ENTRYPOINT ["python", "src/setup_test.py"]
