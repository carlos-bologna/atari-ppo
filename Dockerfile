# Use nvidia/cuda image
FROM nvcr.io/nvidia/pytorch:21.06-py3

# Update OS packages
RUN apt-get update -y

# Some packages need timezone to be installed.
# Let's run the following command to set it beforehand and avoid those 
# packages ask us at installation time (because it can broke Docker build pipeline)
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata

# Install cv2 prerequisites and unrar
RUN apt-get -y install ffmpeg libsm6 libxext6 unrar && \
    apt-get clean

# Update base environment of Conda
#RUN conda update -n base conda # <<<< Please, don't run that. It will break Pytorch.

# Copy requirements file to Docker image
COPY ./requirements.yaml /tmp/requirements.yaml

# Update base environment Conda in base Docker image with new packages
RUN conda env update -f /tmp/requirements.yaml

# In order to use Atari environment with OpenAI Gym, you need install Atari ROMS
# See: https://github.com/openai/atari-py#roms for more informations
RUN wget http://www.atarimania.com/roms/Roms.rar -P /tmp && \
    unrar x -r /tmp/Roms.rar /tmp && \
    unzip /tmp/ROMS.zip -d /tmp && \
    python -m atari_py.import_roms /tmp/ROMS

# Clone Git repo (into home dir)
RUN git clone https://github.com/carlos-bologna/atari-ppo.git

ENTRYPOINT ["python", "src/ppo_train.py"]