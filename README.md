# Play Atari with Reinforcement Learning using Proximal Policy Optimization (PPO)

Let's use [PPO](https://openai.com/blog/openai-baselines-ppo) from OpenAI to play Atari, a famous video game from 80's. AGAIN.

So, it spend a long time since I first created this codes, but, at tha time, I didn't spend mutch time in documentation, then, I decided to run those scripts again and write some comments about it.

Note: if you already tried to run some old Python after some time, time enough to Python change its version, you already know that the idea above will not work. 

So, let`s put these code to work. Again!

I decided to work in a solution to avoid the same problem in the future. Of course, Docker was my first shot.

# Setup

All the command is based in Linux. Sorry (or not)

## GPU
Since we are using GPU to accelerate the model train process, make sure all GPU drivers in your computer is fine. You can do that with the command

```
$ nvidia-smi

+-----------------------------------------------------------------------------+
| NVIDIA-SMI 460.73.01    Driver Version: 460.73.01    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  GeForce GTX 1060    Off  | 00000000:01:00.0 Off |                  N/A |
| N/A   59C    P3    17W /  N/A |    464MiB /  6078MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A       919      G   /usr/lib/xorg/Xorg                 45MiB |
|    0   N/A  N/A    114605      G   /usr/lib/xorg/Xorg                189MiB |
|    0   N/A  N/A    114774      G   /usr/bin/gnome-shell               34MiB |
|    0   N/A  N/A    116119      G   /usr/lib/firefox/firefox          146MiB |
|    0   N/A  N/A    129253      G   /usr/lib/firefox/firefox            1MiB |
|    0   N/A  N/A    163852      G   ...AAAAAAAAA= --shared-files       30MiB |
+-----------------------------------------------------------------------------+

```

To enable GPU inside a Docker container, it necessary to install "NVIDIA Container Toolkit". To help you with that, check this site: 

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

For Ubuntu 20.04, that other website also was very usefull for me, check it out: 

https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia&f=2


To make sure the GPU is working inside a Docker container, NVIDIA has a simple test for us. Run this Docker image:

```
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If you are a luck guy, You will see the same GPU configurator showed with the command "nvidia-smi" above.

Nowadays I feel the GPU and CUDA instalation is not too painfull as before.


## Docker Image

You can either, to  pull the Docker image from DockerHub or build it by your own.

Let's start building a local Docker image from local Dockerfile. Please, run these commands:

```
$ docker build -t atari-ppo .
```





### Using Python Virtual Environment

Another option is trying to use Python virtual environment to reproduce some work whenever you want, but it does not work if it spend time enough to Python change its version. At least it didn't work for me in this case.

If you are willing to try, below are some usefull commands.


```
$ sudo apt install virtualenv
$ virtualenv myenv --python=python3.7
$ source myenv/bin/activate #to activated the environment

$ deactivate #to deactivate it whenever you want

$ pip3 install -r requirements.txt

```


