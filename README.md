# Play Atari with Reinforcement Learning using Proximal Policy Optimization (PPO)

Let's use [PPO](https://openai.com/blog/openai-baselines-ppo) from OpenAI to play Atari, a famous video game from 80's.

So, it spend a long time since I first created these codes, but, at that time, I didn't spend mutch effort in documentation, then, I decided to run the scripts again and write some comments about it.

If you already tried to run some old Python program after some time, time enough to Python change its version, you already know that the idea above will not work. 

So, let`s put these code to work, again!

I decided to work in a solution to avoid the same problem in the future. Of course, Docker was my first shot. This is because, besides Python libraries, you need to install others packages to make OpenAI Gym run smoothly.

# Setup

All the command is based in Linux. Sorry (or not).

## GPU
Since we are using GPU to accelerate the model train process, make sure all GPU drivers in your computer is fine. You can do that with the command:

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

To enable GPU inside a Docker container, it's necessary to install "NVIDIA Container Toolkit". To help you with that, check this site: 

https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

For Ubuntu 20.04, this website also was very usefull for me: 

https://www.server-world.info/en/note?os=Ubuntu_20.04&p=nvidia&f=2


To make sure the GPU is working inside a Docker container, you can run the follow Docker image from NVIDIA:

```
$ sudo docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

If you are a luck guy, You will see the same GPU configurator showed with the command "nvidia-smi" above.

Nowadays I feel the GPU and CUDA instalation is not too painfull as before.


## Docker Image

You can either, to  pull the Docker image from DockerHub or build it by your own.

Let's start building a local Docker image from local Dockerfile. Clone this repo and go inside de repo folder, where you can see the file called "Dockerfile". This is the file that Docker use to build images. It's a good idea to take a look in this file, there are some usefull comments in it.

```
$ docker build -t atari-ppo .
```

## Run Docker image that you just built in interactive mode.

We sugest to map your local directory into the container, if you don't want to use your local files, you will see the project files inside the container anyway, because we clone it from GitHub at the building time. However, if you intend to change some code, it's better to map your local files to avoid to loose your work if happens the Docker crashs. Don't forget to enable GPU inside container with the parameter "--gpus all".

```
$ docker run --gpus all -it --rm -v $PWD:/workspace atari-ppo
```

To see if Reinforcement Learning is playing Atari, run the the command below, inside of container:

```
python src/ppo_play.py -s checkpoints/BreakoutNoFrameskip-v4_best_+411.100_7188480.dat
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


