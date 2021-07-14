import argparse
import gym
#import roboschool
#from lib.model import ActorCritic
import numpy as np
import torch
import lib.model as models
from lib.environment import atari_env

DEFAULT_ENV_ID = "BreakoutNoFrameskip-v4"
DEFAULT_MODEL_NAME = "ActorCriticLSTM"
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_GAME_VIDEOS_FOLDER="game_videos"
DEFAULT_INPUT_SPACE = (1, 80, 80)
DEFAULT_ACTION_SPACE = 4

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", default=DEFAULT_MODEL_NAME, help="Model file to load")
    parser.add_argument("-s", "--state_dict", required=True, help="Model file to load")
    parser.add_argument("-e", "--env", default=DEFAULT_ENV_ID, help="Environment name to use, default=" + DEFAULT_ENV_ID)
    parser.add_argument("-d", "--deterministic", default=True, action="store_true", help="enable deterministic actions")
    parser.add_argument("-r", "--record", default=DEFAULT_GAME_VIDEOS_FOLDER, help="If specified, sets the recording dir, default=game_videos")
    args = parser.parse_args()

    # Autodetect CUDA
    use_cuda = torch.cuda.is_available()
    device   = torch.device("cuda" if use_cuda else "cpu")

    #env = gym.make(args.env)
    env = atari_env(args.env)
    
    if args.record:
        env = gym.wrappers.Monitor(env, args.record, force=True)

    #num_inputs  = env.observation_space.shape[0]
    num_inputs = DEFAULT_INPUT_SPACE

    #num_outputs = env.action_space.shape[0]
    num_outputs = DEFAULT_ACTION_SPACE

    MODEL_CLASS = getattr(models, args.model_name)
    model = MODEL_CLASS(num_inputs, num_outputs, hidden_size=DEFAULT_HIDDEN_SIZE).to(device)
    model.load_state_dict(torch.load(args.state_dict))

    state = env.reset()
    done = False
    total_steps = 0
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        action = np.argmax(dist.probs.detach().cpu().numpy()) if args.deterministic \
                else int(dist.sample().cpu().numpy())
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        total_steps += 1
    env.env.close()
    print("In %d steps we got %.3f reward" % (total_steps, total_reward))
