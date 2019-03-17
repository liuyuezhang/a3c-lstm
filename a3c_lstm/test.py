from envs.atari.env import make_env
from a3c_lstm.model import A3C_LSTM
from common.logger import Logger

import time

import torch
import torch.nn.functional as F


def test(idx, args, T, shared_net, path):
    device = torch.device("cuda")

    torch.manual_seed(args.seed + idx)
    if args.cuda:
        torch.cuda.manual_seed(args.seed + idx)

    env = make_env(args.env, stack_frames=args.stacked_frames,
                   max_episode_steps=args.max_episode_steps,
                   episodic_life=True, reward_clipping=False)
    env.seed(args.seed + idx)

    state = env.reset()
    state_v = torch.from_numpy(state).float().to(device)
    hx = torch.zeros(1, 512).to(device)
    cx = torch.zeros(1, 512).to(device)

    info = True  # game is real done, not end of a life (EpisodicLife)

    net = A3C_LSTM(env.observation_space.shape[0], env.action_space.n).to(device)
    net.eval()

    logger = Logger(name="test", path=path, model=shared_net, start_time=time.time(),
                    print_log=True, save_model=True)

    while T.value < args.num_timesteps:
        # Synchronize thread-specific parameters
        if info:
            net.load_state_dict(shared_net.state_dict())

        # Perform action according to policy
        with torch.no_grad():
            value_v, logit_v, (hx, cx) = net(state_v.unsqueeze(0), (hx, cx))
        prob_v = F.softmax(logit_v, dim=1)
        action_v = torch.multinomial(prob_v, num_samples=1)
        action = int(action_v.item())

        # Receive reward and new state
        state, reward, done, info = env.step(action)
        state_v = torch.from_numpy(state).float().to(device)

        logger.log(T.value, reward, info)

        if done:
            state = env.reset()
            state_v = torch.from_numpy(state).float().to(device)
            hx = torch.zeros(1, 512).to(device)
            cx = torch.zeros(1, 512).to(device)
