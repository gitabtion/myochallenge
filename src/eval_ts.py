import time
import gymnasium as gym
import imageio
import numpy as np
import myosuite
import torch
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import RecurrentCritic, RecurrentActorProb
import envs


def inference():
    static_dict = torch.load('output/training/2023-10-25/20-55-33/policy.pth', map_location='cpu')

    env = gym.make("GymV21Environment-v0", env_id="CustomMyoChallengeRelocateP1-v0")
    env = DummyVectorEnv([lambda: gym.make("GymV21Environment-v0", env_id="CustomMyoChallengeRelocateP1-v0")])

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    actor = RecurrentActorProb(
        layer_num=2,
        state_shape=state_shape,
        action_shape=action_shape,
        device='cuda',
    ).to('cuda')
    critic = RecurrentCritic(
        layer_num=2,
        state_shape=state_shape,
        device='cuda',
    ).to('cuda')
    actor_critic = ActorCritic(actor, critic)
    actor_critic.load_state_dict(static_dict['model'])
    obs_rms = static_dict['obs_rms']

    gym_env = env.venv.workers[0].env
    # time.sleep(5)
    actor_critic.eval()

    frames = []
    with torch.no_grad():
        for e in range(1):
            print(e)
            state = env.reset()
            done = False
            score = 0
            while not done:
                frame = gym_env.render('rgb_array')
                frames.append(frame)
                obs = obs_rms.norm(state)
                act = actor_critic(obs)
                gym_env.mj_render()
                time.sleep(0.1)
            print(score)
        env.close()

    path = 'eval.mp4'
    writer = imageio.get_writer(path, fps=25)
    for i in frames:
        writer.append_data(i)
    writer.close()


if __name__ == '__main__':
    inference()
