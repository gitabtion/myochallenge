import time
from typing import Optional, Type, Union

import gymnasium as gym
import imageio
import numpy as np
import myosuite
import tianshou as ts
import torch
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv, VectorEnvNormObs
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import RecurrentCritic, RecurrentActorProb, ActorProb, Critic
from torch import nn
from torch.distributions import Normal, Independent

from envs.environment_factory import EnvironmentFactory

config = {
    "obs_keys": ['hand_qpos_corrected', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 10.0,
        "rot_dist": 0,
        "act_reg": 0.00,
        "solved": 10.,
        "drop": -1.,
        # "sparse": 10.0,
        "keep_time": -200.,
        # "reach_dist": 4,
        # "norm_solved": 10.,
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    # 'qpos_noise_range': 0.01,  # jnt initialization range
    # 'target_xyz_range': {'high': [0.3, -.1, 0.9], 'low': [0.0, -.45, 0.9]},
    # 'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    # 'obj_xyz_range': {'high': [0.1, -.15, 0.95], 'low': [-0.1, -.35, 0.95]},
    # 'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
    # 'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
    # 'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}

def make_parallel_envs(env_config, num_env, start_index=0,
                       vec_env_cls: Optional[Type[Union[DummyVectorEnv, SubprocVectorEnv, ShmemVectorEnv]]] = None, seed=42):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.create("CustomMyoRelocateP1", **env_config)
            env = gym.make("GymV21Environment-v0", env_id="CustomMyoChallengeRelocateP1-v0",
                           env=env)
            env.gym_env.seed(seed + rank)
            return env

        return _thunk

    return vec_env_cls([make_env(i + start_index) for i in range(num_env)])


def dist(*logits):
    return Independent(Normal(*logits), 1)

def inference():
    static_dict = torch.load('/home/abtion/workspace/myochallenge/output/training/2023-11-08/10-30-17/policy.pth',
                             map_location='cpu')

    env = make_parallel_envs(config, 1, vec_env_cls=DummyVectorEnv)
    _env = env.workers[0].env
    _env.mj_render()
    envs = VectorEnvNormObs(env, update_obs_rms=False)

    state_shape = _env.observation_space.shape or _env.observation_space.n
    action_shape = _env.action_space.shape or _env.action_space.n
    net_a = Net(
        state_shape,
        hidden_sizes=[128, 256, 128],
        activation=nn.GELU,
        device='cuda',
    )
    actor = ActorProb(
        net_a,
        action_shape,
        unbounded=True,
        device='cuda',
    ).to('cuda')
    net_c = Net(
        state_shape,
        hidden_sizes=[128, 256, 128],
        activation=nn.GELU,
        device='cuda',
    )
    critic = Critic(net_c, device='cuda')
    actor_critic = ActorCritic(actor, critic)
    actor_critic.load_state_dict(static_dict['model'], strict=False)
    actor_critic.to('cuda')
    obs_rms = static_dict['obs_rms']

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        dist_fn=dist,
        optim=None,
        discount_factor=0.99,
        action_space=_env.action_space,
        reward_normalization=True,
    )
    policy.load_state_dict(static_dict['model'])
    policy = policy.to('cuda')

    envs.set_obs_rms(obs_rms)

    test_collector = ts.data.Collector(policy, envs, exploration_noise=True)
    result = test_collector.collect(n_episode=10)
    print(result)

    # time.sleep(5)


if __name__ == '__main__':
    inference()
