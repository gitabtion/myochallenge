"""
@Time: 24/10/2023 下午8:20
@Author: Heng Cai
@FileName: ts_relocate.py
@Copyright: 2020-2023 CarbonSilicon.ai
@Description:
"""
import os
import shutil
from datetime import datetime
from typing import Union, Type, Optional
import torch, numpy as np, torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import myosuite
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from torch.distributions import Independent, Normal
import tianshou as ts
from tianshou.env import DummyVectorEnv, SubprocVectorEnv, VectorEnvNormObs, ShmemVectorEnv
from tianshou.utils.net.continuous import ActorProb, Critic
from shimmy import GymV21CompatibilityV0
from tianshou.utils.net.common import Net, ActorCritic
import gymnasium as gym

ENV_NAME = "CustomMyoChallengeRelocateP2-v0"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", now)

config = {
    "obs_keys": ['hand_qpos_corrected', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 15.0,
        "rot_dist": 0,
        "act_reg": 0.01,
        "solved": 100.,
        "drop": -1.,
        # "sparse": 10.0,
        "keep_time": -200.,
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
            env = gym.make("GymV21Environment-v0", env_id="CustomMyoChallengeRelocateP1-v0", env=env)
            env.gym_env.seed(seed + rank)
            return env


        return _thunk

    return vec_env_cls([make_env(i + start_index) for i in range(num_env)])


def dist(*logits):
    return Independent(Normal(*logits), 1)



if __name__ == '__main__':
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    lr, epoch, batch_size = 5e-5, 400, 2048
    train_num, test_num = 128, 10
    gamma, n_step, target_freq = 0.999, 3, 320
    buffer_size = 624000
    eps_train, eps_test = 0.1, 0.05
    step_per_epoch, step_per_collect = 1024000, 2048
    logger = ts.utils.WandbLogger(
        project='ts_arm',
        run_id=now.replace('/', '_'),
        config=config)
    logger.load(SummaryWriter(TENSORBOARD_LOG))

    envs = make_parallel_envs(config, train_num, vec_env_cls=ShmemVectorEnv)
    envs = VectorEnvNormObs(envs)

    eval_envs = make_parallel_envs(config, test_num, vec_env_cls=DummyVectorEnv)
    eval_envs = VectorEnvNormObs(eval_envs, update_obs_rms=False)
    eval_envs.set_obs_rms(envs.get_obs_rms())

    env = eval_envs.venv.workers[0].env

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
    net_a = Net(
        state_shape,
        hidden_sizes=[128, 128, 128],
        activation=nn.ReLU,
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
        hidden_sizes=[128, 128, 128],
        activation=nn.ReLU,
        device='cuda',
    )
    critic = Critic(net_c, device='cuda').to('cuda')
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

    policy = ts.policy.PPOPolicy(
        actor=actor,
        critic=critic,
        dist_fn=dist,
        optim=optim,
        discount_factor=gamma,
        action_space=env.action_space,
        reward_normalization=True,
    )
    train_collector = ts.data.Collector(policy, envs, ts.data.VectorReplayBuffer(buffer_size, train_num),
                                        exploration_noise=True)
    test_collector = ts.data.Collector(policy, eval_envs, exploration_noise=True)


    def save_best_fn(policy):
        state = {"model": policy.state_dict(), "obs_rms": envs.get_obs_rms()}
        torch.save(state, os.path.join(TENSORBOARD_LOG, "policy.pth"))

    result = ts.trainer.OnpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=test_collector,
        max_epoch=epoch,
        step_per_epoch=step_per_epoch,
        repeat_per_collect=4,
        episode_per_test=test_num,
        batch_size=batch_size,
        step_per_collect=int(batch_size*128/train_num),
        save_best_fn=save_best_fn,
        logger=logger,
        test_in_train=False, ).run()
    print(f'Finished training! Use {result["duration"]}')

    policy.eval()
    eval_envs.seed(42)
    test_collector.reset()
    result = test_collector.collect(n_episode=test_num, render=0.0)
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
