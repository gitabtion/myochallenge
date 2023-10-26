import os
import shutil
from datetime import datetime
from typing import Union, Type, Optional

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
import wandb
from wandb.integration.sb3 import WandbCallback
from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from metrics.custom_callbacks import EnvDumpCallback, EvalCallback
from train.trainer import MyoTrainer

# define constants
ENV_NAME = "CustomMyoRelocateP1"

now = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
TENSORBOARD_LOG = os.path.join(ROOT_DIR, "output", "training", now)

load_folder = "trained_models/"
PATH_TO_NORMALIZED_ENV = load_folder + "env_320000000_steps"
PATH_TO_PRETRAINED_NET = load_folder + "rl_models_320000000_steps.zip"

# Reward structure and task parameters:
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
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
    'target_xyz_range': {'high': [0.3, -.1, 0.9], 'low': [0.0, -.45, 0.9]},
    # 'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    # 'obj_xyz_range': {'high': [0.1, -.15, 0.95], 'low': [-0.1, -.35, 0.95]},
    # 'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
    # 'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
    # 'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(env_config, num_env, start_index=0,
                       vec_env_cls: Optional[Type[Union[DummyVecEnv, SubprocVecEnv]]] = None, seed=42):
    def make_env(rank):
        def _thunk():
            env = EnvironmentFactory.create(ENV_NAME, **env_config)
            env = Monitor(env, TENSORBOARD_LOG)
            env.seed(seed + rank)
            return env

        return _thunk

    return vec_env_cls([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # ensure tensorboard log directory exists and copy this file to track
    os.makedirs(TENSORBOARD_LOG, exist_ok=True)
    shutil.copy(os.path.abspath(__file__), TENSORBOARD_LOG)

    run = wandb.init(
        project="sb3_arm",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    # Create and wrap the training and evaluations environments
    envs = make_parallel_envs(config, 2, vec_env_cls=SubprocVecEnv)
    envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    envs.ret_rms = RunningMeanStd(shape=())
    envs.old_reward = np.array([])
    # envs = VecNormalize(envs, norm_obs=True, norm_reward=True, clip_obs=10.)

    eval_env = make_parallel_envs(config, num_env=10, vec_env_cls=DummyVecEnv)
    # eval_env = VecNormalize.load(PATH_TO_NORMALIZED_ENV, eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10, training=False)
    eval_env.obs_rms = envs.obs_rms

    # Define callbacks for evaluation and saving the agent
    eval_callback = EvalCallback(
        eval_env=eval_env,
        n_eval_episodes=10,
        eval_freq=50_000,
        verbose=1,
        save_path=TENSORBOARD_LOG,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=TENSORBOARD_LOG,
        save_vecnormalize=True,
        name_prefix='rl_models',
        verbose=1,
    )

    wandb_callback = WandbCallback(model_save_path="models/", verbose=2)

    # Define trainer
    trainer = MyoTrainer(
        envs=envs,
        env_config=config,
        load_model_path=None,
        log_dir=TENSORBOARD_LOG,
        model_config={
            # "lr_schedule": lambda _: 5e-05,
            "learning_rate": lambda _: 1e-05,
            "clip_range": lambda _: 0.2,
            'gamma': 0.999,
            'batch_size': 2048,
            "n_steps": 128,
            # "policy_kwargs": {"net_arch": [dict(pi=[128, 128], vf=[128, 128])]}
        },
        callbacks=[eval_callback, checkpoint_callback],
        timesteps=600_000_000,
    )

    # Train agent
    trainer.train(total_timesteps=trainer.timesteps)
    trainer.save()
