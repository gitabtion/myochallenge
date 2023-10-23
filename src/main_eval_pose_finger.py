import os

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from definitions import ROOT_DIR
from envs.environment_factory import EnvironmentFactory
from train.trainer import MyoTrainer

# evaluation parameters:
render = True
num_episodes = 2_000

env_name = "CustomMyoFingerPoseRandom"

# Path to normalized Vectorized environment and best model (if not first task)
PATH_TO_NORMALIZED_ENV = os.path.join(ROOT_DIR, "output/training/2022-12-23/12-17-41_finger_pose_random_sds_0.5/rl_model_vecnormalize_800000_steps.pkl")
PATH_TO_PRETRAINED_NET = os.path.join(ROOT_DIR, "output/training/2022-12-23/12-17-41_finger_pose_random_sds_0.5/rl_model_800000_steps.zip")

# Reward structure and task parameters:
config = {
    "weighted_reward_keys": {
        "solved": 1,
        "done": 0,
        "sparse": 0,
    },
    "reset_type": "init",
    "sds_distance": None
}


# Function that creates and monitors vectorized environments:
def make_parallel_envs(
    env_name, env_config, num_env, start_index=0
):  # pylint: disable=redefined-outer-name
    def make_env(_):
        def _thunk():
            env = EnvironmentFactory.create(env_name, **env_config)
            return env

        return _thunk

    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


if __name__ == "__main__":
    # Create vectorized environments:
    envs = make_parallel_envs(env_name, config, num_env=1)

    # Normalize environment:
    if PATH_TO_NORMALIZED_ENV is not None:
        envs = VecNormalize.load(PATH_TO_NORMALIZED_ENV, envs)
    else:
        envs = VecNormalize(envs)
    envs.training = False
    envs.norm_reward = False

    # Create model
    custom_objects = {
        "learning_rate": lambda _: 0,
        "lr_schedule": lambda _: 0,
        "clip_range": lambda _: 0,
    }
    if PATH_TO_PRETRAINED_NET is not None:
        model = RecurrentPPO.load(
            PATH_TO_PRETRAINED_NET,
            env=envs,
            device="cpu",
            custom_objects=custom_objects,
        )
    else:
        model = MyoTrainer(
            envs=envs,
            env_config={},
            load_model_path=PATH_TO_PRETRAINED_NET,
            log_dir=os.path.join(ROOT_DIR, "output", "testing"),
        ).agent

    # EVALUATE
    eval_model = model
    eval_env = EnvironmentFactory.create(env_name, **config)

    # Enjoy trained agent
    perfs = []
    lens = []
    for i in range(num_episodes):
        lstm_states = None
        cum_rew = 0
        step = 0
        # eval_env.reset()
        # eval_env.step(np.zeros(39))
        obs = eval_env.reset()
        episode_starts = np.ones((1,), dtype=bool)
        done = False
        if render:
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
            eval_env.sim.render(mode="window")
        eval_env.sim.render(mode="window")
        while not done:
            if render:
                eval_env.sim.render(mode="window")
                eval_env.sim.render(mode="window")
                eval_env.sim.render(mode="window")
            action, lstm_states = eval_model.predict(
                envs.normalize_obs(obs),
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, rewards, done, info = eval_env.step(action)
            episode_starts = done
            cum_rew += rewards
            step += 1
        lens.append(step)
        perfs.append(cum_rew)
        print("Episode", i, ", len:", step, ", cum rew: ", cum_rew)

        if (i + 1) % 10 == 0:
            len_error = np.std(lens) / np.sqrt(i + 1)
            perf_error = np.std(perfs) / np.sqrt(i + 1)

            print(f"\nEpisode {i+1}/{num_episodes}")
            print(f"Average len: {np.mean(lens):.2f} +/- {len_error:.2f}")
            print(f"Average rew: {np.mean(perfs):.2f} +/- {perf_error:.2f}\n")

    print(f"\nFinished evaluating {PATH_TO_PRETRAINED_NET}!")
