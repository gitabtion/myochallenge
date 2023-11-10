"""
@Time: 20/10/2023 下午4:28
@Author: Heng Cai
@FileName: relocate.py
@Copyright: 2020-2023 CarbonSilicon.ai
@Description:
"""
# pylint: disable=attribute-defined-outside-init, dangerous-default-value, protected-access, abstract-method, arguments-renamed, import-error
import collections
import random

import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myochallenge.relocate_v0 import RelocateEnvV0
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from .environment_factory import EnvironmentFactory


class CustomRelocateEnv(RelocateEnvV0):
    DEFAULT_OBS_KEYS = ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                        'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 10.0,
        "rot_dist": 0.0,
        "act_reg": 0.01,
        "solved": 10000.,
        "drop": -10.,
    }

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)
        self.sim.model.opt.gravity[2] = -9.8

    def _setup(self,
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               **kwargs,
               ):
        super()._setup(obs_keys=obs_keys, weighted_reward_keys=weighted_reward_keys, **kwargs)

    def get_reward_dict(self, obs_dict):
        reach_dist = np.abs(np.linalg.norm(self.obs_dict['reach_err'], axis=-1))
        pos_dist = np.abs(np.linalg.norm(self.obs_dict['pos_err'], axis=-1))
        rot_dist = np.abs(np.linalg.norm(self.obs_dict['rot_err'], axis=-1))
        a_rot = 0.2
        act_mag = np.linalg.norm(self.obs_dict['act'], axis=-1) / self.sim.model.na if self.sim.model.na != 0 else 0
        drop = reach_dist > self.drop_th
        solved = (pos_dist < self.pos_th) and (rot_dist < self.rot_th) and (~drop)
        use_time = self.sim.data.time
        rwd_dict = collections.OrderedDict((
            # Perform reward tuning here --
            # Update Optional Keys section below
            # Update reward keys (DEFAULT_RWD_KEYS_AND_WEIGHTS) accordingly to update final rewards
            # Examples: Env comes pre-packaged with two keys pos_dist and rot_dist
            # Optional Keys
            ('pos_dist', -1 * (2 * pos_dist) + 1. / ((2 * pos_dist) ** 2 + 1)),
            ('rot_dist', a_rot / (rot_dist ** 2 + a_rot)),
            # Must keys
            ('act_reg', -1. * act_mag),
            ('sparse', -rot_dist - 10.0 * pos_dist),
            ('solved', solved),
            ('drop', (reach_dist > 0.05) and ~solved),
            ('done', drop),
            ('keep_time', drop * (1.5 - use_time)),
            ('norm_solved', solved * (use_time > 0.8)),
            ('reach_dist', -1 * (2 * reach_dist) + 1. / ((2 * reach_dist) ** 2 + 1))
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array(
            [2, 0])
        self.sim.model.site_size[self.success_indicator_sid, :] = np.array([.25, ]) if rwd_dict['solved'] else np.array(
            [0.1, ])
        return rwd_dict

    def render(self, mode='rgb_array', height=240, width=320, camera_id=1):
        # assert mode == 'rgb_array', f"env only supports rgb_array rendering, but get {mode}"
        frame = self.sim.renderer.render_offscreen(width=640, height=480, camera_id=camera_id)
        return frame
