"""
@Time: 23/10/2023 下午2:50
@Author: Heng Cai
@FileName: relocate_dm.py
@Copyright: 2020-2023 CarbonSilicon.ai
@Description:
"""
import collections
import random
import gym
import numpy as np
from myosuite.envs.myo.base_v0 import BaseV0
from myosuite.envs.myo.myochallenge.relocate_v0 import RelocateEnvV0
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from .environment_factory import EnvironmentFactory

class CustomRelocateEnv(RelocateEnvV0):
    DEFAULT_OBS_KEYS = ['hand_qpos_corrected', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                        'rot_err']
    DEFAULT_RWD_KEYS_AND_WEIGHTS = {
        "pos_dist": 10.0,
        "rot_dist": 0.0,
        "act_reg": 0.01,
        "solved": 10000.,
        "drop": -10.,
    }

    def __init__(self, model_path, obsd_model_path=None, seed=None, **kwargs):
        # Two step construction (init+setup) is required for pickling to work correctly.
        gym.utils.EzPickle.__init__(self, model_path, obsd_model_path, seed, **kwargs)
        super().__init__(model_path=model_path, obsd_model_path=obsd_model_path, seed=seed, env_credits=self.MYO_CREDIT)
        self._setup(**kwargs)

    def _setup(self,
               target_xyz_range,  # target position range (relative to initial pos)
               target_rxryrz_range,  # target rotation range (relative to initial rot)
               obj_xyz_range=None,  # object position range (relative to initial pos)
               obj_geom_range=None,  # randomization sizes for object geoms
               obj_mass_range=None,  # object size range
               obj_friction_range=None,  # object friction range
               qpos_noise_range=None,  # Noise in joint space for initialization
               obs_keys: list = DEFAULT_OBS_KEYS,
               weighted_reward_keys: list = DEFAULT_RWD_KEYS_AND_WEIGHTS,
               pos_th=.025,  # position error threshold
               rot_th=0.262,  # rotation error threshold
               drop_th=0.50,  # drop height threshold
               **kwargs,
               ):
        self.palm_sid = self.sim.model.site_name2id("S_grasp")
        self.object_sid = self.sim.model.site_name2id("object_o")
        self.object_bid = self.sim.model.body_name2id("Object")
        self.goal_sid = self.sim.model.site_name2id("target_o")
        self.success_indicator_sid = self.sim.model.site_name2id("target_ball")
        self.goal_bid = self.sim.model.body_name2id("target")
        self.target_xyz_range = target_xyz_range
        self.target_rxryrz_range = target_rxryrz_range
        self.obj_geom_range = obj_geom_range
        self.obj_mass_range = obj_mass_range
        self.obj_friction_range = obj_friction_range
        self.obj_xyz_range = obj_xyz_range
        self.qpos_noise_range = qpos_noise_range
        self.pos_th = pos_th
        self.rot_th = rot_th
        self.drop_th = drop_th

        super()._setup(obs_keys=obs_keys,
                       weighted_reward_keys=weighted_reward_keys,
                       **kwargs,
                       )
        keyFrame_id = 0 if self.obj_xyz_range is None else 1
        self.init_qpos[:] = self.sim.model.key_qpos[keyFrame_id].copy()

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
            ('drop', reach_dist > 0.5),
            ('done', drop),
            ('keep_time', drop * (1.5-use_time))
        ))
        rwd_dict['dense'] = np.sum([wt * rwd_dict[key] for key, wt in self.rwd_keys_wt.items()], axis=0)

        # Success Indicator
        self.sim.model.site_rgba[self.success_indicator_sid, :2] = np.array([0, 2]) if rwd_dict['solved'] else np.array(
            [2, 0])
        self.sim.model.site_size[self.success_indicator_sid, :] = np.array([.25, ]) if rwd_dict['solved'] else np.array(
            [0.1, ])
        return rwd_dict

    def render(self, mode='rgb_array', height=240, width=320, camera_id=1):
        assert mode == 'rgb_array', f"env only supports rgb_array rendering, but get {mode}"
        frame = self.sim.renderer.render_offscreen(width=640, height=480, camera_id=camera_id)
        return frame