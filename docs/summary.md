# Summary of our approach

## Key components of the final model

1. On-policy learning with PPO
2. Curriculum learning to guide and stabilize training throughout

### 1. On-policy learning with PPO

First, We conducted highly parallel on-policy training using the Stable Baselines3 Proximal Policy Optimization (PPO) algorithm. Through reward adjustment, we successfully achieved effective manipulation for grasping tasks.

### 2. Curriculum learning

Second, we used a curriculum of training that gradually introduced the tougher aspects of the task.

Based on the progress made in the P1 checkpoint, we have designed a six-step plan for the second phase of curriculum learning:

- Step 1: We think that the first phase is a good initial task, for step 1, we using the checkpoint obtained from P1 training directly.
- Step 2: In this step, we will ensure that the target_xyz_range, obj_geom_range, and obj_friction_range are consistent with the P2 environment.
- Step 3: Similarly, we will align the qpos_noise_range, target_rxryrz_range, and obj_mass_range with the P2 environment.
- Step 4: Based on our experimental observations, changes in obj_xyz_range will disrupt the pre-grasp state. We have modified the initialization range of obj_xyz_range to position it in mid-air and present it in a pre-grasp state, in order to reduce the difficulty of grasping.
- Step 5: Based on the above steps, we believe that the model is already capable of preliminarily grasping objects in the P2 simulation environment. In this step, we proceed with further training directly in the P2 simulation environment.
- Step 6: To further enhance the model's performance, we increased the values of obj_xyz_range, obj_geom_range, and obj_mass_range, aiming to encourage the model to possess stronger generalization capabilities.

- For further details, including hyperparameters, rewards for each step, and checkpoints, please refer to the appendix.

## Appendix

### Curriculum 

#### Phase 1

To ensure sufficient proximity between the hand and the object, we incorporated an additional reward term called "reach." This adjustment was implemented to incentivize the model to position the palm adequately close to the object.
```python
reach = min(0, 0.05 - np.linalg.norm(obs_dict['reach_err']))
```

The following are the weight parameters for each reward term:
```python
"weighted_reward_keys": {
        "act_reg": 1,
        "solved": 100.,
        "sparse": 100.0,
        "reach": 10000,
}
```

#### Phase 2

All the trained models, environment configurations, main files, and tensorboard logs are all present in the [trained_models/](../trained_models) folder. 

The following are environment config:

- step1:
```python
"weighted_reward_keys": {
        "act_reg": 1,
        "solved": 100.,
        "sparse": 100.0,
        "reach": 10000,
}
```

- step2
```python
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 15.0,
        "rot_dist": 0,
        "act_reg": 0.01,
        "solved": 10.,
        "drop": -1.,
        # "sparse": 10.0,
        "keep_time": -200.,
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    'target_xyz_range': {'high': [0.3, -.1, 0.9], 'low': [0.0, -.45, 0.9]},
    'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.15, 0.15]},
    'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}
```
- step3
```python
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 15.0,
        "rot_dist": 0,
        "act_reg": 0.00,
        "solved": 10.,
        "drop": -1.,
        # "sparse": 10.0,
        "keep_time": -200.,
        # "reach_dist": 4,
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    'qpos_noise_range': 0.01,  # jnt initialization range
    'target_xyz_range': {'high': [0.3, -.1, 0.9], 'low': [0.0, -.45, 0.9]},
    'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
    'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
    'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}
```
- step4
```python
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
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
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    'qpos_noise_range': 0.01,  # jnt initialization range
    'target_xyz_range': {'high': [0.3, -.1, 0.9], 'low': [0.0, -.45, 0.9]},
    'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    'obj_xyz_range': {'high': [-0.02, -.20, 1.05], 'low': [-0.03, -.23, 1.05]},
    'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
    'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
    'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}
```
- step5
```python
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 10.0,
        "rot_dist": 0,
        "act_reg": 0.00,
        # "solved": 10.,
        "drop": -1.,
        # "sparse": 10.0,
        "keep_time": -200.,
        # "reach_dist": 4,
        "norm_solved": 10.,
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    'qpos_noise_range': 0.01,  # jnt initialization range
    'target_xyz_range': {'high': [0.3, -.1, 1.05], 'low': [0.0, -.45, 0.9]},
    'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    'obj_xyz_range': {'high': [0.1, -.15, 1.0], 'low': [-0.1, -.35, 1.0]},
    'obj_geom_range': {'high': [.025, .025, .025], 'low': [.015, 0.015, 0.015]},
    'obj_mass_range': {'high': 0.200, 'low': 0.050},  # 50gms to 200 gms
    'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}
```
- step6
```python
config = {
    "obs_keys": ['hand_qpos', 'hand_qvel', 'obj_pos', 'goal_pos', 'pos_err', 'obj_rot', 'goal_rot',
                 'rot_err'],
    "weighted_reward_keys": {
        "pos_dist": 1.0,
        "rot_dist": 0,
        "act_reg": 0.01,
        # "solved": 10.,
        # "drop": -1.,
        # "sparse": 10.0,
        # "keep_time": -200.,
        # "reach_dist": 4,
        "norm_solved": 10.,
    },
    'normalize_act': True,
    'frame_skip': 5,
    'pos_th': 0.1,  # cover entire base of the receptacle
    'rot_th': np.inf,  # ignore rotation errors
    'qpos_noise_range': 0.01,  # jnt initialization range
    'target_xyz_range': {'high': [0.3, -.1, 1.05], 'low': [0.0, -.45, 0.9]},
    'target_rxryrz_range': {'high': [0.2, 0.2, 0.2], 'low': [-.2, -.2, -.2]},
    'obj_xyz_range': {'high': [0.15, -.1, 1.03], 'low': [-0.15, -.4, 0.95]},
    'obj_geom_range': {'high': [.03, .03, .03], 'low': [.01, 0.01, 0.01]},
    'obj_mass_range': {'high': 0.250, 'low': 0.050},  # 50gms to 200 gms
    'obj_friction_range': {'high': [1.2, 0.006, 0.00012], 'low': [0.8, 0.004, 0.00008]}
}
```


### Architecture, algorithm, and hyperparameters

#### Architecture and algorithm

We use [PPO from Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3/blob/v1.6.2/stable_baselines3/ppo/ppo.py) as our base algorithm with the following architecture for both the actor and the critic with nothing shared between the two:

obs --> 128 Linear --> 256 Linear --> 128 Linear --> output

#### Hyperparameters

| Hyperparameter                             | Value |
|--------------------------------------------|-------|
| Discount factor $\gamma$                   | 0.999 |
| Generalized Advantage Estimation $\lambda$ | 0.95  |
| Entropy regularization coefficient         | 0.001 |
| vf coefficient                             | 0.2   |
| clip range                                 | 0.2   |
| Optimizer                                  | Adam  |
| learning rate                              | 5e-5  |
| Batch size                                 | 4096  |
| max grad norm                              | 0.5   |