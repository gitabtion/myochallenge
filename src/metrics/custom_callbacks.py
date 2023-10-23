import os
import time
import numpy as np
import wandb
import imageio
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


class EvaluateLSTM(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, eval_freq, eval_env, name, num_episodes=20, verbose=0):
        super(EvaluateLSTM, self).__init__()
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.name = name
        self.num_episodes = num_episodes

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        if self.num_timesteps % self.eval_freq == 0:

            perfs = []
            for _ in range(self.num_episodes):
                lstm_states, cum_rew, step = None, 0, 0
                obs = self.eval_env.reset()
                episode_starts = np.ones((1,), dtype=bool)
                done = False
                while not done:
                    (
                        action,
                        lstm_states,
                    ) = self.model.predict(  # use the training model to predict (object from super class)
                        self.training_env.normalize_obs(
                            obs
                        ),  # use the training environment to normalize (object from super class)
                        state=lstm_states,
                        episode_start=episode_starts,
                        deterministic=True,
                    )
                    obs, rewards, done, _ = self.eval_env.step(action)
                    episode_starts = done
                    cum_rew += rewards
                    step += 1
                perfs.append(cum_rew)

            self.logger.record(self.name, np.mean(perfs))
        return True


class EnvDumpCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super().__init__(verbose=verbose)
        self.save_path = save_path

    def _on_step(self):
        env_path = os.path.join(self.save_path, "training_env.pkl")
        if self.verbose > 0:
            print("Saving the training environment to path ", env_path)
        self.training_env.save(env_path)
        return True
    
    
class TensorboardCallback(BaseCallback):
    def __init__(self, info_keywords, verbose=0):
        super().__init__(verbose=verbose)
        self.info_keywords = info_keywords
        self.rollout_info = {}
        
    def _on_rollout_start(self):
        self.rollout_info = {key: [] for key in self.info_keywords}
        
    def _on_step(self):
        for key in self.info_keywords:
            vals = [info[key] for info in self.locals["infos"]]
            self.rollout_info[key].extend(vals)
        return True
    
    def _on_rollout_end(self):
        for key in self.info_keywords:
            self.logger.record("rollout/" + key, np.mean(self.rollout_info[key]))

class EvalCallback(BaseCallback):
    def __init__(self, eval_freq, eval_env, verbose=0, n_eval_episodes=10, save_path=os.getcwd()):
        super().__init__(verbose)
        self._vid_log_dir = os.path.join(save_path, 'eval_videos/')
        os.makedirs(self._vid_log_dir, exist_ok=True)
        self._eval_freq = eval_freq
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes

    def _info_callback(self, locals, _):
        if locals['i'] == 0:
            env = locals['env']
            render = env.envs[0].render('rgb_array')
            self._info_tracker['rollout_video'].append(render)

        if locals['done']:
            for k, v in locals['info'].items():
                if isinstance(v, (float, int)):
                    if k not in self._info_tracker:
                        self._info_tracker[k] = []
                    self._info_tracker[k].append(v)
            for k, v in locals['info']['rwd_dict'].items():
                _k = f'rwd_{k}'
                if _k not in self._info_tracker:
                    self._info_tracker[_k] = []
                self._info_tracker[_k].append(v)
            for k, v in locals['info']['obs_dict'].items():
                if ('dist' in k) or ('err' in k):
                    _k = f'rwd_{k}'
                    if _k not in self._info_tracker:
                        self._info_tracker[_k] = []
                    self._info_tracker[_k].append(v)

    def _on_step(self, fps=25) -> bool:
        if self.n_calls % self._eval_freq == 0 or self.n_calls <= 1:
            self._info_tracker = dict(rollout_video=[])
            start_time = time.time()
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self._eval_env,
                n_eval_episodes=self._n_eval_episodes,
                render=False,
                deterministic=False,
                return_episode_rewards=True,
                warn=True,
                callback=self._info_callback,
            )
            end_time = time.time()

            mean_reward, mean_length = np.mean(episode_rewards), np.mean(episode_lengths)
            self.logger.record('eval/time', end_time - start_time)
            self.logger.record('eval/mean_reward', mean_reward)
            self.logger.record('eval/mean_length', mean_length)
            print(self._info_tracker.keys())
            for k, v in self._info_tracker.items():
                if k == 'rollout_video':
                    # pass
                    path = 'eval-call-{}.mp4'.format(self.n_calls)
                    path = os.path.join(self._vid_log_dir, path)
                    writer = imageio.get_writer(path, fps=fps)
                    for i in v:
                        writer.append_data(i)
                    writer.close()
                    print(len(v))
                    if os.path.exists(path):
                        wandb.log({'eval/rollout_video': wandb.Video(path)})
                else:
                    self.logger.record('eval/mean_{}'.format(k), np.mean(v))
            self.logger.dump(self.num_timesteps)
        return True
