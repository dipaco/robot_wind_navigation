#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
import os
import sys
import time
import yaml
import shutil
import signal
import pickle as pkl

__BASE_FOLDER__ = os.path.dirname(os.path.abspath(__file__))
dist_package_folder = os.path.join(__BASE_FOLDER__, 'submodules/pytorch_sac/')
sys.path.append(dist_package_folder)

from video import VideoRecorder
from logger import Logger
#from replay_buffer import ReplayBuffer
from multi_robot_replay_buffer import MultiRobotReplayBuffer
from environments import TurbulentWindEnv, RandomWindEnv, TurbulentFormationEnv
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import utils

import gym
import hydra


def make_env(cfg):
    """Helper function to create dm_control environment"""
    if cfg.env == 'ball_in_cup_catch':
        domain_name = 'ball_in_cup'
        task_name = 'catch'
    else:
        domain_name = cfg.env.split('_')[0]
        task_name = '_'.join(cfg.env.split('_')[1:])

    env = dmc2gym.make(domain_name=domain_name,
                       task_name=task_name,
                       seed=cfg.seed,
                       visualize_reward=True)
    env.seed(cfg.seed)
    assert env.action_space.low.min() >= -1
    assert env.action_space.high.max() <= 1

    return env


def make_turbulence_env(cfg):

    #env = TurbulentWindEnv(cfg)
    #env = gym.make('MountainCarContinuous-v0')
    #env = RandomWindEnv(cfg)
    env = TurbulentFormationEnv(cfg)
    env.seed(cfg.seed)

    return env


class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        '''self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)'''

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.env = make_turbulence_env(cfg)

        cfg.agent.params.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.params.action_dim = self.env.action_space.shape[0]
        cfg.agent.params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]

        self.agent = hydra.utils.instantiate(cfg.agent)

        self.logger = Logger(self.work_dir,
                             save_tb=cfg.log_save_tb,
                             log_frequency=cfg.log_frequency,
                             agent=cfg.agent.name)

        self.replay_buffer = MultiRobotReplayBuffer(self.env.observation_space.shape,
                                          self.env.action_space.shape,
                                          int(cfg.replay_buffer_capacity),
                                          self.cfg.formation_params.num_nodes,
                                          self.device)

        self.video_recorder = VideoRecorder(self.work_dir if cfg.save_video else None, fps=15)

        self.train_episode = 0
        self.step = 0

        #FIXME: Create a class to do this
        # Checkpoints
        self.checkpoint_folder = os.path.join(self.work_dir, 'checkpoints')
        self.latest_checkpoint = os.path.join(self.checkpoint_folder, 'latest.pt')
        self.previous_checkpoint = os.path.join(self.checkpoint_folder, 'previous.pt')
        os.makedirs(self.checkpoint_folder, exist_ok=True)

        if self.cfg.resume and os.path.exists(self.latest_checkpoint):
            self.load_progress()

        # Register all the signal callbacks to control cluster training
        self.exit_code = None
        signal.signal(signal.SIGTERM, self.capture_signal)
        signal.signal(signal.SIGALRM, self.capture_signal)
        signal.alarm(self.cfg.time_to_run)

    def evaluate(self):
        average_episode_reward = 0
        results_folder = os.path.join(self.work_dir, 'results')
        os.makedirs(results_folder, exist_ok=True)
        eval_errors = {}
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            self.agent.reset()
            self.video_recorder.init(enabled=(episode == 0))
            done = False
            episode_reward = 0
            while not done:

                # Terminate with the appropriate exit code
                if self.exit_code == 3:
                    self.save_progress()
                    return self.exit_code

                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, _ = self.env.step(action)
                # Only records the first eval episode
                if episode == 0:
                    self.video_recorder.record(self.env)
                episode_reward += reward.mean()

            average_episode_reward += episode_reward
            # Only records the first eval episode
            if episode == 0:
                self.video_recorder.save(f'{self.step}.mp4')

            # append the the episode errors to the error list
            for k, v in self.env.get_errors().items():
                if k in eval_errors:
                    eval_errors[k].append(v)
                else:
                    eval_errors[k] = [v]

        for k, v in eval_errors.items():
            eval_errors[k] = np.stack(eval_errors[k])

        # save formation error data and plots
        np.save(os.path.join(results_folder, f'step_{self.step}_eval_formation_error.npy'), eval_errors)
        self.env.plot_episode_evaluation(data_dict=eval_errors, results_folder=results_folder, step=self.step)

        average_episode_reward /= self.cfg.num_eval_episodes
        self.logger.log('eval/episode_reward', average_episode_reward, self.step)
        self.logger.dump(self.step)

    def run(self):
        episode_reward, done, ff = 0, True, False
        start_time = time.time()
        num_evaluations = 0
        init_step = self.step   # to handle some troubles when resuming from a checkpoint
        while self.step < self.cfg.num_train_steps:

            # Terminate with the appropriate exit code
            if self.exit_code == 3:
                self.save_progress()
                return self.exit_code

            # Saves a checkpoint after
            if (self.step + 1) % self.cfg.save_steps == 0:
                self.save_progress()

            if done:
                if self.step > init_step:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                #if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                if self.step > 0 and (self.step // self.cfg.eval_frequency) > num_evaluations:
                    num_evaluations = self.step // self.cfg.eval_frequency
                    self.logger.log('eval/episode', self.train_episode, self.step)
                    self.save_progress()

                    # if evaluation ended with exit code 3, we propagate the sys.exit
                    if self.evaluate() == 3:
                        return self.exit_code

                self.logger.log('train/episode_reward', episode_reward,
                                self.step)

                obs = self.env.reset()
                self.agent.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                self.train_episode += 1

                self.logger.log('train/episode', self.train_episode, self.step)

            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update
            if self.step >= self.cfg.num_seed_steps:
                self.agent.update(self.replay_buffer, self.logger, self.step)

            next_obs, reward, done, _ = self.env.step(action)

            # allow infinite bootstrap
            done = float(done)
            done_no_max = 0 if episode_step + 1 == self.env._max_episode_steps else done
            episode_reward += reward.mean()

            self.replay_buffer.add(obs, action, reward, next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1

    def save_progress(self):
        """
        Saves the model into a checkpoint
        """
        if os.path.exists(self.latest_checkpoint):
            shutil.move(self.latest_checkpoint, self.previous_checkpoint)

        torch.save({
            'episode': self.train_episode,
            'step': self.step,
            'actor_state_dict': self.agent.actor.state_dict(),
            'critic_state_dict': self.agent.critic.state_dict(),
            'alpha_state_dict': self.agent.alpha,
            'actor_lr_scheduler_dict': self.agent.actor_lr_scheduler.state_dict(),
            'critic_lr_scheduler_dict': self.agent.critic_lr_scheduler.state_dict(),
            'log_alpha_lr_scheduler_dict': self.agent.log_alpha_lr_scheduler.state_dict(),
            'actor_optimizer_dict': self.agent.actor_optimizer.state_dict(),
            'critic_optimizer_dict': self.agent.critic_optimizer.state_dict(),
            'log_alpha_optimizer_dict': self.agent.log_alpha_optimizer.state_dict(),
            'replay_buffer': self.replay_buffer,
        }, self.latest_checkpoint)

    def load_progress(self):
        checkpoint = torch.load(self.latest_checkpoint)
        self.agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.agent.alpha = checkpoint['alpha_state_dict']
        self.agent.actor_lr_scheduler.load_state_dict(checkpoint['actor_lr_scheduler_dict'])
        self.agent.critic_lr_scheduler.load_state_dict(checkpoint['critic_lr_scheduler_dict'])
        self.agent.log_alpha_lr_scheduler.load_state_dict(checkpoint['log_alpha_lr_scheduler_dict'])
        self.agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_dict'])
        self.agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_dict'])
        self.agent.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_dict'])
        self.replay_buffer = checkpoint['replay_buffer']
        self.train_episode = checkpoint['episode']
        self.step = checkpoint['step']

    def capture_signal(self, signal_num, frame):
        print("Received signal", signal_num)
        #self.logger.log('Received signal')

        if signal_num == signal.SIGTERM:
            self.exit_code = 3
        elif signal_num == signal.SIGALRM:
            self.exit_code = 3
        else:
            self.exit_code = 0


def yaml_to_obj(cfg_dict, node='root'):

    class CFG:
        def as_dict(self):
            return self.__dict__

    cfg = CFG()
    for k, v in cfg_dict.items():
        cfg.__dict__[k] = yaml_to_obj(v) if isinstance(v, dict) else v

    return cfg

@hydra.main(config_path='config/train.yaml', strict=True)
def main(cfg):
    #assert len(sys.argv) > 1, 'No config argument.'
    #with open(sys.argv[1]) as f:
    #    cfg = yaml_to_obj(yaml.load(f, Loader=yaml.FullLoader))
    workspace = Workspace(cfg)
    workspace.run()

    global EXIT_CODE
    EXIT_CODE = workspace.exit_code


if __name__ == '__main__':
    global EXIT_CODE
    EXIT_CODE = None
    main()

    if EXIT_CODE == 3:
        sys.exit(EXIT_CODE)
