import math

import torch
import torch.nn.functional as F
import utils
from agent.sac import SACAgent


class GCNNSACAgent(SACAgent):

    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, num_nodes, use_ns_regularization,
                 ns_regularization_weight, decay_step_size=int(1e10), decay_factor=0.9):
        super().__init__(obs_dim, action_dim, action_range, device, critic_cfg, actor_cfg, discount, init_temperature,
                         alpha_lr, alpha_betas, actor_lr, actor_betas, actor_update_frequency, critic_lr, critic_betas,
                         critic_tau, critic_target_update_frequency, batch_size // int(math.sqrt(num_nodes)), learnable_temperature)

        # set the learning rate scheduler
        self.actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=decay_step_size, gamma=decay_factor)
        self.critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=decay_step_size, gamma=decay_factor)
        self.log_alpha_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.log_alpha_optimizer, step_size=decay_step_size, gamma=decay_factor)

        self.num_nodes = num_nodes
        self.use_ns_regularization = use_ns_regularization
        self.ns_regularization_weight = ns_regularization_weight

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @alpha.setter
    def alpha(self, new_alpha):
        self.log_alpha.data = torch.log(new_alpha)

    def act(self, obs, sample=False, return_ns_loss=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist, _ = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1

        if return_ns_loss:
            return utils.to_np(action[0])
        else:
            return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        bs = obs.shape[0]

        dist, ns_loss = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).view(bs, self.num_nodes, -1).sum(dim=-1)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward + (not_done.repeat(1, self.num_nodes) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_lr_scheduler.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        bs = obs.shape[0]
        dist, ns_loss = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).view(bs, self.num_nodes, -1).sum(dim=-1)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        if self.use_ns_regularization:
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean() + self.ns_regularization_weight * ns_loss
        else:
            actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        if self.use_ns_regularization:
            logger.log('train_actor/ns_loss', ns_loss, step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_scheduler.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.log_alpha_lr_scheduler.step()

        logger.log('train/learning_rate', self.actor_lr_scheduler.get_last_lr()[0], step)
