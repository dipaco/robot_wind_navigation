import torch
import torch.nn.functional as F
from agent.sac import SACAgent


class GCNNSACAgent(SACAgent):

    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, num_nodes):
        super().__init__(obs_dim, action_dim, action_range, device, critic_cfg, actor_cfg, discount, init_temperature,
                         alpha_lr, alpha_betas, actor_lr, actor_betas, actor_update_frequency, critic_lr, critic_betas,
                         critic_tau, critic_target_update_frequency, batch_size, learnable_temperature)

        self.num_nodes = num_nodes

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        bs = obs.shape[0]

        dist = self.actor(next_obs)
        next_action = dist.rsample()
        log_prob = dist.log_prob(next_action).view(bs, self.num_nodes, -1).sum(dim=-1)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
        target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
        target_Q = reward.repeat(1, self.num_nodes) + (not_done.repeat(1, self.num_nodes) * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    def update_actor_and_alpha(self, obs, logger, step):
        bs = obs.shape[0]
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).view(bs, self.num_nodes, -1).sum(dim=-1)
        actor_Q1, actor_Q2 = self.critic(obs, action)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()