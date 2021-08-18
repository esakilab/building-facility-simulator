import torch
import torch.nn as nn
import numpy as np
import math
import buffer
from time import time
from abc import ABC, abstractmethod


def caluculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True)\
        - 0.5*math.log(2*math.pi)*log_stds.size(-1)
    log_pis = gaussian_log_probs - \
        torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

    return log_pis


def reparameterize(means, log_stds):
    stds = log_stds.exp()
    noises = torch.randn_like(means)
    us = means + stds*noises
    actions = torch.tanh(us)
    log_pis = caluculate_log_pi(log_stds, noises, actions)
    return actions, log_pis


class Critic_network(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.fc1 = nn.Linear(state_shape[0] + action_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.output1 = nn.Linear(256, 1)

        self.fc3 = nn.Linear(state_shape[0] + action_shape[0], 256)
        self.fc4 = nn.Linear(256, 256)
        self.output2 = nn.Linear(256, 1)

        self.net1 = nn.Sequential(self.fc1,
                                  nn.ReLU(),
                                  self.fc2,
                                  nn.ReLU(),
                                  self.output1)

        self.net2 = nn.Sequential(self.fc3,
                                  nn.ReLU(),
                                  self.fc4,
                                  nn.ReLU(),
                                  self.output2)

    def forward(self, states, actions):
        q1 = self.net1(torch.cat([states, actions], dim=-1))

        q2 = self.net2(torch.cat([states, actions], dim=-1))

        return q1, q2


class Actor_network(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.fc1 = nn.Linear(state_shape[0], 256)
        self.fc2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, action_shape[0] * 2)
        self.net = nn.Sequential(self.fc1,
                                 nn.ReLU(),
                                 self.fc2,
                                 nn.ReLU(),
                                 self.output)

    def forward(self, states):
        actions = self.net(states).chunk(2, dim=-1)[0]
        actions = torch.tanh(actions)
        return actions

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp(-20, 2))


class Algorithm(ABC):

    def choose_action(self, state):
        """ 確率論的な行動と，その行動の確率密度の対数 \log(\pi(a|s)) を返す． """
        state = torch.tensor(state, dtype=torch.float,
                             device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()

    def exploit(self, state):
        """ 決定論的な行動を返す． """
        state = torch.tensor(state, dtype=torch.float,
                             device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    @abstractmethod
    def update(self):
        """ 1回分の学習を行う． """
        pass


class SAC(Algorithm):
    def __init__(self, state_shape, action_shape,  device,  seed=0,
                 batch_size=256, gamma=0.99, lr=3e-4, alpha=0.2, buff_size=10**4, start_steps=2*10**3, tau=5e-3, reward_scale=1.0):
        super().__init__()

        #np.random.seed(seed)
        #torch.manual_seed(seed)
        #torch.cuda.manual_seed(seed)

        self.replay_buffer = buffer.ReplayBuffer(
            buff_size=buff_size, state_shape=state_shape, action_shape=action_shape, device=device)
        self.actor = Actor_network(
            state_shape=state_shape, action_shape=action_shape).to(device)
        self.critic = Critic_network(
            state_shape=state_shape, action_shape=action_shape).to(device)
        self.critic_target = Critic_network(
            state_shape=state_shape, action_shape=action_shape).to(device).eval()

        self.critic_target.load_state_dict(self.critic.state_dict())
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr)

        self.batch_size = batch_size
        self.learning_steps = 0
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.buff_size = buff_size
        self.start_steps = start_steps
        self.tau = tau
        self.alpha = alpha
        self.reward_scale = reward_scale

    def update(self):
        states, actions, next_states, rewards, dones = self.replay_buffer.sample_buffer(
            self.batch_size)

        self.update_critic(states, actions, rewards, dones, next_states)
        self.update_actor(states)
        self.update_target()

    def update_critic(self, states, actions, rewards, dones, next_states):
        curr_qs1, curr_qs2 = self.critic(states, actions)

        with torch.no_grad():
            next_actions, log_pis = self.actor.sample(next_states)
            next_qs1, next_qs2 = self.critic_target(next_states, next_actions)
            next_qs = torch.min(next_qs1, next_qs2) - self.alpha * log_pis
        target_qs = rewards * self.reward_scale + \
            (1.0 - dones) * self.gamma * next_qs
        loss_critic1 = (curr_qs1 - target_qs).pow_(2).mean()
        loss_critic2 = (curr_qs2 - target_qs).pow_(2).mean()
        self.critic_optimizer.zero_grad()
        (loss_critic1 + loss_critic2).backward(retain_graph=False)
        self.critic_optimizer.step()

    def update_actor(self, states):
        actions, log_pis = self.actor.sample(states)
        qs1, qs2 = self.critic(states, actions)
        loss_actor = (self.alpha * log_pis - torch.min(qs1, qs2)).mean()

        self.actor_optimizer.zero_grad()
        loss_actor.backward(retain_graph=False)
        self.actor_optimizer.step()

    def update_target(self):
        for t, s in zip(self.critic_target.parameters(), self.critic.parameters()):
            t.data.mul_(1.0 - self.tau)
            t.data.add_(self.tau * s.data)
