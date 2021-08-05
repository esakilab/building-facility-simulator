import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, buff_size, state_shape, action_shape, device):
        self.buff_size = buff_size
        self._p = 0
        self.n = 0
        self.states = torch.empty(
            (buff_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buff_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buff_size, 1), dtype=torch.float, device=device)
        self.terminals = torch.empty(
            (buff_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buff_size, *state_shape), dtype=torch.float, device=device)

    def add(self, state, action, next_state, reward, done):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.terminals[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buff_size
        self.n = min(self.n + 1, self.buff_size)

    def sample_buffer(self, batch_size):
        idxes = np.random.randint(low=0, high=self.n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.next_states[idxes],
            self.rewards[idxes],
            self.terminals[idxes]
        )

    def train_batchs(self, batch_size):
        num = len(self)
        indices = np.random.permutation(range(num))
        for i in range(0, num, batch_size):
            j = min(num, i + batch_size)
            if (j - i) < batch_size and i != 0:
                return
            batch_size = j - i
            batch_indices = indices[i:j]
            states = self.states[batch_indices]
            actions = self.actions[batch_indices]
            next_states = self.next_states[batch_indices]
            rewards = self.rewards[batch_indices]
            yield states, actions, next_states, rewards

    def __len__(self):
        return min(self.n, self.buff_size)
