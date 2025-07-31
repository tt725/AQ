import numpy as np

class Env():
    def __init__(self):
        self.STATE_S = 0
        self.nState = 1
        self.nAction = 13
        self.Leave = 0
        self.state = self.STATE_S

    def reset(self):
        self.state = self.STATE_S
        return self.state

    def state_test(self, state):
        self.state = state
        return self.state

    def step(self, action):
        if action == self.Leave:
            if np.random.random() >= 19.0 / 38:
                reward = -1
            else:
                reward = 1
        elif action >=1 and action <=6:
            if np.random.random() >= 18.0 / 38:
                reward = -1
            else:
                reward = 1
        else:
            if np.random.random() >= 12.0 / 38:
                reward = -1
            else:
                reward = 2
        self.state = self.STATE_S
        return self.state, reward, False

    def action_number(self, state):
        return self.nAction