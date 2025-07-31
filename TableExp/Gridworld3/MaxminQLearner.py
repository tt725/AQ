import numpy as np
from Env import Env

class MaxminQLearner:
    def __init__(self, number_estimator, epsilon=1.0, gamma=0.95, learningRate=1.0):
        self.number_estimator = number_estimator
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.init_Q_table()

    def init_Q_table(self):
        self.Q = np.random.normal(0, 0.01, size=(self.number_estimator, Env().world_size, Env().world_size, 4))
        self.Count_S_A = np.zeros(shape=(self.number_estimator, Env().world_size, Env().world_size, 4))
        self.Count_S = np.zeros(shape=(Env().world_size, Env().world_size))

    def explore(self, state):
        self.Count_S[state[0]][state[1]] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state[0]][state[1]], 0.5)
        if np.random.random() >= epsilon_temp:
            allQ = []
            for i in range(4):
                tempQ = []
                for j in range(self.number_estimator):
                    tempQ.append(self.Q[j][state[0]][state[1]][i])
                allQ.append(np.mean(tempQ))
            action = np.argmax(allQ)
        else:
            action = np.random.choice(4)
        return action

    def learning(self, state, action, reward, next_state, done):
        index = np.random.randint(0, self.number_estimator)
        self.Count_S_A[index][state[0]][state[1]][action] += 1
        Y = reward
        if not done:
            allQ = []
            for i in range(4):
                tempQ = []
                for j in range(self.number_estimator):
                    tempQ.append(self.Q[j][next_state[0]][next_state[1]][i])
                allQ.append(min(tempQ))
            Y += self.gamma * max(allQ)
        lr = self.learningRate / np.power(self.Count_S_A[index][state[0]][state[1]][action], 1.0)
        self.Q[index][state[0]][state[1]][action] += lr * (Y - self.Q[index][state[0]][state[1]][action])

