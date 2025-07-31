import numpy as np
from Env import Env
import math


class SoftAQLearner:
    def __init__(self, epsilon=1.0, gamma=0.95, learningRate=1.0, parameter=1):
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.parameter =parameter
        self.init_Q_table()

    def init_Q_table(self):
        self.Q1 = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Q2 = np.random.normal(0, 0.01, size=(Env().nState, Env().nAction))
        self.Count_S_A_1 = np.zeros(shape=(Env().nState, Env().nAction))
        self.Count_S_A_2 = np.zeros(shape=(Env().nState, Env().nAction))
        self.Count_S = np.zeros(shape=Env().nState)

    def explore(self, state):
        self.Count_S[state] += 1
        epsilon_temp = self.epsilon / np.power(self.Count_S[state], 0.5)
        action_number = Env().action_number(state)
        if np.random.random() >= epsilon_temp:
            Q3 = [(self.Q1[state][i] + self.Q2[state][i]) / 2.0 for i in range(action_number)]
            action = np.argmax(Q3[:])
        else:
            action = np.random.choice(action_number)
        return action

    def softmax1(self, next_state):
        action_number = Env().action_number(next_state)
        positive = abs(self.gamma * self.Q1[next_state][
            np.argmax(self.Q1[next_state][:action_number])]) / 200.0
        negative = abs(self.gamma * self.Q2[next_state][
            np.argmax(self.Q1[next_state][:action_number])]) / 200.0
        try:
            z_exp = [math.exp(bias * self.parameter) for bias in [positive, negative]]
            sum_z_exp = sum(z_exp)
            choose = np.random.choice([False, True], p=[i / sum_z_exp for i in z_exp])
        except Exception as e:
            if positive > negative:
                return False
            else:
                return True
        return choose

    def softmax2(self, next_state):
        action_number = Env().action_number(next_state)
        positive = abs(self.gamma * self.Q2[next_state][
            np.argmax(self.Q2[next_state][:action_number])]) / 200.0
        negative = abs(self.gamma * self.Q1[next_state][
            np.argmax(self.Q2[next_state][:action_number])]) / 200.0
        try:
            z_exp = [math.exp(bias * self.parameter) for bias in [positive, negative]]
            sum_z_exp = sum(z_exp)
            choose = np.random.choice([False, True], p=[i / sum_z_exp for i in z_exp])
        except Exception as e:
            if positive > negative:
                return False
            else:
                return True
        return choose

    def learning(self, state, action, reward, next_state, done):
        Y = reward
        if np.random.random() >= 0.5:
            self.Count_S_A_1[state][action] += 1
            lr_1 = self.learningRate / np.power(self.Count_S_A_1[state][action], 0.8)
            if not done:
                action_number = Env().action_number(next_state)
                if self.softmax1(next_state):
                    Y += self.gamma * self.Q1[next_state][np.argmax(self.Q1[next_state][:action_number])]
                else:
                    Y += self.gamma * self.Q2[next_state][np.argmax(self.Q1[next_state][:action_number])]
            self.Q1[state][action] += lr_1 * (Y - self.Q1[state][action])
        else:
            self.Count_S_A_2[state][action] += 1
            lr_2 = self.learningRate / np.power(self.Count_S_A_2[state][action], 0.8)
            if not done:
                action_number = Env().action_number(next_state)
                if self.softmax2(next_state):
                    Y += self.gamma * self.Q2[next_state][np.argmax(self.Q2[next_state][:action_number])]
                else:
                    Y += self.gamma * self.Q1[next_state][np.argmax(self.Q2[next_state][:action_number])]
            self.Q2[state][action] += lr_2 * (Y - self.Q2[state][action])
