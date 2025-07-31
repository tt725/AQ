import numpy as np
from Logger import Logger
import time
from Env import Env
from QLearner import QLearner


def QLearning():
    max_Q_S_5_repeat = np.zeros((10000, 5))
    max_Q_S_repeat = np.zeros((10000, 100))
    for repeat in range(10000):
        env = Env()
        agent = QLearner()
        max_Q_S_5, max_Q_S = QUpdate(env=env, agent=agent)
        max_Q_S_5_repeat[int(repeat)] = max_Q_S_5
        max_Q_S_repeat[int(repeat)] = max_Q_S
        if repeat % 1000 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_S_5 is: \n{}".format(list(max_Q_S_5_repeat.mean(axis=0))))
    log.logger.info("max_Q_S is: \n{}".format(list(max_Q_S_repeat.mean(axis=0))))


def QUpdate(env, agent):
    max_Q_S_5 = np.zeros(5)
    max_Q_S = np.zeros(100)
    state = env.reset()
    for step in range(10000):
        if step % 100 == 0:
            max_Q_S[step // 100] = agent.maxQ(env.STATE_S)
        if step < 5:
            max_Q_S_5[step] = agent.maxQ(env.STATE_S)
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done)
        state = next_state
    return max_Q_S_5, max_Q_S


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "Q" + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    QLearning()
