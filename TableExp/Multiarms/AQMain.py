import numpy as np
from Logger import Logger
import time
from Env import Env
from AQLearner import AQLearner


def AQLearning(M, N):
    max_Q_S_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = AQLearner()
        max_Q_S= AQUpdate(env=env, agent=agent, M=M, N=N)
        max_Q_S_repeat[int(repeat)] = max_Q_S
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("max_Q_S is: \n{}".format(list(max_Q_S_repeat.mean(axis=0))))


def AQUpdate(env, agent, M, N):
    max_Q_S = np.zeros(100)
    state = env.reset()
    for step in range(10000):
        if step % 100 == 0:
            max_Q_S[step // 100] = agent.maxQ(env.STATE_S)
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done, M, N)
        state = next_state
    return max_Q_S

if __name__ == "__main__":
    for N in [1, 2, 4, 8, 16]:
        log = Logger(
            './Result/log.' + "AQ" + f" ({4},{N}) " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
            level='debug')
        AQLearning(M=4, N=N)

    for M in [1, 2, 8, 16]:
        log = Logger(
            './Result/log.' + "AQ" + f" ({M},{4}) " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
            level='debug')
        AQLearning(M=M, N=4)

