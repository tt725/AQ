import numpy as np
from Logger import Logger
import time
from Env import Env
from AQLearner import AQLearner


def AQLearning(M, N):
    S_leave_P_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = AQLearner()
        S_leave_P = AQUpdate(env, agent, M, N)
        S_leave_P_repeat[int(repeat)] = S_leave_P
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("S_leave_P is: \n{}".format(list(S_leave_P_repeat.mean(axis=0))))


def AQUpdate(env, agent, M, N):
    S_leave_P = np.zeros(100)
    state = env.reset()
    S_visit = 0.0
    S_leave = 0.0
    for step in range(1000000):
        if state == env.STATE_S:
            S_visit += 1.0
        action = agent.explore(state)
        if state == env.STATE_S and action == env.Leave:
            S_leave += 1.0
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done, M, N)
        state = next_state
        if step % 10000 == 0:
            S_leave_P[step // 10000] = S_leave / S_visit
    return S_leave_P


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

