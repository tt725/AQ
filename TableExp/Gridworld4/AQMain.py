import numpy as np
from Logger import Logger
import time
from Env import Env
from AQLearner import AQLearner


def AQLearning(M, N):
    mean_reward_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = AQLearner()
        mean_reward = AQUpdate(env, agent, M, N)
        mean_reward_repeat[int(repeat)] = mean_reward
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("mean_reward is: \n{}".format(list(mean_reward_repeat.mean(axis=0))))


def AQUpdate(env, agent, M, N):
    per_reward = -1.0
    mean_reward = np.zeros(100)
    state = env.reset()
    for step in range(50000):
        if step % 500 == 0:
            mean_reward[step // 500] = per_reward
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done, M, N)
        per_reward = (per_reward * step + reward) / (step + 1)
        state = next_state
        if done:
            state = env.reset()
    return mean_reward


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

