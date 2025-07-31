import numpy as np
from Logger import Logger
import time
from Env import Env
from SoftmaxQLearner import SoftmaxQLearner


def SoftmaxQLearning():
    mean_reward_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = SoftmaxQLearner()
        mean_reward = SoftmaxQUpdate(env, agent)
        mean_reward_repeat[int(repeat)] = mean_reward
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("mean_reward is: \n{}".format(list(mean_reward_repeat.mean(axis=0))))


def SoftmaxQUpdate(env, agent):
    per_reward = -1.0
    mean_reward = np.zeros(100)
    state = env.reset()
    for step in range(50000):
        if step % 500 == 0:
            mean_reward[step // 500] = per_reward
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        agent.learning(state, action, reward, next_state, done)
        per_reward = (per_reward * step + reward) / (step + 1)
        state = next_state
        if done:
            state = env.reset()
    return mean_reward


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "SoftmaxQ" + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')

    SoftmaxQLearning()
