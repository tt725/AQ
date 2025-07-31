import numpy as np
from Logger import Logger
import time
from Env import Env
from AdaEQLearner import AdaEQLearner

NUMBER_ESTIMATOR = 10

def AdaEQLearning():
    mean_reward_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = AdaEQLearner(number_estimator=NUMBER_ESTIMATOR)
        mean_reward = AdaEQUpdate(env, agent)
        mean_reward_repeat[int(repeat)] = mean_reward
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("mean_reward is: \n{}".format(list(mean_reward_repeat.mean(axis=0))))


def AdaEQUpdate(env, agent):
    per_reward = -1.0
    mean_reward = np.zeros(100)
    state = env.reset()
    for step in range(50000):
        if step % 500 == 0:
            mean_reward[step // 500] = per_reward
        action = agent.explore(state)
        next_state, reward, done = env.step(action)
        MC_reward_state_action = testQ(agent, reward, next_state, done)
        agent.learning(state, action, reward, next_state, done, MC_reward_state_action)
        per_reward = (per_reward * step + reward) / (step + 1)
        state = next_state
        if done:
            state = env.reset()
    return mean_reward


def testQ(agent, reward, next_state, done):
    max_ep_len = 10
    step = 1
    All_rewad = [reward]
    env_test = Env()
    state = env_test.state_test(next_state)
    while not (done or step == max_ep_len):
        action = agent.explore(state)
        next_state, reward, done = env_test.step(action)
        All_rewad.append(reward)
        state = next_state
        step += 1
    MC_reward_state_action = 0
    for i in reversed(All_rewad):
        MC_reward_state_action = 0.95 * MC_reward_state_action + i
    return MC_reward_state_action


if __name__ == "__main__":
    log = Logger(
        './Result/log.' + "AdaEQ" + " " + (time.strftime("%Y.%m.%d.%H.%M.%S", time.localtime())),
        level='debug')
    AdaEQLearning()

