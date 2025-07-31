import numpy as np
from Logger import Logger
import time
from Env import Env
from AdaEQLearner import AdaEQLearner

NUMBER_ESTIMATOR = 10

def AdaEQLearning():
    S_leave_P_repeat = np.zeros((1000, 100))
    for repeat in range(1000):
        env = Env()
        agent = AdaEQLearner(number_estimator=NUMBER_ESTIMATOR)
        S_leave_P = AdaEQUpdate(env=env, agent=agent)
        S_leave_P_repeat[int(repeat)] = S_leave_P
        if repeat % 100 == 0:
            log.logger.info("************************************")
            log.logger.info("repeat experiment number is {}".format(repeat))
    log.logger.info("************************************")
    log.logger.info("S_leave_P is: \n{}".format(list(S_leave_P_repeat.mean(axis=0))))


def AdaEQUpdate(env, agent):
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
        MC_reward_state_action = testQ(agent, reward, next_state, done)
        agent.learning(state, action, reward, next_state, done, MC_reward_state_action)
        state = next_state
        if step % 10000 == 0:
            S_leave_P[step // 10000] = S_leave / S_visit
    return S_leave_P


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

