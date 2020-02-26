import gym
import numpy as np
from models.ai import AI
from models.human import HUMAN
from models.env import ENV
from utils.config import config
from models.agent import AGENT

RE_LEN = config.getint('ai', 'RS_LEN')
STEPS = config.getint('human', 'STEPS')
HYPERBOLIC_DISCOUNT = config.getfloat('human', 'HYPERBOLIC_DISCOUNT')
EXPONENTIAL_DISCOUNT = config.getfloat('human', 'EXPONENTIAL_DISCOUNT')
HYPERBOLIC = config.getboolean('human', 'HYPERBOLIC')

# env = gym.make('FrozenLake-v0', is_slippery=False)
# env = gym.make('FrozenLake8x8-v0', is_slippery=False)
# env = gym.make('Taxi-v3')

def main():
    env = ENV()
    ai = AI(env, steps=STEPS, epsilon_exploration=0.1, Horizon=RE_LEN)
    ai.plot_value()
    params = {'gamma': None, 'k':None}
    if HYPERBOLIC:
        params['gamma'] = EXPONENTIAL_DISCOUNT
    else:
        params['k'] = HYPERBOLIC_DISCOUNT
    human = HUMAN(env, params)

    s = env.reset()
    step = 0
    while step < STEPS:
        step += 1
        env.render()
        infos = ai.recommend(s)
        a = human.decide(s, infos)
        ai.searcher.update_belief(a)
        s, r, done, _ = env.step(a)
        human.rewarded(r)
        if done:
            break
    env.close()
    print("total reward {} in {} steps.".format(np.sum(human.rs), step))

def test():
    env = ENV()
    agent = AGENT(10, 1, env)
    s = env.reset()
    step = 0
    rs = []
    while step < STEPS:
        step += 1
        env.render()
        a = agent.decide(s)
        s, r, done, _ = env.step(a)
        rs.append(r)
        if done:
            break
    env.close()
    print("total reward {} in {} steps.".format(np.sum(rs), step))

if __name__ == "__main__":
    test()
    # main()
