import numpy as np
from numpy import array
import matplotlib as mpl
import copy, sys

def _greedy(Q,s):
    qmax = np.max(Q[s])
    actions = []
    for i,q in enumerate(Q[s]):
        if q == qmax:
            actions.append(i)
    return actions

def greedy(Q,s):
    return np.random.choice(_greedy(Q,s))

def ep_greedy(Q,s,ep):
    if np.random.rand() < ep:
        return np.random.choice(len(Q[s]))
    else:
        return greedy(Q,s)

def action(policy,s):
    nA = len(policy[s])
    if sum(policy[s]) != 1:
        p = policy[s] / sum(policy[s])
    else:
        p = policy[s]
    return np.random.choice(nA,p=p)

def qlearn(env,gamma=1,alpha=0.9,ep=0.05,steps=1000,episodes=1000):
    # np.random.seed(3)
    # env.seed(5)
    nS = env.nS
    nA = env.nA
    Q = np.zeros((nS,nA))
    min_ep = 0.01
    factor = episodes/ep*min_ep
    for episode in range(episodes):
        if episode % factor == 0: 
            ep -= min_ep
        s = env.reset()
        done = False
        step = 0
        while not done and step<steps:
            a = ep_greedy(Q,s,ep)
            ss, r, done, _ = env.step(a)
            Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[ss]) - Q[s,a])
            s = ss
            step += 1
    return Q

