from utils.learn import qlearn, greedy
from utils.config import config
from utils.search import SEARCHER
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from copy import deepcopy

GAMMA = config.getfloat('ai', 'GAMMA')
ALPHA = config.getfloat('ai', 'ALPHA')
EPSILON  = config.getfloat('ai', 'EPSILON')
EPISODES = config.getint('ai', 'EPISODES')


class AI:
    def __init__(self, env, steps, epsilon_exploration, Horizon=100):
        '''
        AI agent in Human-AI team.
        Inputs:
        - env: environment of the world 
        - Horizon: maximal length of reward sequences
        - steps: maximal length of sequential decision making
        - epsilon_exploration: fraction of steps using for inferring agent latent variables
        '''
        self.env = env
        self.Horizon = Horizon
        self._learn()
        params = dict()
        params['l'] = 0
        params['u'] = 1
        params['n'] = 100
        params['N'] = 1000
        self.searcher = SEARCHER(params)
        self.timestamp = 0
        self.steps = steps
        self.epsilon_exploration = epsilon_exploration

    def _learn(self):
        '''
        Learn the optimal policy for the environment
        '''
        self.Q = qlearn(self.env,gamma=GAMMA,alpha=ALPHA,ep=EPSILON,episodes=EPISODES)

    def recommend(self, cur_s):
        '''
        Provide recommendation as well as associated information to human
        Inputs:
        - cur_s: observation(current state) from the environment
        Outputs:
        - infos: a dictionary that contains recommendation and associated information
            - eu: expected utilities of each action
        '''
        if self.timestamp >= self.epsilon_exploration * self.steps:
            s = cur_s
        else:
            sars = self._generate_sars()
            s = self.searcher.choose_state(sars)
        infos = dict()
        infos['eu'] = self.Q[s]
        infos['action'] = greedy(self.Q, s)
        infos['rs'] = self._simulate(s)
        return infos

    def _generate_sars(self):
        '''
        Generate state-action rewards sequences.
        '''
        nS = self.env.nS
        nA = self.env.nA
        sars = np.zeros((nS,nA, self.Horizon))

        for s in range(nS):
            self.env.set_state_s(s)
            sars[s] = self._simulate(s)

        return sars

    def _simulate(self,x):
        nA = self.env.nA
        reward_sequences = np.zeros((nA, self.Horizon))
        for a in range(nA):
            s_env = deepcopy(self.env)
            ca = deepcopy(a)
            for i in range(self.Horizon):
                s, r, done, _ = s_env.step(ca)
                ca = greedy(self.Q, s)
                reward_sequences[a, i] = r
                if done:
                    break
        return reward_sequences

    def plot_value(self):
        maps = self.env.maps
        Vs = np.max(self.Q, axis=1)
        height = len(maps)
        width = len(maps[0])
        image = np.empty((height,width))
        for x,row in enumerate(maps):
            for y,letter in enumerate(row):
                if letter == 'x':
                    image[x,y] = -1
                else:
                    image[x,y] = 1
        cmap = mpl.colors.ListedColormap(['red','white'])
        bounds = [-1,0,1]
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        _, ax = plt.subplots()
        ax.set_xticks(range(width))
        ax.set_yticks(range(height))
        plt.imshow(image,cmap=cmap,extent=[0,width,0,height],norm=norm)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        Vs /= np.max(Vs)
        circles = []
        for s,val in enumerate(Vs):          
            x,y = self.env.s_to_xy[s]
            x = height - x - 0.5
            y = y + 0.5
            circle = plt.Circle((y,x),val**2/2.1,color='k')
            ax.add_artist(circle)
            circles.append(circle)
        plt.show()
