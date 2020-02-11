from utils.learn import qlearn, greedy
from utils.config import config
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
    def __init__(self, env, reward_seq_len=100):
        '''
        AI agent in Human-AI team.
        Inputs:
        - env: environment of the world 
        '''
        self.env = env
        self.reward_seq_len = reward_seq_len
        self._learn()

    def _learn(self):
        '''
        Learn the optimal policy for the environment
        '''
        self.Q = qlearn(self.env,gamma=GAMMA,alpha=ALPHA,ep=EPSILON,episodes=EPISODES)

    def recommend(self, x):
        '''
        Provide recommendation as well as associated information to human
        Inputs:
        - x: observation(current state) from the environment
        Outputs:
        - infos: a dictionary that contains recommendation and associated information
            - eu: expected utilities of each action
        '''
        infos = dict()
        infos['eu'] = self.Q[x]
        # infos['action'] = greedy(self.Q, x)
        infos['rs'] = self._simulate(x)

        return infos

    def _simulate(self,x):
        nA = self.env.nA
        reward_sequences = np.zeros((nA, self.reward_seq_len))
        for a in range(nA):
            s_env = deepcopy(self.env)
            ca = deepcopy(a)
            for i in range(self.reward_seq_len):
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
