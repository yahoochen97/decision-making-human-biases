import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils.config import config

MAPS = '''
xxxxxxVxxxx
xxxxxxoxxxx
xxxxxooooox
xxxxxoxxxox
xxxxxoxxxox
xxxxDoxxxox
xxxxxoxxxox
xxxxxoxxxox
xxxxxooooox
xxxxxoxxNxx
xxDoooxxxxx
xxxxxoxxxxx
xxxxxoxxxxx
'''

LEFT = config.getint('env', 'LEFT')
RIGHT = config.getint('env', 'RIGHT')
UP = config.getint('env', 'UP')
DOWN = config.getint('env', 'DOWN')

TERMINATION = config.get('env','TERMINATION').split('\n')
REWARD = config.get('env','REWARD').split('\n')
NON_TERM_REWARD = config.getfloat("env", "NON_TERM_REWARD")
REWARD = list(map(int, REWARD)) 
COLORS = config.get('env','COLORS').split('\n')
LABELS = config.get('env','LABELS').split('\n')

def build_states(maps):
    xy_to_s = {}
    s_to_xy = {}
    nS = 0
    for x,row in enumerate(maps):
        for y,letter in enumerate(row):
            if letter != 'x':
                xy_to_s[(x,y)] = nS
                s_to_xy[nS] = (x,y)
                nS += 1
    return xy_to_s, s_to_xy, nS

def gen_trans_mat(maps,xy_to_s, p=1):
    moveup = lambda x,y:(x-1,y)
    movedown = lambda x,y:(x+1,y)
    moveleft = lambda x,y:(x,y-1)
    moveright = lambda x,y:(x,y+1)
        
    def reward(x,y):
        letter = maps[x][y]
        if letter in TERMINATION:
            idx = TERMINATION.index(letter)
            return REWARD[idx], True
        else:
            return NON_TERM_REWARD, False

    def out(maps, x,y):
        height = len(maps)
        width = len(maps[0])
        if x<0 or y<0 or x>=height or y>= width:
            return True
        return False

    P = {}
    for x,row in enumerate(maps):
        for y,letter in enumerate(row):
            # TODO: Do we allow multiple destinations?
            if letter == 'x':
                continue
            s = xy_to_s[(x,y)]
            P[s] = {}
            for a in range(4):
                if letter in TERMINATION:
                    P[s][a] = (1,s,0,True)
                    continue
                item = []
                for a_ in range(4):
                    if a == UP:
                        x_,y_ = moveup(x,y)
                    elif a == DOWN:
                        x_,y_ = movedown(x,y)
                    elif a == LEFT:
                        x_,y_ = moveleft(x,y)
                    elif a == RIGHT:
                        x_,y_ = moveright(x,y)
                    if out(maps, x_, y_) or maps[x_][y_] == 'x':
                        x_,y_ = (x,y)
                    rew,done = reward(x_,y_)
                    prob = p if a_ == a else (1-p)/3
                    s_ = xy_to_s[(x_, y_)]
                    item += [(prob,s_,rew,done)]
                P[s][a] = item.copy()
    return P

def plot_env(maps, cur_xy):
    height = len(maps)
    width = len(maps[0])
    image = np.empty((height,width))
    for x,row in enumerate(maps):
        for y,letter in enumerate(row):
            if letter == 'x':
                image[x,y] = 0
            elif letter in TERMINATION:
                image[x,y] = 2 + TERMINATION.index(letter)
            # elif letter == 'N':
            #     image[x,y] = 2
            # elif letter == 'D':
            #     image[x,y] = 3
            # elif letter == 'V':
            #     image[x,y] = 4
            else:
                image[x,y] = 1
    x, y = cur_xy
    image[x, y] = 2 + len(TERMINATION)
    # colors = ['white','pink','grey','red','orange', 'blue']
    # labels = ['empty','road','Noodles','Donut','Vegetarian','Alice']
    cmap = mpl.colors.ListedColormap(COLORS) 
    plt.xticks([], [])
    plt.yticks([], [])
    img = plt.imshow(image, cmap=cmap,extent=[0,width,0,height],aspect=width/height)
    patches = [mpatches.Patch(color=img.cmap(img.norm(i)),
                label=LABELS[i]) for i in range(1,len(LABELS))]
    plt.legend(handles=patches, loc=2)
    plt.show(block=False)
    plt.pause(1)
    plt.close("all")


class ENV:
    def __init__(self, maps=MAPS, start_xy=(12,5), p=1):
        maps = maps.split('\n')[1:-1]
        self.maps = maps
        self.nA = 4
        self.xy_to_s, self.s_to_xy, self.nS = build_states(maps)
        self.P = gen_trans_mat(maps, self.xy_to_s, p=p)
        self.init_state = self.cur_state = self.xy_to_s[start_xy]

    def set_state(self, xy):
        self.cur_state = self.xy_to_s[xy]

    def reset(self):
        self.cur_state = self.init_state
        return self.cur_state

    def step(self, a):
        ts = self.P[self.cur_state][a]
        ps = [t[0] for t in ts]
        prob, new_state, r, done = ts[np.random.choice([i for i in range(4)], p=ps)]
        self.cur_state = new_state
        return new_state, r, done, {'prob': prob}

    def render(self):
        cur_xy = self.s_to_xy[self.cur_state]
        plot_env(self.maps, cur_xy)

    def close(self):
        pass

