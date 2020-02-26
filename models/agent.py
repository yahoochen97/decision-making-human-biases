import numpy as np

class AGENT:
    '''
        Agent with human biases.
    '''
    def __init__(self, alpha, k, env):
        '''
        Initialization:

        Inputs:
        - alpha: alpha parameter in softmax
        - k: hyperbolic discounting factor
        - env: the environment to interact with
        '''
        self.alpha = alpha
        self.k = k
        self.env = env


    def learn(self):
        '''
        Find optimal policy according to human biases.
        '''
        pass

    def decide(self, s, d=0):
        return self._agent(s, d)

    def _expUtility(self,s,a,d):
        if s in self.env.term_s:
            return 0
        ts = self.env.P[s][a]
        ps = [t[0] for t in ts]
        _, new_s, r, _ = ts[np.random.choice([i for i in range(self.env.nA)], p=ps)]
        u = 1/(1+self.k*d) * r
        new_a = self._agent(new_s, d+1)
        return u + self._expUtility(new_s, new_a, d+1)

    def _agent(self, s, d):
        eu = [0 for _ in range(self.env.nA)]
        for a in range(self.env.nA):
            eu[a] = self._expUtility(s, a, d)

        tmp = np.exp(self.alpha*eu)
        eu /= tmp

        return np.random.choice([i for i in range(self.env.nA)], p=eu)

