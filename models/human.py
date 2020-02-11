import numpy as np

class HUMAN:
    def __init__(self, env, params):
        '''
        Human agent in Human-AI team.
        Inputs:
        - env: environment of the world 
        '''
        self.gamma = params['gamma']
        self.k = params['k']
        self.env = env
        self.rs = []
        
    def discount_factor_function(self, t):
        if self.gamma is not None:
            return np.power(self.gamma, t)
        elif self.k is not None:
            return 1/(1+self.k*t)
        return 1

    def decide(self, x, infos):
        '''
        Decide whether to accept or override AI's recommendation
        Inputs:
        - x: observation(current state) from the environment
        - infos: AI's recommendation and associated information
        Outputs:
        - decision: decision making by human
        '''
        # recommended_action = infos['action']
        rs = infos['rs']
        # eu = infos['eu']
        vs = self._evaluate_actions(rs)
        print(vs)
        decision = self._optimal(vs)

        return decision

    def rewarded(self, r):
        self.rs.append(r)

    def _evaluate_actions(self, rs):
        nA, _ = rs.shape
        nA = self.env.nA
        vs = [0 for _ in range(nA)]
        for a in range(nA): 
            vs[a] = self.prospect(rs[a])
        return vs  

    def _optimal(self, vs):  
        qmax = np.max(vs)
        actions = []
        for i,q in enumerate(vs):
            if q == qmax:
                actions.append(i)
        return np.random.choice(actions)

    def _cummulative_utilities(self, rs, f):
        utilities = 0
        for t,r in enumerate(rs) :
            utilities += f(t)*r
        return utilities

    def prospect(self, rs=None):
        if rs is None:
            rs = self.rs
        return self._cummulative_utilities(rs, self.discount_factor_function)