import numpy as np

class SEARCHER:
    def __init__(self, params):
        '''
        Initialize the searcher with uniform prior.

        Inputs:
        - params: dictionary containing l(lower_bound), u(upper_bound), n(num of particles)
        '''
        l = params['l']
        u = params['u']
        n = params['n']
        N = params['N']
        self.N = N
        self.grid = np.linspace(l, u, n)
        self.prior = np.ones((n,))/n
        self.rss = None

    def choose_state(self, sars):
        '''
        Choose a state that maximizes the information gain.

        Inputs:
        - sars: state action reward sequences of shape (nS, nA, T)
        '''
        nS, _, _ = sars.shape
        EIG = [0 for _ in range(nS)]

        for s in range(nS):
            pa = self._compute_pa(sars[s])
            Ha = self._discrete_entropy(pa)
            EHa = self._expected_entropy(sars[s])
            EIG[s] = Ha - EHa

        s = self._greedy(EIG)
        self.rss = sars[s]
        return s

    def update_belief(self, a):
        '''
        Calculate posterior belief after observing decision's decision.

        Inputs:
        - a: Observed agent's decision.
        '''
        if self.rss is None:
            print("Have not picked state yet.")
            return 
        likelihoods = self._compute_likelihood(self.rss, random_flag=False)
        posterior = np.multiply(likelihoods[a], self.prior)
        tmp = np.sum(posterior)
        self.prior = posterior / tmp

    def _greedy(self, vs):
        vmax = np.max(vs)
        actions = []
        for i,v in enumerate(vs):
            if v == vmax:
                actions.append(i)

        return np.random.choice(actions)

    def _expected_entropy(self, rss):
        ks = np.random.choice(self.grid, size=self.N, replace=True, p=self.prior)
        Haks = [0 for _ in range(self.N)]

        for i,k in enumerate(ks):
            pak = self._softmax(rss, k)
            Haks[i] = self._discrete_entropy(pak)
        
        EHa = np.mean(Haks)
        return EHa     
            
    def _discrete_entropy(self, ps):
        '''
        Calculate entropy of the discrete probability distribution.

        Inputs:
        - ps: probability mass function
        '''
        lps = np.log(ps)
        entropy = -np.sum(np.multiply(ps, lps))
        return entropy

    def _compute_likelihood(self, rss, random_flag=True):
        nA, _ = rss.shape
        if random_flag:
            ks = np.random.choice(self.grid, size=self.N, replace=True, p=self.prior)
            likelihoods = np.zeros((nA, self.N))
        else:
            ks = self.grid
            likelihoods = np.zeros((nA, len(ks)))

        for i,k in enumerate(ks):
            likelihoods[:, i] = self._softmax(rss, k)
        
        return likelihoods
    
    def _compute_pa(self, rss):
        likelihoods = self._compute_likelihood(rss, random_flag=True)
        pa = np.mean(likelihoods, axis=1)
        return pa

    def _softmax(self, rss, k):
        exp_us = np.exp(self._utilities(rss, k))
        tmp = np.sum(exp_us)
        exp_us /= tmp
        return exp_us
    

    def _utilities(self, rss, k):
        us = [self._utility(rss[i], k) for i in range(len(rss))]
        return np.array(us)

    def _utility(self, rs, k):
        u = 0
        for t, r in enumerate(rs):
            u += r/(1+k*t)
        return u